import re
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from textmorph.config import RETRIEVAL_MODEL, RERANKER_MODEL
from textmorph.storage import DocumentStore, decode_embedding

_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


@lru_cache(maxsize=3)
def _embedding_model(model_id: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall(text.casefold())


def bm25_scores(query: str, passages: Sequence[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    query_tokens = set(_tokens(query))
    docs = [_tokens(passage) for passage in passages]
    if not docs or not query_tokens:
        return [0.0] * len(docs)
    document_frequency = {
        term: sum(term in set(document) for document in docs)
        for term in query_tokens
    }
    average_length = sum(map(len, docs)) / len(docs) or 1.0
    scores = []
    for doc in docs:
        frequencies = {term: doc.count(term) for term in query_tokens}
        score = 0.0
        for term, frequency in frequencies.items():
            if not frequency:
                continue
            idf = np.log(1 + (len(docs) - document_frequency[term] + 0.5) /
                         (document_frequency[term] + 0.5))
            denominator = frequency + k1 * (1 - b + b * len(doc) / average_length)
            score += float(idf * frequency * (k1 + 1) / denominator)
        scores.append(score)
    return scores


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> Dict[int, float]:
    if k < 1:
        raise ValueError("RRF constant must be positive.")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("Each ranking must have one RRF weight.")
    fused: Dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item_id in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + float(weight) / (k + rank)
    return fused


def reindex_collection(
    store: DocumentStore,
    collection_id: str,
    model_id: str = RETRIEVAL_MODEL,
    batch_size: int = 32,
    progress: Optional[Callable[[float], None]] = None,
) -> int:
    if batch_size < 1:
        raise ValueError("Embedding batch size must be positive.")
    chunks = store.get_chunks(collection_id)
    if not chunks:
        raise ValueError("There are no document chunks to index.")
    model = _embedding_model(model_id)
    store.set_collection_document_status(collection_id, "indexing")
    try:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = model.encode(
                [chunk.source_text for chunk in batch],
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            store.save_embedding_batch(
                collection_id,
                {chunk.id: vector.tolist() for chunk, vector in zip(batch, vectors)},
                model_id=model_id,
            )
            if progress:
                progress(min(1.0, (start + len(batch)) / len(chunks)))
        store.mark_collection_indexed(collection_id, model_id)
    except Exception:
        store.set_collection_document_status(collection_id, "index_failed")
        raise
    return len(chunks)


def _dense_scores(query: str, chunks, model_id: str) -> List[float]:
    vectors = [decode_embedding(chunk.embedding_json) if chunk.embedding_model == model_id else None for chunk in chunks]
    if not any(vector is not None for vector in vectors):
        return [0.0] * len(chunks)
    model = _embedding_model(model_id)
    query_vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = []
    for vector in vectors:
        if vector is None:
            scores.append(0.0)
        else:
            values = np.asarray(vector, dtype=np.float32)
            scores.append(float(np.dot(query_vector, values)))
    return scores


def retrieve(
    store: DocumentStore,
    collection_id: str,
    query: str,
    top_k: int = 6,
    candidate_k: int = 30,
    model_id: str = RETRIEVAL_MODEL,
    rerank: bool = False,
) -> List[Dict]:
    if not query or not query.strip():
        raise ValueError("Enter a question or search query.")
    if top_k < 1 or candidate_k < top_k:
        raise ValueError("Candidate count must be at least the requested result count.")
    chunks = store.get_chunks(collection_id)
    if not chunks:
        raise ValueError("This collection has no indexed documents.")
    texts = [chunk.source_text for chunk in chunks]
    lexical_scores = bm25_scores(query, texts)
    lexical_order = sorted(range(len(chunks)), key=lambda index: lexical_scores[index], reverse=True)
    dense_scores = _dense_scores(query, chunks, model_id)
    dense_order = sorted(range(len(chunks)), key=lambda index: dense_scores[index], reverse=True)
    has_dense = any(score != 0.0 for score in dense_scores)
    lexical_order = [index for index in lexical_order if lexical_scores[index] > 0][:candidate_k]
    dense_order = [index for index in dense_order if dense_scores[index] > 0.1][:candidate_k]
    rankings = [lexical_order]
    weights = [1.0]
    if has_dense and dense_order:
        rankings.append(dense_order)
        weights.append(1.0)
    fused = reciprocal_rank_fusion(rankings, weights=weights)
    candidate_ids = sorted(fused, key=fused.get, reverse=True)[:candidate_k]

    if rerank and candidate_ids:
        model = _reranker()
        pairs = [(query, chunks[index].source_text) for index in candidate_ids]
        rerank_scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
        score_by_index = {
            index: float(score) for index, score in zip(candidate_ids, rerank_scores)
        }
        candidate_ids.sort(key=score_by_index.get, reverse=True)

    results = []
    for index in candidate_ids[:top_k]:
        chunk = chunks[index]
        doc = chunk.document
        results.append({
            "document_id": doc.id,
            "document_name": doc.filename,
            "page_number": chunk.page_number,
            "paragraph_number": chunk.paragraph_number,
            "heading": chunk.heading,
            "chunk_id": chunk.id,
            "source_text": chunk.source_text,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
            "retrieval_score": float(fused[index]),
            "dense_score": dense_scores[index],
            "keyword_score": lexical_scores[index],
        })
    return results
