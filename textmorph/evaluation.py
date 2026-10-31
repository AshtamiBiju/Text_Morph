import math
from typing import Dict, Iterable, Sequence


def retrieval_metrics(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    cutoffs=(5, 10),
) -> Dict[str, float]:
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Provide at least one labeled relevant chunk to calculate retrieval metrics.")
    if not cutoffs or any(cutoff < 1 for cutoff in cutoffs):
        raise ValueError("Metric cutoffs must be positive integers.")
    metrics = {}
    for cutoff in cutoffs:
        selected = list(retrieved_ids[:cutoff])
        hits = [1 if item in relevant else 0 for item in selected]
        metrics[f"Recall@{cutoff}"] = sum(hits) / len(relevant)
        metrics[f"Precision@{cutoff}"] = sum(hits) / cutoff
        reciprocal_rank = next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
        metrics[f"MRR@{cutoff}"] = reciprocal_rank
        dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
        ideal_hits = min(len(relevant), cutoff)
        ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        metrics[f"nDCG@{cutoff}"] = dcg / ideal_dcg if ideal_dcg else 0.0
    return metrics


def citation_coverage(answer: str, cited_chunk_ids: Iterable[str], retrieved_chunk_ids: Iterable[str]) -> float:
    cited = set(cited_chunk_ids)
    retrieved = set(retrieved_chunk_ids)
    if not answer.strip() or not cited:
        return 0.0
    if not cited.issubset(retrieved):
        raise ValueError("Citation IDs must come from the retrieved evidence set.")
    return len(cited) / max(1, len(retrieved))


def evaluate_retrieval_query(store, collection_id: str, query: str, relevant_chunk_ids: Sequence[str]) -> Dict:
    from textmorph.retrieval import retrieve

    labels = [item.strip() for item in relevant_chunk_ids if item.strip()]
    if not labels:
        raise ValueError("Enter at least one labeled relevant chunk ID. Metrics are not inferred.")
    evidence = retrieve(store, collection_id, query, top_k=10, candidate_k=30)
    retrieved_ids = [item["chunk_id"] for item in evidence]
    metrics = retrieval_metrics(retrieved_ids, labels, cutoffs=(5, 10))
    result = {
        "query": query,
        "relevant_chunk_ids": labels,
        "retrieved_chunk_ids": retrieved_ids,
        "metrics": metrics,
        "note": "Metrics use only the relevance labels supplied for this query.",
    }
    store.save_analysis(collection_id, "retrieval_evaluation", "completed", result)
    return result
