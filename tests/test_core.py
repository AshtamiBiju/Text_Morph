import pytest

from textmorph.documents import chunk_sections, extract_document, ingest_file, read_document
from textmorph.evaluation import citation_coverage, retrieval_metrics
from textmorph.intelligence import ask_documents, compare_versions, detect_contradictions, verify_claim, validate_evidence
from textmorph.nlp import clean_text, readability, rouge_eval, split_sentences, text_statistics
from textmorph.retrieval import bm25_scores, reciprocal_rank_fusion, retrieve
from textmorph.storage import DocumentStore


def test_cleaning_preserves_paragraphs_and_unicode():
    assert clean_text("  café\x00  test \r\n\r\n\r\nsecond  ") == "café test\n\nsecond"


def test_sentence_splitter_keeps_short_sentences_and_skips_chapter_headings():
    assert split_sentences("Chapter 2\nYes. This is a longer sentence!") == [
        "Yes.",
        "This is a longer sentence!",
    ]


def test_short_sentence_paraphrase_and_duplicate_filter(monkeypatch):
    import textmorph.nlp as nlp

    class FakeBatch(dict):
        def to(self, _device):
            return self

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return FakeBatch(input_ids=[[1, 2]])

        def batch_decode(self, outputs, skip_special_tokens=True):
            return ["Yes.", "Indeed.", "Indeed."]

    class FakeModel:
        def generate(self, **_kwargs):
            return [1, 2, 3]

    monkeypatch.setattr(nlp, "load_seq2seq", lambda _model_id: (FakeTokenizer(), FakeModel()))
    assert nlp.paraphrase_one("Yes.", count=3) == ["Indeed."]


def test_long_summarization_uses_chunks_without_truncating_combined_input(monkeypatch):
    import textmorph.nlp as nlp

    class FakeTokenizer:
        model_max_length = 10

        def __call__(self, text, add_special_tokens=False, **_kwargs):
            return {"input_ids": text.replace("summarize:", "").split()}

    input_chunks = []
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(nlp, "load_seq2seq", lambda _model_id: (tokenizer, object()))

    def fake_summarize_one(text, _model_id, _tokenizer, _model, _max_words):
        input_chunks.append(text)
        return "short summary"

    monkeypatch.setattr(nlp, "_summarize_one", fake_summarize_one)
    text = (
        "Alpha fact matters. Beta result follows. Gamma evidence supports. "
        "Delta finding appears. Epsilon report confirms."
    )
    assert nlp.summarize(text, "DistilBART", 60) == "short summary"
    initial_chunks = input_chunks[:3]
    assert len(initial_chunks) == 3
    assert all(term in " ".join(initial_chunks) for term in ("Alpha", "Beta", "Gamma", "Delta", "Epsilon"))


def test_text_metrics_and_readability_handle_short_text():
    metrics = text_statistics("A short sentence.\n\nA second one.")
    assert metrics["Words"] == 6
    assert metrics["Sentences"] == 2
    assert metrics["Paragraphs"] == 2
    assert "Flesch Reading Ease" in readability("A short sentence.")


def test_rouge_reports_precision_recall_and_f1():
    result = rouge_eval("A cat sat on the mat.", "A cat sat on the mat.")
    assert "| ROUGE1 | 1.0000 | 1.0000 | 1.0000 |" in result
    assert "Precision" in result and "Recall" in result and "F1" in result


def test_chunk_metadata_preserves_page_paragraph_and_offsets():
    chunks = chunk_sections(
        [{"text": "One two three four five six seven.", "page_number": 2, "paragraph_number": 8}],
        chunk_size=4,
        overlap=1,
    )
    assert len(chunks) == 2
    assert all(chunk["page_number"] == 2 for chunk in chunks)
    assert all(chunk["paragraph_number"] == 8 for chunk in chunks)
    assert chunks[0]["start_offset"] == 0
    assert chunks[1]["start_offset"] > chunks[0]["start_offset"]
    assert "seven." in chunks[-1]["source_text"]


def test_txt_extraction_and_docx_heading_metadata():
    sections, pages = extract_document(b"First paragraph.\n\nSecond paragraph.", ".txt")
    assert pages is None
    assert len(sections) == 2
    assert sections[0]["paragraph_number"] == 1
    assert sections[1]["paragraph_number"] == 2
    assert sections[1]["start_offset"] > sections[0]["start_offset"]
    assert sections[1]["text"] == "Second paragraph."

    docx = pytest.importorskip("docx")
    from io import BytesIO

    document = docx.Document()
    document.add_heading("Study findings", level=1)
    document.add_paragraph("The study evaluated a model.")
    buffer = BytesIO()
    document.save(buffer)
    sections, _ = extract_document(buffer.getvalue(), ".docx")
    assert sections[0]["heading"] == "Study findings"
    assert sections[0]["text"] == "Study findings"
    assert sections[1]["heading"] == "Study findings"


def test_scanned_and_corrupt_pdfs_are_reported():
    from io import BytesIO
    from pypdf import PdfWriter
    from pypdf.errors import PdfReadError

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    with pytest.raises(ValueError, match="OCR is required"):
        extract_document(buffer.getvalue(), ".pdf")
    with pytest.raises(PdfReadError):
        extract_document(b"not a pdf", ".pdf")


def test_bm25_and_rrf_rank_relevant_passage():
    scores = bm25_scores(
        "neural retrieval",
        ["A neural model improves retrieval.", "The sky is clear."],
    )
    assert scores[0] > scores[1]
    fused = reciprocal_rank_fusion([[10, 20], [20, 10]], k=60)
    assert fused[10] == fused[20]
    assert fused[10] > 0


def test_retrieval_metrics_require_labeled_relevance():
    metrics = retrieval_metrics(["a", "x", "b"], {"a", "b"}, cutoffs=(2, 3))
    assert metrics["Recall@2"] == 0.5
    assert metrics["Precision@3"] == pytest.approx(2 / 3)
    assert metrics["MRR@3"] == 1
    assert 0 < metrics["nDCG@3"] <= 1
    assert citation_coverage("Grounded answer", ["a"], ["a", "b"]) == 0.5
    with pytest.raises(ValueError, match="retrieved evidence"):
        citation_coverage("Grounded answer", ["unknown"], ["a"])


def test_collection_ingestion_search_evidence_and_persistence(tmp_path):
    store = DocumentStore(tmp_path)
    collection = store.create_collection("Research")
    source = tmp_path / "source.txt"
    source.write_text(
        "A neural retrieval system ranks passages using semantic relevance. "
        "Keyword search can retrieve exact terms from research documents.",
        encoding="utf-8",
    )
    document = ingest_file(store, collection.id, str(source))
    assert document.status == "needs_embedding"
    assert store.list_documents(collection.id)[0].filename == "source.txt"

    with pytest.raises(ValueError, match="already in this collection"):
        ingest_file(store, collection.id, str(source))

    results = retrieve(store, collection.id, "neural retrieval", top_k=2, candidate_k=30)
    assert results
    assert results[0]["document_id"] == document.id
    assert results[0]["page_number"] is None
    assert results[0]["start_offset"] < results[0]["end_offset"]
    assert validate_evidence(store, collection.id, results) == results

    response = ask_documents(store, collection.id, "What ranks passages?")
    assert "passages" in response["answer"]
    assert response["evidence"][0]["chunk_id"] == results[0]["chunk_id"]
    assert store.delete_document(collection.id, document.id)
    assert store.get_chunks(collection.id) == []
    assert store.document_count(collection.id) == 0
    store.close()

    restarted = DocumentStore(tmp_path)
    assert restarted.get_collection(collection.id).name == "Research"
    assert restarted.list_documents(collection.id) == []


def test_collection_isolation_and_delete(tmp_path):
    store = DocumentStore(tmp_path)
    first = store.create_collection("First")
    second = store.create_collection("Second")
    file_path = tmp_path / "data.txt"
    file_path.write_text("A document stored only in the first collection.", encoding="utf-8")
    ingest_file(store, first.id, str(file_path))
    assert store.get_chunks(second.id) == []
    assert store.delete_collection(first.id)
    assert store.get_collection(first.id) is None
    assert store.get_collection(second.id).name == "Second"


def test_embedding_index_persists_across_restart(tmp_path, monkeypatch):
    import numpy as np
    import textmorph.retrieval as retrieval

    class FakeEmbeddingModel:
        def encode(self, texts, **_kwargs):
            return np.array(
                [[1.0, 0.0] if "solar" in text.casefold() else [0.0, 1.0] for text in texts],
                dtype=np.float32,
            )

    monkeypatch.setattr(retrieval, "_embedding_model", lambda _model_id: FakeEmbeddingModel())
    store = DocumentStore(tmp_path)
    collection = store.create_collection("Indexed")
    source = tmp_path / "solar.txt"
    source.write_text("Solar panels convert sunlight into electricity.", encoding="utf-8")
    document = ingest_file(store, collection.id, str(source))
    assert retrieval.reindex_collection(store, collection.id, model_id="test-model") == 1
    assert store.list_documents(collection.id)[0].status == "indexed"
    store.close()

    reopened = DocumentStore(tmp_path)
    results = retrieval.retrieve(
        reopened,
        collection.id,
        "solar energy",
        top_k=1,
        candidate_k=30,
        model_id="test-model",
    )
    assert results[0]["document_id"] == document.id


def test_optional_cross_encoder_reranks_retrieved_candidates(tmp_path, monkeypatch):
    import textmorph.retrieval as retrieval

    class FakeReranker:
        def predict(self, pairs, **_kwargs):
            return [0.1, 0.9][:len(pairs)]

    monkeypatch.setattr(retrieval, "_reranker", lambda: FakeReranker())
    store = DocumentStore(tmp_path)
    collection = store.create_collection("Reranking")
    for filename, text in (
        ("first.txt", "Neural search is useful for ranking documents."),
        ("second.txt", "Neural retrieval finds relevant passages."),
    ):
        path = tmp_path / filename
        path.write_text(text, encoding="utf-8")
        ingest_file(store, collection.id, str(path))
    initial = retrieval.retrieve(store, collection.id, "neural", top_k=2, candidate_k=30)
    reranked = retrieval.retrieve(store, collection.id, "neural", top_k=2, candidate_k=30, rerank=True)
    assert len(initial) == len(reranked) == 2
    assert reranked[0]["document_name"] == initial[1]["document_name"]


def test_claim_verification_uses_real_retrieved_evidence(tmp_path, monkeypatch):
    import textmorph.intelligence as intelligence

    store = DocumentStore(tmp_path)
    collection = store.create_collection("Claims")
    path = tmp_path / "study.txt"
    path.write_text("The solar unit achieved 50 percent efficiency during the laboratory test.", encoding="utf-8")
    ingest_file(store, collection.id, str(path))
    monkeypatch.setattr(
        intelligence,
        "classify_nli",
        lambda _premise, _hypothesis: {"entailment": 0.9, "contradiction": 0.02, "neutral": 0.08},
    )
    result = verify_claim(
        store,
        collection.id,
        "The solar unit achieved 50 percent efficiency.",
    )
    assert result["status"] == "Supported"
    assert result["evidence_coverage"] == 1
    assert result["evidence"][0]["chunk_id"]


def test_numerical_discrepancies_are_not_called_contradictions(tmp_path, monkeypatch):
    import numpy as np
    import textmorph.intelligence as intelligence

    class FakeEmbedder:
        def encode(self, texts, **_kwargs):
            return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

    store = DocumentStore(tmp_path)
    collection = store.create_collection("Metrics")
    for filename, text in (
        ("report-a.txt", "The model achieved 92% accuracy on Dataset A."),
        ("report-b.txt", "The model achieved 85% accuracy on Dataset A."),
    ):
        path = tmp_path / filename
        path.write_text(text, encoding="utf-8")
        ingest_file(store, collection.id, str(path))
    monkeypatch.setattr(intelligence, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(
        intelligence,
        "classify_nli",
        lambda _premise, _hypothesis: {"entailment": 0.1, "contradiction": 0.1, "neutral": 0.8},
    )
    findings = detect_contradictions(store, collection.id, similarity_threshold=0.5)
    assert len(findings) == 1
    assert findings[0]["classification"] == "Numerical discrepancy"
    assert findings[0]["chunk_id_a"] and findings[0]["chunk_id_b"]


def test_upload_rejects_unsupported_and_oversized_files(tmp_path):
    store = DocumentStore(tmp_path)
    collection = store.create_collection("Files")
    bad_file = tmp_path / "notes.md"
    bad_file.write_text("Some content.", encoding="utf-8")
    with pytest.raises(ValueError, match="TXT, PDF, or DOCX"):
        ingest_file(store, collection.id, str(bad_file))
    too_large = tmp_path / "large.txt"
    too_large.write_bytes(b"0123456789")
    with pytest.raises(ValueError, match="no larger"):
        ingest_file(store, collection.id, str(too_large), max_size_bytes=5)


def test_short_document_is_available_for_paraphrasing(tmp_path):
    source = tmp_path / "short.txt"
    source.write_text("Yes.", encoding="utf-8")
    assert read_document(str(source)) == "Yes."


def test_claim_free_version_diff_reports_added_removed(monkeypatch):
    import textmorph.intelligence as intelligence

    class FakeEmbedder:
        def encode(self, sentences, convert_to_numpy=True, normalize_embeddings=True):
            vectors = []
            for sentence in sentences:
                token = sentence.split()[0].casefold()
                vectors.append({
                    "same": [1.0, 0.0, 0.0],
                    "old": [0.0, 1.0, 0.0],
                    "new": [0.0, 0.0, 1.0],
                }[token])
            return __import__("numpy").array(vectors, dtype=float)

    monkeypatch.setattr(intelligence, "get_embedder", lambda: FakeEmbedder())
    changes = compare_versions("Same facts remain. Old sentence removed.", "Same facts remain. New sentence added.")
    classifications = [item["classification"] for item in changes]
    assert "Unchanged" in classifications
    assert "Removed" in classifications
    assert "Added" in classifications


def test_collection_duplicate_name_rejected(tmp_path):
    store = DocumentStore(tmp_path)
    store.create_collection("Notes")
    with pytest.raises(ValueError, match="already exists"):
        store.create_collection("Notes")
