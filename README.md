# TextMorph

TextMorph is a Gradio NLP studio with a cream grid-paper interface and black, white, and yellow styling. Its original **Summarize**, **Paraphrase**, **Compare meaning**, and **Evaluate** tabs remain available. A fifth **Document Intelligence** workspace adds persistent document collections and source-grounded analysis.

## Features

- Abstractive summarization with DistilBART and T5-small. Long inputs are split against each tokenizer's context limit and summarized hierarchically without silently truncating the combined summaries.
- T5 paraphrasing with distinct-result filtering, short-sentence support, and best-effort style prompts.
- Sentence-BERT semantic similarity and a sentence-level heatmap.
- ROUGE-1, ROUGE-2, and ROUGE-L precision, recall, and F1 against a user-provided reference.
- Text statistics and English-oriented readability estimates.
- TXT, PDF, and DOCX ingestion into named collections. Files are limited to 25 MB each and 20 documents per collection; duplicate files are detected by SHA-256.
- Configurable limits and chunk settings through environment variables; the default is 450 whitespace-delimited tokens with 80-token overlap and a 20,000-chunk collection cap.
- SQLite persistence for collections, documents, chunks, embeddings, evidence, analysis runs, and Q&A history.
- BM25 keyword search, optional Sentence-Transformer dense retrieval, and Reciprocal Rank Fusion. Embeddings are stored in SQLite and searched with a NumPy cosine-similarity adapter; FAISS is not required.
- Evidence-first document Q&A using extractive passages by default, optional local Ollama (`qwen2.5:3b`), or optional local FLAN-T5. Returned source passages are validated against stored chunk metadata.
- Optional NLI-based claim verification and cross-document conflict analysis. Numerical discrepancies are reported separately and are not treated as proof of contradiction.
- Optional spaCy dependency-based knowledge graph extraction, semantic document-version comparison, and TXT/JSON/DOCX result exports.
- Labeled retrieval evaluation (Recall@5/10, Precision@5/10, MRR, and nDCG); metrics require user-supplied relevant chunk IDs.
- CPU support. GPU is used when available by PyTorch and Sentence-Transformers.

## Setup

Python 3.11 or 3.12 is recommended for broad compatibility with the ML dependencies.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

If PowerShell blocks activation, use Command Prompt:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:7860>. The app binds to localhost by default and does not enable Gradio public sharing. To change the bind host/port, set `GRADIO_SERVER_NAME` and `GRADIO_SERVER_PORT`. Hugging Face Spaces are detected through `SPACE_ID` and bind to `0.0.0.0`.

The first use of a model downloads its weights from Hugging Face. Model download errors are shown in the relevant feature; optional models are not loaded during app startup. CPU inference works but can be slow. For a CUDA installation, follow the matching PyTorch installation instructions at <https://pytorch.org/get-started/locally/>.

## Optional features

Install optional visualization and readability dependencies:

```bash
python -m pip install -r requirements-optional.txt
python -m spacy download en_core_web_sm
```

The interactive graph requires both `pyvis` and the spaCy English model. If either is missing, graph extraction or visualization reports the missing optional requirement.

### Local Ollama answering

Install Ollama from <https://ollama.com/>, then run:

```bash
ollama pull qwen2.5:3b
ollama serve
```

Select **Ollama (local)** in **Ask Documents**. Ollama is optional and is never required to launch TextMorph. If the service is unavailable, the app returns retrieved passages in extractive mode and displays the reason.

## Model selection

| Feature | Model |
|---|---|
| Summarization | `sshleifer/distilbart-cnn-12-6`, `t5-small` |
| Paraphrasing | `Vamsi/T5_Paraphrase_Paws` |
| Similarity and document comparison | `sentence-transformers/all-MiniLM-L6-v2` |
| Document dense retrieval | `sentence-transformers/multi-qa-mpnet-base-dot-v1` |
| Candidate reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| NLI claim/conflict analysis | `cross-encoder/nli-deberta-base` |
| Optional local answer generation | Ollama `qwen2.5:3b` or `google/flan-t5-base` |

Models load lazily. Dense document indexing and cross-encoder reranking are opt-in. The NLI label mapping is read from the model configuration and checked before classification; for the configured model, the repository config maps contradiction→0, entailment→1, neutral→2.

## Data, privacy, and limits

- By default, uploaded files, the SQLite database, and exports are stored under the ignored `data/` directory. Set `TEXTMORPH_DATA_DIR` to use another local directory.
- `TEXTMORPH_MAX_FILE_SIZE_MB`, `TEXTMORPH_MAX_DOCUMENTS_PER_COLLECTION`, `TEXTMORPH_MAX_CHUNKS_PER_COLLECTION`, `TEXTMORPH_CHUNK_SIZE`, `TEXTMORPH_CHUNK_OVERLAP`, and `TEXTMORPH_RETRIEVAL_MODEL` configure ingestion limits and retrieval.
- Collections are isolated during retrieval. Removing a document or collection removes its stored file and dependent database records.
- SQLite tables are initialized with SQLAlchemy `create_all`; `PRAGMA user_version` tracks a small schema migration. There is no general migration framework yet. Back up `data/metadata.sqlite` and collection files before upgrading.
- Uploaded documents are treated as untrusted evidence. They are never executed. The local answer prompt tells the model not to follow instructions in source text, and citations are assembled from database records rather than generated by the model.
- Similarity is not proof of factual equivalence. NLI predictions and graph relations are model/parse outputs that require human review. Claim verification only checks support in the selected documents; it is not universal fact-checking.
- Conflict analysis is limited to 500 chunks per run and version comparison to 1,000 sentences per document.
- Scanned PDFs require OCR. OCR is not included.
- T5 style prompts are best-effort and may not reliably follow every requested style.
- Embedding search uses a persistent NumPy exact-search fallback rather than FAISS. BM25 is implemented locally. Large collections may need more memory and inference time, especially on CPU.
- RAG evaluation utilities require a labeled relevance set; the project does not ship benchmark labels or claim benchmark results. Ragas, SNLI/MultiNLI evaluation, and comprehensive scientific extraction benchmarks are not bundled.

## Tests

Install development dependencies and run the offline unit tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use local fixtures and mocked models; they do not download model weights. To exercise neural inference, first launch the app and use each model-dependent feature with network access for the initial model download.

## Project layout

```text
app.py
textmorph/
  config.py             Settings and model registry
  documents.py          Secure TXT/PDF/DOCX extraction and chunking
  evaluation.py         Labeled retrieval and citation metrics
  exports.py            TXT, JSON, CSV, and DOCX exports
  intelligence.py       Grounded Q&A, NLI, claims, and semantic diffs
  knowledge_graph.py    Evidence-backed graph extraction/visualization
  nlp.py                Summarization, paraphrasing, metrics, and ROUGE
  retrieval.py          BM25, dense vectors, RRF, and optional reranking
  storage.py            SQLite persistence and collection isolation
  ui.py                 Gradio interface and original hand-drawn styling
tests/
requirements.txt
requirements-optional.txt
requirements-dev.txt
```

## Deploy

Create a Hugging Face Space, choose **Gradio**, and include `app.py`, the `textmorph/` package, `requirements.txt`, and this README. Spaces bind to `0.0.0.0` automatically. Do not enable public sharing for collections containing private documents.
