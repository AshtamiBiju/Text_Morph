# TextMorph
 
### An NLP Studio for Understanding, Transforming and Analyzing Documents.
 
TextMorph is a modular NLP application that goes beyond traditional text processing. It combines transformer-based language models, semantic search, hybrid retrieval and evidence-grounded document analysis in a single workspace.
 
What started as a text summarization and paraphrasing tool has evolved into a **Document Intelligence Platform** that can process multiple documents, retrieve relevant evidence, identify potentially conflicting claims and explore relationships between concepts.
 
Built with Python and Gradio, TextMorph brings together language processing and document analysis through an interactive, locally runnable interface.
 
---
 
## Overview
 
TextMorph retains its original NLP capabilities while introducing a dedicated Document Intelligence workspace.
 
| Workspace | Capabilities |
|---|---|
| **Summarize** | Abstractive summarization with DistilBART and T5-small |
| **Paraphrase** | Sentence rewriting with multiple outputs and style options |
| **Compare Meaning** | Semantic similarity, sentence-level comparison and heatmaps |
| **Evaluate** | ROUGE-based summary evaluation |
| **Document Intelligence** | Multi-document ingestion, hybrid retrieval, grounded Q&A, contradiction analysis, knowledge graphs and semantic version comparison |
 
## Features
 
### 01. Core NLP Studio
 
**Text Summarization**
- Abstractive summarization using DistilBART and T5-small.
- Token-aware document splitting.
- Hierarchical summarization for long inputs.
- Configurable summary generation.
 
**Text Paraphrasing**
- T5-based paraphrase generation.
- Multiple distinct outputs with duplicate filtering.
- Support for short sentences.
- Best-effort writing styles: Standard, Formal, Simple English, Academic and Creative.
 
**Semantic Similarity**
- Sentence-BERT embeddings using `all-MiniLM-L6-v2`.
- Cosine similarity scoring.
- Sentence-level semantic comparison.
- Similarity matrix and heatmap visualization.
 
**Summary Evaluation**
- ROUGE-1, ROUGE-2 and ROUGE-L.
- Precision, recall and F1 scores.
- Evaluation against user-provided reference summaries.
 
**Text Analysis**
- Word, character, sentence and paragraph counts.
- Average sentence length.
- Estimated reading time.
- Flesch Reading Ease and Flesch-Kincaid Grade Level.
 
### 02. Document Workspace
 
Organize and analyze multiple documents through persistent collections.
 
- Upload PDF, DOCX and TXT files.
- Create and manage named document collections.
- Detect duplicate files using SHA-256 fingerprints.
- Extract text and preserve available page and paragraph metadata.
- Split documents into overlapping chunks.
- Generate and store document embeddings.
- View document processing status and collection statistics.
- Remove documents and rebuild indexes.
 
**Default ingestion limits**
 
| Setting | Default |
|---|---:|
| Maximum file size | 25 MB |
| Documents per collection | 20 |
| Chunk size | 450 tokens |
| Chunk overlap | 80 tokens |
| Maximum chunks per collection | 20,000 |
 
Limits and chunking settings can be configured through environment variables.
 
### 03. Evidence-Grounded Document Q&A
 
Ask questions across an entire document collection and retrieve the passages that support the answer.
 
TextMorph uses a Retrieval-Augmented Generation (RAG) pipeline combining:
 
1. Query embedding generation.
2. Dense semantic retrieval.
3. BM25 keyword retrieval.
4. Reciprocal Rank Fusion (RRF).
5. Optional Cross-Encoder reranking.
6. Evidence selection and answer generation.
7. Source citation validation.
 
**Answer generation options**
 
- Extractive mode: returns relevant passages without generative inference.
- Ollama: optional local generation using `qwen2.5:3b`.
- FLAN-T5: optional local generation using `google/flan-t5-base`.
 
Every retrieved passage retains its document name, page number, chunk ID and source text.
 
Citations are assembled and validated using stored metadata rather than trusting the language model to invent source references.
 
### 04. Contradiction and Claim Analysis
 
Use Natural Language Inference (NLI) to examine relationships between claims found across documents.
 
Capabilities include:
 
- Candidate factual sentence extraction.
- Semantic matching of potentially related claims.
- Entailment, contradiction and neutral classification.
- Numerical discrepancy detection.
- Cross-document conflict reporting.
- Evidence-backed claim verification.
 
Numerical differences are reported separately rather than automatically classified as contradictions.
 
Claim verification produces one of three outcomes:
 
- **Supported:** relevant document evidence supports the claim.
- **Contradicted:** relevant evidence conflicts with the claim.
- **Insufficient evidence:** the collection does not provide enough information to determine either outcome.
 
This is document-grounded verification, not universal fact-checking.
 
### 05. Knowledge Graph Generation
 
Extract entities and relationships from documents to build an evidence-backed knowledge graph.
 
Entity categories include:
 
- People and organizations.
- Technologies and algorithms.
- Datasets and evaluation metrics.
- Research concepts.
 
Graph capabilities:
 
- Entity extraction using spaCy.
- Dependency-based relationship extraction.
- Graph construction with NetworkX.
- Interactive visualization with PyVis.
- Entity and relationship filtering.
- Identification of frequently occurring and highly connected entities.
- Graph export.
 
Relationships retain source evidence and document references. Simple co-occurrence is distinguished from explicitly extracted relationships.
 
### 06. Semantic Document Comparison
 
Compare two document versions to identify changes in meaning rather than relying exclusively on textual differences.
 
The comparison pipeline includes:
 
- Sentence segmentation.
- Sentence embedding generation.
- Semantic sentence alignment.
- Bipartite matching.
- Numerical claim comparison.
- NLI-based meaning analysis.
 
Changes are classified as:
 
- Added.
- Removed.
- Unchanged.
- Reworded.
- Meaning modified.
- Potentially contradictory.
 
Original and revised passages are displayed alongside their similarity scores and source locations.
 
### 07. Retrieval Evaluation
 
Evaluate retrieval quality using a user-provided set of relevant chunk IDs.
 
Implemented metrics:
 
- Recall@5 and Recall@10.
- Precision@5 and Precision@10.
- Mean Reciprocal Rank (MRR).
- Normalized Discounted Cumulative Gain (nDCG).
 
Evaluation scores are calculated from supplied relevance labels. TextMorph does not ship fabricated benchmark results.
 
### 08. Export and Persistence
 
**Export formats**
- TXT
- JSON
- DOCX
 
CSV export is also available where appropriate.
 
**Persistent storage**
- SQLite database.
- SQLAlchemy data models.
- Stored document collections.
- Persistent document chunks and embeddings.
- Evidence records and analysis history.
- Collection-scoped retrieval.
 
Data is stored locally by default.
 
---
 
## System Architecture
 
TextMorph separates its interface, NLP processing, document ingestion, retrieval and intelligence services into dedicated modules.
 
```text
                     TEXTMORPH
                         |
                 Gradio Interface
                         |
          +--------------+--------------+
          |              |              |
       Core NLP      Document       Intelligence
        Studio       Workspace        Engine
          |              |              |
     Summarization   Ingestion         RAG
     Paraphrasing    Chunking          NLI
     Similarity      Metadata          Claims
     Evaluation      Embeddings        Knowledge Graph
                     Storage           Semantic Diff
                         |
                 Retrieval Engine
                         |
              +----------+----------+
              |                     |
          BM25 Search          Dense Search
              |                     |
              +----------+----------+
                         |
                 Reciprocal Rank
                    Fusion
                         |
                Cross-Encoder
                  Reranking
                         |
                 Evidence Trace
                         |
                 SQLite Storage
```
 
### Project Structure
 
```text
TextMorph/
│
├── app.py
├── requirements.txt
├── requirements-optional.txt
├── requirements-dev.txt
├── README.md
│
├── textmorph/
│   ├── config.py
│   ├── documents.py
│   ├── evaluation.py
│   ├── exports.py
│   ├── intelligence.py
│   ├── knowledge_graph.py
│   ├── nlp.py
│   ├── retrieval.py
│   ├── storage.py
│   └── ui.py
│
├── tests/
│
└── data/
    ├── collections/
    ├── exports/
    └── metadata.sqlite
```
 
---
 
## Technology Stack
 
| Category | Technologies |
|---|---|
| Language | Python |
| Frontend | Gradio |
| Deep Learning | PyTorch |
| NLP | Hugging Face Transformers |
| Sentence Embeddings | Sentence-Transformers |
| Similarity | scikit-learn, NumPy |
| Keyword Retrieval | BM25 |
| Vector Search | NumPy cosine-similarity adapter |
| Reranking | Cross-Encoder |
| Natural Language Inference | DeBERTa-based NLI |
| Knowledge Graph | spaCy, NetworkX, PyVis |
| Database | SQLite, SQLAlchemy |
| Document Processing | pypdf, python-docx |
| Evaluation | ROUGE, custom retrieval metrics |
| Visualization | Matplotlib |
 
## Model Selection
 
| Task | Model |
|---|---|
| Summarization | `sshleifer/distilbart-cnn-12-6` |
| Summarization | `t5-small` |
| Paraphrasing | `Vamsi/T5_Paraphrase_Paws` |
| Semantic Similarity | `sentence-transformers/all-MiniLM-L6-v2` |
| Dense Document Retrieval | `sentence-transformers/multi-qa-mpnet-base-dot-v1` |
| Cross-Encoder Reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| NLI Analysis | `cross-encoder/nli-deberta-base` |
| Optional Answer Generation | `qwen2.5:3b` via Ollama |
| Optional Answer Generation | `google/flan-t5-base` |
 
Models are loaded lazily to avoid unnecessary memory consumption. Dense retrieval, reranking and optional generation models do not need to load when the corresponding functionality is unused.
 
---
 
## Getting Started
 
### Requirements
 
- Python 3.11 or 3.12 recommended.
- Internet access for the initial model downloads.
- CPU-compatible environment.
- CUDA-compatible GPU is optional.
 
### Installation
 
**Windows (PowerShell)**
 
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
 
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
 
python app.py
```
 
If PowerShell blocks virtual environment activation, use Command Prompt:
 
```bat
py -3.11 -m venv .venv
.\.venv\Scripts\activate.bat
 
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
 
python app.py
```
 
**macOS / Linux**
 
```bash
python3 -m venv .venv
source .venv/bin/activate
 
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
 
python app.py
```
 
Open the application:
 
```text
http://127.0.0.1:7860
```
 
TextMorph binds to localhost by default and does not enable Gradio public sharing.
 
The first use of a model downloads its weights from Hugging Face. CPU inference is supported, although some operations may take longer.
 
For CUDA-enabled PyTorch in
has context menu

