import html
import json
from pathlib import Path

import gradio as gr

from textmorph.config import MAX_FILE_SIZE_MB, SUMMARY_MODELS
from textmorph.documents import ingest_file, read_document
from textmorph.evaluation import evaluate_retrieval_query
from textmorph.exports import export_result
from textmorph.intelligence import ask_documents, compare_versions, detect_contradictions, verify_claim
from textmorph.knowledge_graph import extract_knowledge_graph, graph_evidence_markdown, graph_visualization_html
from textmorph.nlp import (
    paraphrase,
    readability,
    rouge_eval,
    similarity,
    similarity_analysis,
    summarize_timed,
    text_statistics,
)
from textmorph.retrieval import reindex_collection
from textmorph.storage import DocumentStore

CSS = """
:root {
  --tm-ink: #20201d;
  --tm-muted: #65645e;
  --tm-paper: #fffef9;
  --tm-yellow: #ffdf57;
}
.gradio-container {
  max-width: 1180px !important;
  margin: auto !important;
  color: var(--tm-ink) !important;
  font-family: "Trebuchet MS", "Comic Sans MS", cursive !important;
  background-color: var(--tm-paper) !important;
  background-image:
    linear-gradient(rgba(32,32,29,.12) 1px, transparent 1px),
    linear-gradient(90deg, rgba(32,32,29,.12) 1px, transparent 1px) !important;
  background-size: 28px 28px !important;
}
#hero {
  position: relative;
  overflow: hidden;
  background: var(--tm-paper);
  border: 2px solid var(--tm-ink);
  border-radius: 8px 14px 9px 13px;
  box-shadow: 7px 7px 0 var(--tm-ink);
  padding: 30px 34px;
  color: var(--tm-ink);
  margin: 8px 8px 25px 0;
  transform: rotate(-.35deg);
}
#hero::after {
  content: "✳";
  position: absolute;
  right: 38px;
  top: 20px;
  color: var(--tm-yellow);
  -webkit-text-stroke: 1.5px var(--tm-ink);
  font-size: 66px;
  transform: rotate(12deg);
}
#hero h1 {
  font-family: "Comic Sans MS", "Trebuchet MS", cursive;
  font-size: clamp(38px,5vw,58px);
  letter-spacing: -2px;
  margin: 0 0 8px 0;
  color: var(--tm-ink);
}
#hero p { color: var(--tm-muted); font-size: 16px; max-width: 680px; }
#hero .eyebrow {
  display: inline-block;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: var(--tm-ink);
  background: var(--tm-yellow);
  padding: 5px 9px;
  border: 1.5px solid var(--tm-ink);
  border-radius: 3px 7px 4px 6px;
  font-size: 11px;
  font-weight: 700;
  transform: rotate(-1deg);
}
.sidebar-card {
  border: 2px solid var(--tm-ink) !important;
  border-radius: 10px 7px 12px 8px !important;
  background: rgba(255,254,249,.94) !important;
  box-shadow: 4px 4px 0 var(--tm-ink);
  padding: 18px !important;
}
.gradio-container .tabs {
  border: 2px solid var(--tm-ink) !important;
  border-radius: 10px 14px 9px 12px !important;
  background: rgba(255,254,249,.96) !important;
  box-shadow: 5px 5px 0 var(--tm-ink);
  padding: 12px !important;
}
.gradio-container button.primary {
  background: var(--tm-yellow) !important;
  color: var(--tm-ink) !important;
  border: 2px solid var(--tm-ink) !important;
  border-radius: 7px 10px 6px 9px !important;
  box-shadow: 3px 3px 0 var(--tm-ink);
  font-weight: 700 !important;
  transition: transform .15s ease, box-shadow .15s ease;
}
.gradio-container button.primary:hover {
  transform: translate(2px, 2px) rotate(-.5deg);
  box-shadow: 1px 1px 0 var(--tm-ink);
}
.gradio-container textarea,
.gradio-container input,
.gradio-container [data-testid="dropdown"] {
  border: 1.5px solid var(--tm-ink) !important;
  border-radius: 7px 9px 6px 8px !important;
  background: #fff !important;
}
.gradio-container label,
.gradio-container h2,
.gradio-container h3 { color: var(--tm-ink) !important; }
.gradio-container a,
.gradio-container button[role="tab"] { color: var(--tm-ink) !important; }
.gradio-container button[role="tab"][aria-selected="true"] {
  background: var(--tm-yellow) !important;
  border: 1.5px solid var(--tm-ink) !important;
  border-radius: 6px 8px 5px 7px !important;
}
footer { display: none !important; }
"""

_store = None


def _get_store():
    global _store
    if _store is None:
        _store = DocumentStore()
    return _store


def _collection_choices():
    return [(collection.name, collection.id) for collection in _get_store().list_collections()]


def _document_choices(collection_id):
    if not collection_id:
        return []
    return [(document.filename, document.id) for document in _get_store().list_documents(collection_id)]


def _resolve_text(text, file_path):
    return read_document(file_path) if file_path else text


def ui_summarize(text, file_path, model_label, max_words):
    try:
        source = _resolve_text(text, file_path)
        summary, duration = summarize_timed(source, model_label, max_words)
        details = {
            "Input": text_statistics(source),
            "Summary": text_statistics(summary),
            "Summary readability": readability(summary),
        }
        return summary, f"Completed in {duration:.2f}s\n\n```json\n{json.dumps(details, indent=2, ensure_ascii=False)}\n```"
    except Exception as error:
        return f"Error: {error}", ""


def ui_paraphrase(text, file_path, count, style):
    try:
        result = paraphrase(_resolve_text(text, file_path), int(count), style=style)
        return result + "\n\n> Style prompts are best-effort; the selected model may not reliably follow every style."
    except Exception as error:
        return f"Error: {error}"


def ui_similarity(text_a, text_b):
    try:
        score = similarity(text_a, text_b)
        try:
            _document_score, _matrix, figure = similarity_analysis(text_a, text_b)
        except RuntimeError as error:
            return score + f"\n\n> Heatmap unavailable: {error}", None
        return score, figure
    except Exception as error:
        return f"Error: {error}", None


def ui_rouge(generated, reference):
    try:
        return rouge_eval(generated, reference)
    except Exception as error:
        return f"Error: {error}"


def ui_create_collection(name):
    try:
        collection = _get_store().create_collection(name)
        choices = _collection_choices()
        return (
            gr.Dropdown(choices=choices, value=collection.id),
            f"Created collection: {collection.name}",
            ui_collection_info(collection.id)[0],
            gr.Dropdown(choices=[]),
        )
    except Exception as error:
        return gr.Dropdown(choices=_collection_choices()), f"Error: {error}", "", gr.Dropdown(choices=[])


def ui_rename_collection(collection_id, name):
    if not collection_id:
        return gr.Dropdown(choices=_collection_choices()), "Select a collection first.", "", gr.Dropdown(choices=[])
    try:
        collection = _get_store().rename_collection(collection_id, name)
        return (
            gr.Dropdown(choices=_collection_choices(), value=collection.id),
            f"Renamed collection to {collection.name}.",
            ui_collection_info(collection.id)[0],
            gr.Dropdown(choices=_document_choices(collection.id)),
        )
    except Exception as error:
        return gr.Dropdown(choices=_collection_choices(), value=collection_id), f"Error: {error}", "", gr.Dropdown(choices=[])


def ui_delete_collection(collection_id):
    if not collection_id:
        return gr.Dropdown(choices=_collection_choices(), value=None), "Select a collection first.", "", gr.Dropdown(choices=[])
    try:
        if not _get_store().delete_collection(collection_id):
            return gr.Dropdown(choices=_collection_choices(), value=None), "Collection not found.", "", gr.Dropdown(choices=[])
        return gr.Dropdown(choices=_collection_choices(), value=None), "Collection and its stored files were deleted.", "", gr.Dropdown(choices=[])
    except Exception as error:
        return gr.Dropdown(choices=_collection_choices(), value=collection_id), f"Error: {error}", "", gr.Dropdown(choices=[])


def ui_collection_info(collection_id):
    if not collection_id:
        return "Select or create a collection.", gr.Dropdown(choices=[], value=None)
    try:
        store = _get_store()
        documents = store.list_documents(collection_id)
        chunks = store.get_chunks(collection_id)
        pages = sum(document.page_count or 0 for document in documents)
        indexed = sum(document.status == "indexed" for document in documents)
        info = (
            f"**Documents:** {len(documents)} · **Pages:** {pages} · **Chunks:** {len(chunks)} · "
            f"**Dense-indexed documents:** {indexed}\n\n"
        )
        if documents:
            info += "| Document | Pages | Chunks | Status |\n|---|---:|---:|---|\n"
            for document in documents:
                filename = html.escape(document.filename).replace("|", "&#124;")
                info += f"| {filename} | {document.page_count or '—'} | {len(document.chunks)} | {document.status} |\n"
        else:
            info += "No documents have been added to this collection."
        return info, gr.Dropdown(choices=_document_choices(collection_id), value=None)
    except Exception as error:
        return f"Error: {error}", gr.Dropdown(choices=[], value=None)


def ui_upload_documents(collection_id, file_paths, progress=gr.Progress()):
    if not collection_id:
        return "Select a collection before uploading.", "", gr.Dropdown(choices=[])
    if not file_paths:
        return "Select one or more TXT, PDF, or DOCX files.", "", gr.Dropdown(choices=_document_choices(collection_id))
    errors, added = [], []
    for index, file_path in enumerate(file_paths, 1):
        progress(index / len(file_paths), desc=f"Processing {Path(file_path).name}")
        try:
            document = ingest_file(_get_store(), collection_id, file_path)
            added.append(document.filename)
        except (OSError, ValueError, RuntimeError) as error:
            errors.append(f"{Path(file_path).name}: {error}")
    status = f"Added {len(added)} document(s): {', '.join(added)}" if added else "No documents were added."
    if errors:
        status += "\n\nCould not add:\n- " + "\n- ".join(errors)
    return status, ui_collection_info(collection_id)[0], gr.Dropdown(choices=_document_choices(collection_id))


def ui_delete_document(collection_id, document_id):
    if not collection_id or not document_id:
        return "Select a document to remove.", "Select a collection."
    try:
        deleted = _get_store().delete_document(collection_id, document_id)
        return (
            "Document removed." if deleted else "Document not found.",
            ui_collection_info(collection_id)[0],
            gr.Dropdown(choices=_document_choices(collection_id)),
        )
    except Exception as error:
        return f"Error: {error}", ui_collection_info(collection_id)[0], gr.Dropdown(choices=_document_choices(collection_id))


def ui_reindex(collection_id, progress=gr.Progress()):
    if not collection_id:
        return "Select a collection first.", "Select a collection."
    try:
        progress(0, desc="Loading the retrieval model")
        count = reindex_collection(
            _get_store(),
            collection_id,
            progress=lambda fraction: progress(fraction, desc="Embedding document chunks"),
        )
        return f"Indexed {count} chunks with the optional retrieval model.", ui_collection_info(collection_id)[0]
    except Exception as error:
        return f"Indexing failed: {error}", ui_collection_info(collection_id)[0]


def _evidence_html(evidence):
    if not evidence:
        return "<p>No validated source passages were found.</p>"
    sections = []
    for index, item in enumerate(evidence, 1):
        page = f" · page {item['page_number']}" if item["page_number"] is not None else ""
        heading = f" · {item['heading']}" if item.get("heading") else ""
        sections.append(
            "<details><summary>"
            f"[S{index}] {html.escape(item['document_name'])}{page}{heading} "
            f"· relevance {item['retrieval_score']:.4f} · chunk {html.escape(item['chunk_id'])}</summary><blockquote>"
            f"{html.escape(item['source_text'])}</blockquote></details>"
        )
    return "".join(sections)


def ui_ask(collection_id, question, provider, model, rerank):
    try:
        result = ask_documents(
            _get_store(),
            collection_id,
            question,
            provider=provider,
            model=model,
            rerank=rerank,
        )
        citations = []
        for index, item in enumerate(result["evidence"], 1):
            source_name = html.escape(item["document_name"]).replace("|", "&#124;")
            location = f", page {item['page_number']}" if item["page_number"] is not None else ""
            citations.append(f"- [S{index}] {source_name}{location}")
        citation_block = "\n\n**Sources**\n\n" + "\n".join(citations) if citations else ""
        timings = (
            f"\n\n_Retrieval: {result['retrieval_seconds']:.2f}s · "
            f"Generation: {result['generation_seconds']:.2f}s_"
        )
        note = f"\n\n> {result['provider_note']}" if result["provider_note"] else ""
        return result["answer"] + citation_block + timings + note, _evidence_html(result["evidence"]), result
    except Exception as error:
        return f"Error: {error}", "", None


def ui_verify_claim(collection_id, claim):
    try:
        result = verify_claim(_get_store(), collection_id, claim)
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as error:
        return f"Error: {error}"


def ui_find_conflicts(collection_id, progress=gr.Progress()):
    try:
        progress(0, desc="Comparing evidence from different documents")
        findings = detect_contradictions(_get_store(), collection_id)
        return json.dumps(findings, indent=2, ensure_ascii=False) if findings else "No eligible cross-document claim pairs were found."
    except Exception as error:
        return f"Analysis failed: {error}"


def ui_graph(collection_id):
    try:
        graph = extract_knowledge_graph(_get_store(), collection_id)
        table = graph_evidence_markdown(graph)
        try:
            visualization = graph_visualization_html(graph)
            note = ""
        except RuntimeError as error:
            visualization = ""
            note = f"\n\n{error}"
        return table + note, visualization
    except Exception as error:
        return f"Graph extraction failed: {error}", ""


def ui_compare_versions(file_a, file_b):
    try:
        if not file_a or not file_b:
            raise ValueError("Upload both the original and revised documents.")
        original = read_document(file_a)
        revised = read_document(file_b)
        return json.dumps(compare_versions(original, revised), indent=2, ensure_ascii=False)
    except Exception as error:
        return f"Comparison failed: {error}"


def ui_evaluate_retrieval(collection_id, question, relevant_ids):
    try:
        labels = [item.strip() for item in relevant_ids.replace(",", "\n").splitlines()]
        result = evaluate_retrieval_query(_get_store(), collection_id, question, labels)
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as error:
        return f"Evaluation failed: {error}"


def ui_export(result, file_format):
    if not result:
        raise gr.Error("Run an analysis before exporting its result.")
    return export_result(result, file_format)


def build_app():
    store = _get_store()
    initial_collections = [(item.name, item.id) for item in store.list_collections()]
    initial_collection_id = initial_collections[0][1] if initial_collections else None
    initial_info, _initial_documents = ui_collection_info(initial_collection_id)
    with gr.Blocks(
        title="TextMorph | Intelligent Text Studio",
        theme=gr.themes.Soft(primary_hue="yellow", neutral_hue="stone"),
        css=CSS,
    ) as demo:
        gr.HTML("""
        <div id="hero">
          <div class="eyebrow">A little workshop for words</div>
          <h1>Words, reimagined.</h1>
          <p>Summarize long documents, rewrite sentences and compare meaning. A thoughtful set of text tools, all on one page.</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["sidebar-card"]):
                gr.Markdown("### Your workspace\nChoose a tool below. Paste text or upload a document where supported.")
                gr.Markdown("**Supported documents**\n\nTXT · PDF · DOCX")
                gr.Markdown("**Models**\n\nDistilBART · T5 · Sentence-BERT")
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("✦ Summarize"):
                        gr.Markdown("### Turn lengthy text into a clear summary")
                        sum_input = gr.Textbox(lines=10, label="Paste your text", placeholder="Paste an article, chapter or report here...")
                        sum_file = gr.File(label="Or upload a document", file_types=[".txt", ".pdf", ".docx"], type="filepath")
                        with gr.Row():
                            sum_model = gr.Dropdown(list(SUMMARY_MODELS), value="DistilBART", label="Model")
                            sum_length = gr.Slider(60, 220, value=130, step=10, label="Maximum summary tokens")
                        sum_btn = gr.Button("Generate summary", variant="primary")
                        sum_output = gr.Markdown(label="Summary")
                        sum_details = gr.Markdown()
                        sum_btn.click(ui_summarize, [sum_input, sum_file, sum_model, sum_length], [sum_output, sum_details])

                    with gr.Tab("✎ Paraphrase"):
                        gr.Markdown("### Say it differently, keep the meaning")
                        para_input = gr.Textbox(lines=9, label="Paste your text", placeholder="Enter a paragraph or a few complete sentences...")
                        para_file = gr.File(label="Or upload a document", file_types=[".txt", ".pdf", ".docx"], type="filepath")
                        with gr.Row():
                            para_count = gr.Slider(1, 5, value=3, step=1, label="Variants per sentence")
                            para_style = gr.Dropdown(
                                ["Standard", "Formal", "Simple English", "Academic", "Creative"],
                                value="Standard",
                                label="Style (best effort)",
                            )
                        para_btn = gr.Button("Generate paraphrases", variant="primary")
                        para_output = gr.Markdown()
                        para_btn.click(ui_paraphrase, [para_input, para_file, para_count, para_style], para_output)

                    with gr.Tab("↔ Compare meaning"):
                        gr.Markdown("### How close are two pieces of text?")
                        with gr.Row():
                            text_a = gr.Textbox(lines=8, label="Text A")
                            text_b = gr.Textbox(lines=8, label="Text B")
                        sim_btn = gr.Button("Compare texts", variant="primary")
                        sim_output = gr.Markdown()
                        sim_plot = gr.Plot(label="Sentence-level similarity heatmap")
                        sim_btn.click(ui_similarity, [text_a, text_b], [sim_output, sim_plot])

                    with gr.Tab("◎ Evaluate"):
                        gr.Markdown("### Evaluate a generated summary against a human reference")
                        generated = gr.Textbox(lines=6, label="Generated summary")
                        reference = gr.Textbox(lines=6, label="Reference summary")
                        rouge_btn = gr.Button("Calculate ROUGE scores", variant="primary")
                        rouge_output = gr.Markdown()
                        rouge_btn.click(ui_rouge, [generated, reference], rouge_output)

                    with gr.Tab("✺ Document Intelligence"):
                        gr.Markdown(
                            "### A local, evidence-first document workspace\n"
                            "Uploaded files stay on this machine. Models are loaded only when their feature is used."
                        )
                        with gr.Tabs():
                            with gr.Tab("Document Workspace"):
                                with gr.Row():
                                    collection = gr.Dropdown(
                                        choices=initial_collections,
                                        label="Collection",
                                        value=initial_collection_id,
                                    )
                                    collection_name = gr.Textbox(label="Collection name", placeholder="Research papers")
                                with gr.Row():
                                    create_collection_btn = gr.Button("Create collection", variant="primary")
                                    rename_collection_btn = gr.Button("Rename selected")
                                    delete_collection_btn = gr.Button("Delete collection")
                                collection_status = gr.Markdown()
                                upload_files = gr.File(
                                    label=f"Add documents (TXT, PDF, DOCX; up to {MAX_FILE_SIZE_MB} MB each)",
                                    file_count="multiple",
                                    file_types=[".txt", ".pdf", ".docx"],
                                    type="filepath",
                                )
                                upload_btn = gr.Button("Upload and process", variant="primary")
                                upload_status = gr.Markdown()
                                collection_info = gr.Markdown(value=initial_info)
                                with gr.Row():
                                    document_to_delete = gr.Dropdown(
                                        label="Remove a document",
                                        choices=_document_choices(initial_collection_id),
                                    )
                                    delete_document_btn = gr.Button("Remove document")
                                    reindex_btn = gr.Button("Build semantic index", variant="primary")
                                reindex_status = gr.Markdown()

                            with gr.Tab("Ask Documents"):
                                question = gr.Textbox(lines=3, label="Question", placeholder="What evidence do the documents provide?")
                                with gr.Row():
                                    provider = gr.Dropdown(
                                        ["Extractive evidence", "Ollama (local)", "FLAN-T5 (local)"],
                                        value="Extractive evidence",
                                        label="Answer mode",
                                    )
                                    ollama_model = gr.Textbox(value="qwen2.5:3b", label="Ollama model")
                                    use_reranker = gr.Checkbox(value=False, label="Use optional Cross-Encoder reranker")
                                ask_btn = gr.Button("Ask this collection", variant="primary")
                                answer_output = gr.Markdown()
                                evidence_output = gr.HTML()
                                answer_state = gr.State()
                                with gr.Row():
                                    export_format = gr.Dropdown(["txt", "json", "docx"], value="json", label="Export format")
                                    export_btn = gr.Button("Export latest answer")
                                    export_file = gr.File(label="Download")
                                ask_btn.click(
                                    ui_ask,
                                    [collection, question, provider, ollama_model, use_reranker],
                                    [answer_output, evidence_output, answer_state],
                                )
                                export_btn.click(ui_export, [answer_state, export_format], export_file)

                            with gr.Tab("Find Conflicts"):
                                gr.Markdown("NLI classifications are model estimates, not proof. Numerical differences are reported separately from contradictions.")
                                conflict_btn = gr.Button("Compare claims across documents", variant="primary")
                                conflict_output = gr.Code(language="json")
                                conflict_btn.click(ui_find_conflicts, [collection], conflict_output)

                            with gr.Tab("Knowledge Graph"):
                                graph_btn = gr.Button("Extract explicit relationships", variant="primary")
                                graph_output = gr.Markdown()
                                graph_visual = gr.HTML()
                                graph_btn.click(ui_graph, [collection], [graph_output, graph_visual])

                            with gr.Tab("Compare Versions"):
                                with gr.Row():
                                    version_a = gr.File(label="Original document", file_types=[".txt", ".pdf", ".docx"], type="filepath")
                                    version_b = gr.File(label="Revised document", file_types=[".txt", ".pdf", ".docx"], type="filepath")
                                compare_btn = gr.Button("Compare document versions", variant="primary")
                                compare_output = gr.Code(language="json")
                                compare_btn.click(ui_compare_versions, [version_a, version_b], compare_output)

                            with gr.Tab("Verify Claims"):
                                claim = gr.Textbox(lines=3, label="Claim to check against this collection")
                                verify_btn = gr.Button("Check evidence", variant="primary")
                                claim_output = gr.Code(language="json")
                                verify_btn.click(ui_verify_claim, [collection, claim], claim_output)

                            with gr.Tab("Evaluate Retrieval"):
                                gr.Markdown("Supply labeled relevant chunk IDs from your own judged examples; no benchmark labels or scores are invented.")
                                eval_question = gr.Textbox(lines=2, label="Evaluation query")
                                relevant_ids = gr.Textbox(
                                    lines=3,
                                    label="Relevant chunk IDs (comma- or newline-separated)",
                                )
                                eval_btn = gr.Button("Calculate retrieval metrics", variant="primary")
                                eval_output = gr.Code(language="json")
                                eval_btn.click(
                                    ui_evaluate_retrieval,
                                    [collection, eval_question, relevant_ids],
                                    eval_output,
                                )

                        collection.change(ui_collection_info, collection, [collection_info, document_to_delete])
                        create_collection_btn.click(
                            ui_create_collection,
                            collection_name,
                            [collection, collection_status, collection_info, document_to_delete],
                        )
                        rename_collection_btn.click(
                            ui_rename_collection,
                            [collection, collection_name],
                            [collection, collection_status, collection_info, document_to_delete],
                        )
                        delete_collection_btn.click(
                            ui_delete_collection,
                            collection,
                            [collection, collection_status, collection_info, document_to_delete],
                        )
                        upload_btn.click(
                            ui_upload_documents,
                            [collection, upload_files],
                            [upload_status, collection_info, document_to_delete],
                        )
                        delete_document_btn.click(
                            ui_delete_document,
                            [collection, document_to_delete],
                            [upload_status, collection_info, document_to_delete],
                        )
                        reindex_btn.click(ui_reindex, collection, [reindex_status, collection_info])

        gr.Markdown(
            "---\n"
            "<center>TextMorph · Built with Transformers, Sentence-BERT and Gradio</center>\n\n"
            "*Generated text may omit or change details. Similarity scores do not prove factual correctness. "
            "Scanned PDFs require OCR before text can be extracted.*"
        )
    return demo


demo = build_app()
