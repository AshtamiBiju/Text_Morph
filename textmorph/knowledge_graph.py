from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List

import networkx as nx

from textmorph.storage import DocumentStore

_RELATIONS = {
    "use": "USES",
    "utilize": "USES",
    "employ": "USES",
    "evaluate": "EVALUATES",
    "assess": "EVALUATES",
    "test": "EVALUATES",
    "compare": "COMPARES",
    "improve": "IMPROVES",
    "enhance": "IMPROVES",
    "associate": "ASSOCIATED_WITH",
}


@lru_cache(maxsize=1)
def _load_spacy_model(model_name: str):
    import spacy

    return spacy.load(model_name)


def extract_knowledge_graph(store: DocumentStore, collection_id: str, model_name: str = "en_core_web_sm") -> Dict:
    try:
        import spacy
    except ImportError as error:
        raise RuntimeError("Graph extraction requires spaCy. Install the optional requirements and `en_core_web_sm`.") from error
    try:
        nlp = _load_spacy_model(model_name)
    except OSError as error:
        raise RuntimeError(
            f"spaCy model {model_name!r} is not installed. Run `python -m spacy download {model_name}`."
        ) from error

    edges = []
    for chunk in store.get_chunks(collection_id):
        parsed = nlp(chunk.source_text)
        for sentence in parsed.sents:
            entity_spans = list(sentence.ents)
            if sentence.root.dep_:
                try:
                    entity_spans.extend(span for span in sentence.noun_chunks if span not in entity_spans)
                except (NotImplementedError, ValueError) as error:
                    raise RuntimeError("The selected spaCy model does not provide dependency-based noun chunks.") from error
            for token in sentence:
                relationship = _RELATIONS.get(token.lemma_.casefold())
                if not relationship or token.pos_ != "VERB":
                    continue
                subject_tokens = {
                    child.i for child in token.children
                    if child.dep_ in {"nsubj", "nsubjpass", "csubj"}
                }
                object_tokens = {
                    child.i for child in token.children
                    if child.dep_ in {"dobj", "obj", "attr", "oprd"}
                }
                for prep in token.children:
                    if prep.dep_ in {"prep", "agent"}:
                        object_tokens.update(
                            child.i for child in prep.children if child.dep_ in {"pobj", "obj"}
                        )
                source = next(
                    (span for span in entity_spans if any(token_id in subject_tokens for token_id in range(span.start, span.end))),
                    None,
                )
                target = next(
                    (span for span in entity_spans if any(token_id in object_tokens for token_id in range(span.start, span.end))),
                    None,
                )
                if source is None or target is None or source.text.casefold() == target.text.casefold():
                    continue
                edges.append({
                    "source_entity": source.text.strip(),
                    "source_type": source.label_ or "CONCEPT",
                    "target_entity": target.text.strip(),
                    "target_type": target.label_ or "CONCEPT",
                    "relationship": relationship,
                    "document_id": chunk.document_id,
                    "document_name": chunk.document.filename,
                    "chunk_id": chunk.id,
                    "page_number": chunk.page_number,
                    "paragraph_number": chunk.paragraph_number,
                    "evidence_sentence": sentence.text.strip(),
                    "extraction_method": "spaCy dependency parse",
                })

    deduplicated = {}
    for edge in edges:
        key = (
            edge["source_entity"].casefold(),
            edge["relationship"],
            edge["target_entity"].casefold(),
            edge["chunk_id"],
        )
        deduplicated[key] = edge
    edges = list(deduplicated.values())
    store.save_graph(collection_id, edges)
    nodes = {}
    graph = nx.MultiDiGraph()
    for edge in edges:
        nodes[edge["source_entity"]] = edge["source_type"]
        nodes[edge["target_entity"]] = edge["target_type"]
        graph.add_node(edge["source_entity"], entity_type=edge["source_type"])
        graph.add_node(edge["target_entity"], entity_type=edge["target_type"])
        graph.add_edge(
            edge["source_entity"],
            edge["target_entity"],
            key=edge["chunk_id"],
            relationship=edge["relationship"],
            document_id=edge["document_id"],
            chunk_id=edge["chunk_id"],
            page_number=edge["page_number"],
            evidence_sentence=edge["evidence_sentence"],
            extraction_method=edge["extraction_method"],
        )
    result = {
        "nodes": [{"name": name, "type": kind} for name, kind in sorted(nodes.items())],
        "edges": edges,
        "statistics": {
            "entity_count": graph.number_of_nodes(),
            "relationship_count": graph.number_of_edges(),
            "most_connected_entities": [
                {"name": name, "degree": degree}
                for name, degree in sorted(graph.degree(), key=lambda item: (-item[1], item[0]))[:10]
            ],
        },
    }
    store.save_analysis(collection_id, "knowledge_graph", "completed", result)
    return result


def graph_visualization_html(graph: Dict, height: int = 550) -> str:
    import html

    try:
        from pyvis.network import Network
    except ImportError as error:
        raise RuntimeError("Interactive graph display requires pyvis. Install `pyvis` from requirements-optional.txt.") from error
    network = Network(height=f"{height}px", width="100%", directed=True, notebook=False)
    for node in graph["nodes"]:
        name = html.escape(node["name"])
        network.add_node(name, label=name, title=html.escape(node["type"]))
    for edge in graph["edges"]:
        network.add_edge(
            html.escape(edge["source_entity"]),
            html.escape(edge["target_entity"]),
            label=html.escape(edge["relationship"]),
            title=html.escape(f"{edge['document_name']}: {edge['evidence_sentence']}"),
        )
    with NamedTemporaryFile("w", suffix=".html", encoding="utf-8", delete=False) as file:
        path = Path(file.name)
    try:
        network.write_html(str(path), notebook=False, open_browser=False)
        document = path.read_text(encoding="utf-8")
        return document
    finally:
        path.unlink(missing_ok=True)


def graph_evidence_markdown(graph: Dict) -> str:
    import html

    if not graph["edges"]:
        return "No explicit supported relationships were found in this collection."
    lines = ["| Entity | Relationship | Entity | Source | Evidence |", "|---|---|---|---|---|"]
    for edge in graph["edges"]:
        values = [
            edge["source_entity"],
            edge["relationship"],
            edge["target_entity"],
            edge["document_name"],
            edge["evidence_sentence"],
        ]
        lines.append(
            "| " + " | ".join(
                html.escape(str(value), quote=True).replace("|", "&#124;").replace("\n", " ")
                for value in values
            ) + " |"
        )
    return "\n".join(lines)
