import hashlib
import io
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from textmorph.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, MAX_FILE_SIZE_BYTES
from textmorph.nlp import clean_text, strip_gutenberg
from textmorph.storage import Document, DocumentStore

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
_WORD_PATTERN = re.compile(r"\S+")


def _extract_txt(raw: bytes) -> Tuple[List[Dict], Optional[int]]:
    text = raw.decode("utf-8-sig", errors="replace")
    paragraphs, current_lines, current_start, offset = [], [], None, 0

    def finish_paragraph():
        content = "".join(current_lines).rstrip("\r\n")
        return {
            "text": content,
            "page_number": None,
            "paragraph_number": len(paragraphs) + 1,
            "start_offset": current_start + len(content) - len(content.lstrip()),
        }

    for line in text.splitlines(keepends=True):
        if line.strip():
            if current_start is None:
                current_start = offset
            current_lines.append(line)
        elif current_lines:
            paragraphs.append(finish_paragraph())
            current_lines, current_start = [], None
        offset += len(line)
    if current_lines:
        paragraphs.append(finish_paragraph())
    if not paragraphs:
        paragraphs.append({
            "text": text,
            "page_number": None,
            "paragraph_number": None,
            "start_offset": 0,
        })
    return paragraphs, None


def _extract_pdf(raw: bytes) -> Tuple[List[Dict], Optional[int]]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    if reader.is_encrypted:
        raise ValueError("This PDF is password-protected and cannot be processed.")
    pages = []
    for page_number, page in enumerate(reader.pages, 1):
        pages.append({
            "text": page.extract_text() or "",
            "page_number": page_number,
            "paragraph_number": None,
        })
    if not any(item["text"].strip() for item in pages):
        raise ValueError("No selectable text was found. This may be a scanned PDF; OCR is required.")
    return pages, len(reader.pages)


def _extract_docx(raw: bytes) -> Tuple[List[Dict], Optional[int]]:
    from docx import Document as WordDocument

    doc = WordDocument(io.BytesIO(raw))
    paragraphs = []
    active_heading = None
    for paragraph_number, paragraph in enumerate(doc.paragraphs, 1):
        text = paragraph.text.strip()
        if text:
            style = getattr(paragraph.style, "name", "") or ""
            if style.lower().startswith("heading"):
                active_heading = text
            paragraphs.append({
                "text": text,
                "page_number": None,
                "paragraph_number": paragraph_number,
                "heading": active_heading,
            })
    return paragraphs, None


def extract_document(raw: bytes, suffix: str) -> Tuple[List[Dict], Optional[int]]:
    suffix = suffix.lower()
    if suffix == ".txt":
        sections = _extract_txt(raw)
    elif suffix == ".pdf":
        sections = _extract_pdf(raw)
    elif suffix == ".docx":
        sections = _extract_docx(raw)
    else:
        raise ValueError("Please upload a TXT, PDF, or DOCX file.")
    parts, page_count = sections
    for part in parts:
        part["text"] = clean_text(strip_gutenberg(part["text"]))
    if not any(part["text"] for part in parts):
        raise ValueError("The uploaded document contains no extractable text.")
    return parts, page_count


def read_document(file_path: str) -> str:
    if not file_path:
        return ""
    path = Path(file_path)
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Please upload a TXT, PDF, or DOCX file.")
    if not path.is_file() or path.stat().st_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"The document is missing or exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB file-size limit.")
    raw = path.read_bytes()
    if len(raw) > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"The document exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB file-size limit.")
    sections, _page_count = extract_document(raw, path.suffix)
    text = "\n\n".join(section["text"] for section in sections if section["text"])
    if not text.strip():
        raise ValueError("No extractable text was found. Scanned PDFs require OCR.")
    return text


def _chunk_section(
    text: str,
    start_offset: int,
    page_number: Optional[int],
    paragraph_number: Optional[int],
    size: int,
    overlap: int,
    heading: Optional[str] = None,
) -> List[Dict]:
    tokens = list(_WORD_PATTERN.finditer(text))
    if not tokens:
        return []
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        if end < len(tokens):
            target = end
            minimum = max(start + 1, end - 40)
            for candidate in range(end - 1, minimum - 1, -1):
                if re.search(r"[.!?][\"')\\]]?$", tokens[candidate].group()):
                    target = candidate + 1
                    break
            end = max(start + 1, target)
        char_start = tokens[start].start()
        char_end = tokens[end - 1].end()
        source_text = text[char_start:char_end].strip()
        if source_text:
            chunks.append({
                "source_text": source_text,
                "page_number": page_number,
                "paragraph_number": paragraph_number,
                "heading": heading,
                "start_offset": start_offset + char_start,
                "end_offset": start_offset + char_end,
            })
        if end >= len(tokens):
            break
        start = max(start + 1, end - overlap)
    return chunks


def chunk_sections(
    sections: Iterable[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict]:
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk overlap must be non-negative and smaller than chunk size.")
    chunks = []
    offset = 0
    for section in sections:
        text = section["text"]
        section_offset = section.get("start_offset", offset)
        if text:
            chunks.extend(
                _chunk_section(
                    text,
                    section_offset,
                    section.get("page_number"),
                    section.get("paragraph_number"),
                    chunk_size,
                    overlap,
                    section.get("heading"),
                )
            )
        offset = section_offset + len(text) + 1
    return chunks


def ingest_file(
    store: DocumentStore,
    collection_id: str,
    source_path: str,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> Document:
    if not source_path:
        raise ValueError("Select a document to upload.")
    original_path = Path(source_path)
    suffix = original_path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Please upload a TXT, PDF, or DOCX file.")
    if not original_path.is_file():
        raise ValueError("The selected upload is not a readable file.")
    size = original_path.stat().st_size
    if size == 0:
        raise ValueError("The selected file is empty.")
    if size > max_size_bytes:
        raise ValueError(f"Files must be no larger than {max_size_bytes // (1024 * 1024)} MB.")
    raw = original_path.read_bytes()
    if len(raw) > max_size_bytes:
        raise ValueError(f"Files must be no larger than {max_size_bytes // (1024 * 1024)} MB.")
    fingerprint = hashlib.sha256(raw).hexdigest()
    sections, page_count = extract_document(raw, suffix)
    chunks = chunk_sections(sections)
    if not chunks:
        raise ValueError("No text chunks could be created from this document.")
    collection_dir = (store.data_dir / "collections" / collection_id / "documents").resolve()
    if not collection_dir.is_relative_to(store.data_dir):
        raise ValueError("Invalid collection storage path.")
    collection_dir.mkdir(parents=True, exist_ok=True)
    document_id = str(uuid4())
    stored_path = collection_dir / f"{document_id}{suffix}"
    stored_path.write_bytes(raw)
    filename = Path(original_path.name).name[:255]
    try:
        return store.register_document(
            collection_id=collection_id,
            filename=filename,
            sha256=fingerprint,
            file_path=stored_path,
            file_type=suffix.lstrip("."),
            page_count=page_count,
            chunks=chunks,
        )
    except Exception:
        stored_path.unlink(missing_ok=True)
        raise
