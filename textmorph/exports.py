import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from textmorph.config import DATA_DIR


def export_result(result, file_format: str, prefix: str = "textmorph-report") -> str:
    file_format = file_format.lower().lstrip(".")
    if file_format not in {"txt", "json", "csv", "docx"}:
        raise ValueError("Choose TXT, JSON, CSV, or DOCX export.")
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "-", prefix).strip("-_") or "textmorph-report"
    export_dir = (DATA_DIR / "exports").resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    if not export_dir.is_relative_to(DATA_DIR.resolve()):
        raise ValueError("Invalid export location.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    destination = export_dir / f"{safe_prefix}-{timestamp}-{uuid4().hex[:8]}.{file_format}"
    branded = {
        "application": "TextMorph",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }

    if file_format == "json":
        destination.write_text(json.dumps(branded, ensure_ascii=False, indent=2), encoding="utf-8")
    elif file_format == "txt":
        if isinstance(result, str):
            body = result
        else:
            body = json.dumps(result, ensure_ascii=False, indent=2)
        destination.write_text(f"TextMorph\nExported: {branded['exported_at_utc']}\n\n{body}", encoding="utf-8")
    elif file_format == "csv":
        rows = result if isinstance(result, list) else [result]
        if not rows or not all(isinstance(row, dict) for row in rows):
            raise ValueError("CSV export requires a list of records or one record.")
        fieldnames = sorted({key for row in rows for key in row})
        with destination.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                })
    else:
        from docx import Document

        document = Document()
        document.add_heading("TextMorph", level=0)
        document.add_paragraph(f"Exported: {branded['exported_at_utc']}")
        document.add_paragraph(result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2))
        document.save(destination)
    return str(destination)
