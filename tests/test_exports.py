import json

import pytest

import textmorph.exports as exports


def test_json_and_csv_exports_use_unique_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(exports, "DATA_DIR", tmp_path)
    first = exports.export_result({"answer": "grounded"}, "json")
    second = exports.export_result([{"chunk_id": "a", "score": 0.8}], "csv")
    assert first != second
    from pathlib import Path

    assert json.loads(Path(first).read_text(encoding="utf-8"))["result"]["answer"] == "grounded"
    assert "chunk_id,score" in Path(second).read_text(encoding="utf-8")


def test_export_rejects_bad_formats_and_invalid_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(exports, "DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="TXT, JSON, CSV, or DOCX"):
        exports.export_result("result", "exe")
    with pytest.raises(ValueError, match="CSV export requires"):
        exports.export_result("string", "csv")
