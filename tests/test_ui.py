from app import demo


def test_original_tabs_and_document_intelligence_are_present():
    labels = [
        item.get("props", {}).get("label")
        for item in demo.config["components"]
        if item.get("type") == "tabitem"
    ]
    assert labels[:4] == [
        "✦ Summarize",
        "✎ Paraphrase",
        "↔ Compare meaning",
        "◎ Evaluate",
    ]
    assert "✺ Document Intelligence" in labels
    assert len(demo.config["dependencies"]) > 0
