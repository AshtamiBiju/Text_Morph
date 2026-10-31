import os

from textmorph.config import SUMMARY_MODELS as SUMMARIZERS
from textmorph.documents import read_document
from textmorph.nlp import (
    clean_text,
    make_chunks,
    paraphrase,
    paraphrase_one,
    readability,
    rouge_eval,
    similarity,
    split_sentences,
    strip_gutenberg,
    summarize,
    text_statistics,
    validate_text,
)
from textmorph.ui import build_app

APP_TITLE = "TextMorph"
demo = build_app()

if __name__ == "__main__":
    default_host = "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"
    host = os.environ.get("GRADIO_SERVER_NAME", default_host)
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        share=False,
    )
