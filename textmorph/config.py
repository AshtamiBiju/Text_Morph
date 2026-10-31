import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("TEXTMORPH_DATA_DIR", APP_ROOT / "data")).expanduser()
DATABASE_PATH = DATA_DIR / "metadata.sqlite"
MAX_FILE_SIZE_MB = int(os.environ.get("TEXTMORPH_MAX_FILE_SIZE_MB", "25"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_DOCUMENTS_PER_COLLECTION = int(os.environ.get("TEXTMORPH_MAX_DOCUMENTS_PER_COLLECTION", "20"))
MAX_CHUNKS_PER_COLLECTION = int(os.environ.get("TEXTMORPH_MAX_CHUNKS_PER_COLLECTION", "20000"))
DEFAULT_CHUNK_SIZE = int(os.environ.get("TEXTMORPH_CHUNK_SIZE", "450"))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("TEXTMORPH_CHUNK_OVERLAP", "80"))
if MAX_FILE_SIZE_MB < 1 or MAX_DOCUMENTS_PER_COLLECTION < 1 or MAX_CHUNKS_PER_COLLECTION < 1 or DEFAULT_CHUNK_SIZE < 1:
    raise ValueError("TextMorph size, document, and chunk limits must be positive integers.")
if DEFAULT_CHUNK_OVERLAP < 0 or DEFAULT_CHUNK_OVERLAP >= DEFAULT_CHUNK_SIZE:
    raise ValueError("TEXTMORPH_CHUNK_OVERLAP must be non-negative and smaller than chunk size.")
SUMMARY_MODELS = {
    "DistilBART": "sshleifer/distilbart-cnn-12-6",
    "T5-small": "t5-small",
}
PARAPHRASE_MODEL = "Vamsi/T5_Paraphrase_Paws"
SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_MODEL = os.environ.get(
    "TEXTMORPH_RETRIEVAL_MODEL",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-base"
FALLBACK_GENERATOR_MODEL = "google/flan-t5-base"
