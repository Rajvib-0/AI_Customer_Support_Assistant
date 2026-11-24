"""Central settings. Override any of them with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(os.getenv("APP_BASE_DIR", ".")).resolve()
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", BASE_DIR / "storage"))
UPLOAD_DIR = STORAGE_DIR / "uploads"
DOCS_FILE = STORAGE_DIR / "documents.json"          # extracted pages of every uploaded document
CLASSIFIER_PATH = Path(os.getenv("CLASSIFIER_PATH", BASE_DIR / "models" / "classifier.joblib"))

# Models (all free / open source)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-large")      # "google/flan-t5-base" is lighter

# RAG parameters
CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", 90))
CHUNK_MIN_WORDS = int(os.getenv("CHUNK_MIN_WORDS", 15))
TOP_K = int(os.getenv("TOP_K", 4))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", 0.30))       # below this -> "not in documents"
RELATIVE_MARGIN = float(os.getenv("RELATIVE_MARGIN", 0.10))
CONTEXT_CHUNKS = int(os.getenv("CONTEXT_CHUNKS", 3))
CURRENCY_SYMBOL = "₹"      # some PDF fonts extract the rupee sign as "I" or "■"

# Uploads
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 10))
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

# Only set CLASSIFIER_PREPROCESS=1 if your saved model was trained on text that you cleaned *outside* the pipeline.
CLASSIFIER_PREPROCESS = os.getenv("CLASSIFIER_PREPROCESS", "0") == "1"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
