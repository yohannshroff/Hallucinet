"""Shared path constants and logging helper used by every script in this
directory. Keeping paths here means each script can be run from any working
directory (they all resolve paths relative to the repo root, not to cwd).
"""

import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_TEXT_DIR = RAW_DIR / "text"
SOURCE_LIST_CSV = RAW_DIR / "source_list.csv"
SOURCES_MANIFEST_CSV = RAW_DIR / "sources_manifest.csv"

KG_SEED_DIR = DATA_DIR / "kg_seed"
ENTITIES_CSV = KG_SEED_DIR / "entities.csv"
RELATIONSHIPS_CSV = KG_SEED_DIR / "relationships.csv"

PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_JSONL = PROCESSED_DIR / "chunks.jsonl"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
CHUNK_EMBEDDINGS_NPY = EMBEDDINGS_DIR / "chunk_embeddings.npy"
FAISS_INDEX_DIR = PROCESSED_DIR / "faiss_index"
FAISS_INDEX_FILE = FAISS_INDEX_DIR / "index.faiss"
ID_MAP_JSON = FAISS_INDEX_DIR / "id_map.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a simple, consistent console format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
