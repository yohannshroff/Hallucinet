"""Shared path constants, config, and helpers used by every script in this
repo. Keeping paths here means each script can be run from any working
directory (they all resolve paths relative to the repo root, not to cwd).
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

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

# Neo4j connection config (Week 3+), read from .env -- see docs/manual_setup_neo4j.md
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "changeme")

# Ollama connection config (Week 5+), read from .env -- see docs/manual_setup_ollama.md
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")

# Controlled vocabularies, kept in sync with docs/schema.md.
ENTITY_TYPES = {"Person", "Location", "Event", "Organization", "Cause"}
RELATION_VOCAB = {
    "led", "ruled", "fought_against", "allied_with", "killed", "executed",
    "captured", "besieged", "located_in", "part_of", "caused_by", "triggered",
    "succeeded_by", "appointed_by", "commanded", "declared_emperor_by",
    "died_during", "member_of", "disbanded_by",
}


def build_alias_index(entities_df) -> dict:
    """Map every lowercased entity name and alias to its entity_id.

    Shared by validate_kg_csv.py (spreadsheet validation) and kg/ (graph
    ingestion + entity resolution) so both use exactly the same resolution
    rules.
    """
    index = {}
    for _, row in entities_df.iterrows():
        index[row["name"].strip().lower()] = row["entity_id"]
        aliases = row.get("aliases", "")
        if aliases:
            for alias in str(aliases).split(";"):
                alias = alias.strip()
                if alias:
                    index[alias.lower()] = row["entity_id"]
    return index


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a simple, consistent console format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
