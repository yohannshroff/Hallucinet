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
SOURCES_MANIFEST_CSV = RAW_DIR / "sources_manifest.csv"

KG_SEED_DIR = DATA_DIR / "kg_seed"
ENTITIES_CSV = KG_SEED_DIR / "entities.csv"
RELATIONSHIPS_CSV = KG_SEED_DIR / "relationships.csv"

# Knowledge-graph build cache (see kg/find_candidates.py, verify_candidates.py,
# consolidate_triples.py). The slow LLM verification results are kept so the
# graph is built once; consolidating them into the CSVs is fast and repeatable.
KG_BUILD_DIR = DATA_DIR / "kg_build"
MENTIONS_JSON = KG_BUILD_DIR / "mentions.json"
CANDIDATES_JSONL = KG_BUILD_DIR / "candidates.jsonl"  # rule-based facts (contain book sentences -> gitignored)
ENTITY_TYPES_JSON = KG_BUILD_DIR / "entity_types.json"  # LLM-assigned type per name (names only)
COOCCURRENCE_JSONL = KG_BUILD_DIR / "cooccurrence.jsonl"  # entity pairs co-mentioned in sentences: names, counts, page; no book text
EXTRACTED_JSONL = KG_BUILD_DIR / "extracted.jsonl"  # grammar-extracted facts + NLI score: names, relations, source page; no book text

PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_JSONL = PROCESSED_DIR / "chunks.jsonl"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
CHUNK_EMBEDDINGS_NPY = EMBEDDINGS_DIR / "chunk_embeddings.npy"
FAISS_INDEX_DIR = PROCESSED_DIR / "faiss_index"
FAISS_INDEX_FILE = FAISS_INDEX_DIR / "index.faiss"
ID_MAP_JSON = FAISS_INDEX_DIR / "id_map.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Neo4j connection config (Part 3+), read from .env -- see docs/manual_setup_neo4j.md
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "changeme")

# Ollama connection config (Part 5+), read from .env -- see docs/manual_setup_ollama.md
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
# Unload the LLM from RAM after this much idle time (frees ~3 GB on an 8 GB Mac),
# and cap the reply length so a rambling answer can't run away with CPU/GPU time.
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "5m")
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "400"))
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))  # matches the server's default context

# Small helper models (embeddings, NLI, spaCy) run on CPU with few threads on purpose:
# measured on this machine, the Apple GPU gave no speedup for them (0.51s vs 0.55s for an
# 80-pair NLI batch) and would compete with Ollama for GPU memory; 2 threads cost only ~10%
# and leave cores free for Ollama and the rest of the OS.
MODEL_DEVICE = os.environ.get("HALLUCINET_DEVICE", "cpu")
MODEL_THREADS = int(os.environ.get("HALLUCINET_THREADS", "2"))

# Controlled vocabularies, kept in sync with docs/schema.md.
ENTITY_TYPES = {"Person", "Location", "Event", "Organization", "Cause"}
RELATION_VOCAB = {
    "led", "ruled", "fought_against", "allied_with", "killed", "executed",
    "captured", "besieged", "located_in", "part_of", "caused_by", "triggered",
    "succeeded_by", "appointed_by", "commanded", "declared_emperor_by",
    "died_during", "member_of", "disbanded_by",
    "associated_with",  # co-mentioned in the same sentences (see kg/extract_facts.py)
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


def configure_runtime() -> None:
    """Apply the CPU-thread cap once, before the first model is loaded."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch

        torch.set_num_threads(MODEL_THREADS)
    except ImportError:
        pass


def is_web_url(url: str) -> bool:
    """True for http(s) links; False for local file references like
    'data/raw/book.pdf#page=5' (those are citations, not clickable links)."""
    return str(url).startswith(("http://", "https://"))


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a simple, consistent console format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
