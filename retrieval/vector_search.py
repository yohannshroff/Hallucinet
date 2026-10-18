"""Thin wrapper around scripts/query_index.py's FAISS search for use by
hybrid_retrieve.py. Kept separate from graph_search.py so hybrid_retrieve.py
can call both independently and merge the results.

Usage:
    python retrieval/vector_search.py "Who led the resistance at Jhansi?" --k 5
"""

import argparse
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import EMBEDDING_MODEL_NAME, FAISS_INDEX_DIR, MODEL_DEVICE, configure_runtime, get_logger  # noqa: E402
from query_index import load_index, search as faiss_search  # noqa: E402

log = get_logger("vector_search")

_model = None  # lazy-loaded singleton, same pattern as entity_extraction's spaCy model


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        configure_runtime()
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=MODEL_DEVICE)
    return _model


@lru_cache(maxsize=2)
def _load_index_cached(index_dir: str, mtime_ns: int):
    return load_index(Path(index_dir))


def get_index(index_dir: Path = FAISS_INDEX_DIR):
    """Load the FAISS index + id_map once per process, not once per query.
    Keyed on the index file's mtime so rebuilding the index is picked up
    without restarting the API."""
    index_dir = Path(index_dir)
    return _load_index_cached(str(index_dir), (index_dir / "index.faiss").stat().st_mtime_ns)


def search(query: str, index_dir: Path = FAISS_INDEX_DIR, k: int = 5) -> list:
    index, id_map = get_index(index_dir)
    return faiss_search(query, index, id_map, get_model(), k=k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--index_dir", type=Path, default=FAISS_INDEX_DIR)
    args = parser.parse_args()

    results = search(args.query, index_dir=args.index_dir, k=args.k)

    log.info(f"top {len(results)} chunk(s) for: {args.query!r}")
    for r in results:
        print(f"  score={r['score']:.3f}  {r['title']}  ({r['source_url']})")


if __name__ == "__main__":
    main()
