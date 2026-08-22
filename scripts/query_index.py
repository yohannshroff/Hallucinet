"""Query the FAISS index built by build_faiss_index.py -- a manual smoke
test for the Week 1-2 vector retrieval pipeline.

Usage:
    python scripts/query_index.py "Who led the resistance at Jhansi?" --k 5
"""

import argparse
import json
from pathlib import Path

from common import EMBEDDING_MODEL_NAME, FAISS_INDEX_DIR, get_logger

log = get_logger("query_index")


def load_index(index_dir: Path):
    import faiss

    index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "id_map.json", encoding="utf-8") as f:
        id_map = json.load(f)
    return index, id_map


def search(query: str, index, id_map: list, model, k: int = 5) -> list:
    import numpy as np

    query_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = id_map[idx]
        results.append(
            {
                "score": float(score),
                "doc_id": entry["doc_id"],
                "title": entry["title"],
                "source_url": entry["source_url"],
                "text": entry["text"],
            }
        )
    return results


def main():
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Question to search for")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--index_dir", type=Path, default=FAISS_INDEX_DIR)
    args = parser.parse_args()

    index, id_map = load_index(args.index_dir)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    results = search(args.query, index, id_map, model, k=args.k)

    log.info(f"Top {len(results)} results for: {args.query!r}")
    for i, r in enumerate(results, start=1):
        snippet = r["text"][:200].replace("\n", " ")
        print(f"\n[{i}] score={r['score']:.3f}  {r['title']}  ({r['source_url']})")
        print(f"    {snippet}...")


if __name__ == "__main__":
    main()
