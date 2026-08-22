"""Build a FAISS index from the chunk embeddings, plus an id_map.json so
search results can be traced back to chunk text and source URLs.

Uses a flat inner-product index (cosine similarity, since embeddings are
L2-normalized) -- the corpus is small enough that IVF/HNSW aren't needed.

Usage:
    python scripts/build_faiss_index.py
    python scripts/build_faiss_index.py --embeddings data/processed/embeddings/chunk_embeddings.npy \
        --chunks data/processed/chunks.jsonl --out_dir data/processed/faiss_index
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common import CHUNK_EMBEDDINGS_NPY, CHUNKS_JSONL, FAISS_INDEX_DIR, get_logger

log = get_logger("build_faiss_index")


def build_index(embeddings: np.ndarray):
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def main():
    import faiss

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=CHUNK_EMBEDDINGS_NPY)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_JSONL)
    parser.add_argument("--out_dir", type=Path, default=FAISS_INDEX_DIR)
    args = parser.parse_args()

    embeddings = np.load(args.embeddings)

    chunks = []
    with open(args.chunks, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    if len(chunks) != embeddings.shape[0]:
        raise SystemExit(
            f"chunk count ({len(chunks)}) doesn't match embedding count ({embeddings.shape[0]}) "
            "-- re-run build_embeddings.py after the latest clean_and_chunk.py output"
        )

    index = build_index(embeddings)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.out_dir / "index.faiss"))

    id_map = [
        {
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "title": c["title"],
            "source_url": c["source_url"],
            "text": c["text"],
        }
        for c in chunks
    ]
    with open(args.out_dir / "id_map.json", "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)

    log.info(f"Index built with {index.ntotal} vectors, dim={embeddings.shape[1]}")
    log.info(f"id_map.json has {len(id_map)} entries -> {args.out_dir}")


if __name__ == "__main__":
    main()
