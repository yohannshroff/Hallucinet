"""Embed each chunk in chunks.jsonl with sentence-transformers
all-MiniLM-L6-v2 and save the resulting matrix as a .npy file.

Usage:
    python scripts/build_embeddings.py
    python scripts/build_embeddings.py --chunks data/processed/chunks.jsonl \
        --out_dir data/processed/embeddings --model all-MiniLM-L6-v2
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common import CHUNKS_JSONL, EMBEDDING_MODEL_NAME, EMBEDDINGS_DIR, get_logger

log = get_logger("build_embeddings")


def load_chunks(path: Path) -> list:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def embed_chunks(texts: list, model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,  # so cosine similarity == inner product later
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_JSONL)
    parser.add_argument("--out_dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--model", default=EMBEDDING_MODEL_NAME)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    if not chunks:
        raise SystemExit(f"no chunks found in {args.chunks} -- run clean_and_chunk.py first")

    texts = [c["text"] for c in chunks]
    log.info(f"embedding {len(texts)} chunks with {args.model}")
    embeddings = embed_chunks(texts, model_name=args.model)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "chunk_embeddings.npy"
    np.save(out_path, embeddings)

    log.info(f"Saved embeddings: shape {embeddings.shape} -> {out_path}")


if __name__ == "__main__":
    main()
