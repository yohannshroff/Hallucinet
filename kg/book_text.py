"""Rebuild continuous document text from data/processed/chunks.jsonl.

The chunker stores overlapping 180-word windows (40-word overlap); the
knowledge graph is built from exactly the same text the vector index
searches, so we stitch each document's chunks back together here.
"""

import json
from collections import defaultdict
from pathlib import Path

CHUNK_OVERLAP_WORDS = 40  # must match scripts/clean_and_chunk.py (--overlap)


def reconstruct_documents(chunks_path: Path) -> list:
    """Returns [{"doc_id", "title", "source_url", "text"}], in document order."""
    by_doc = defaultdict(list)
    for line in open(chunks_path, encoding="utf-8"):
        rec = json.loads(line)
        by_doc[rec["doc_id"]].append(rec)

    docs = []
    for doc_id, chunks in by_doc.items():
        chunks.sort(key=lambda r: r["chunk_index"])
        words = []
        for i, rec in enumerate(chunks):
            w = rec["text"].split()
            words += w if i == 0 else w[CHUNK_OVERLAP_WORDS:]
        docs.append(
            {"doc_id": doc_id, "title": chunks[0]["title"], "source_url": chunks[0]["source_url"], "text": " ".join(words)}
        )
    return docs
