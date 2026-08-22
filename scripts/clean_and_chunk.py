"""Clean the fetched Wikipedia text and split it into overlapping word-window
chunks for embedding.

Chunking: 180-word windows with 40-word overlap (stride 140), sized to stay
comfortably under MiniLM's ~256-subtoken limit after word->subtoken
expansion.

Usage:
    python scripts/clean_and_chunk.py
    python scripts/clean_and_chunk.py --input_dir data/raw/text \
        --manifest data/raw/sources_manifest.csv \
        --output data/processed/chunks.jsonl --chunk_size 180 --overlap 40
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from common import CHUNKS_JSONL, RAW_TEXT_DIR, SOURCES_MANIFEST_CSV, get_logger

log = get_logger("clean_and_chunk")

# Wikipedia plain-text extracts use "== Section ==" / "=== Subsection ==="
# style headers; strip them and cut off boilerplate tail sections.
SECTION_HEADER_RE = re.compile(r"^=+\s*.*?\s*=+$", re.MULTILINE)
CITATION_MARKER_RE = re.compile(r"\[\d+\]")
BOILERPLATE_HEADERS = ("References", "External links", "See also", "Further reading", "Notes")


def clean_text(raw: str) -> str:
    # Cut off everything from the first boilerplate section onward.
    cut_index = len(raw)
    for header in BOILERPLATE_HEADERS:
        match = re.search(rf"==\s*{re.escape(header)}\s*==", raw)
        if match:
            cut_index = min(cut_index, match.start())
    raw = raw[:cut_index]

    raw = CITATION_MARKER_RE.sub("", raw)
    raw = SECTION_HEADER_RE.sub("", raw)
    raw = re.sub(r"\n{2,}", "\n\n", raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    return raw.strip()


def chunk_words(words: list, chunk_size: int = 180, overlap: int = 40) -> list:
    """Split a word list into overlapping windows. Returns a list of word lists."""
    if not words:
        return []
    stride = chunk_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(words):
        window = words[start : start + chunk_size]
        chunks.append(window)
        if start + chunk_size >= len(words):
            break
        start += stride
    return chunks


def process_file(path: Path, manifest_row: dict, chunk_size: int, overlap: int) -> list:
    raw = path.read_text(encoding="utf-8")
    cleaned = clean_text(raw)
    words = cleaned.split()
    word_chunks = chunk_words(words, chunk_size=chunk_size, overlap=overlap)

    records = []
    for i, chunk in enumerate(word_chunks):
        text = " ".join(chunk)
        records.append(
            {
                "chunk_id": f"{manifest_row['doc_id']}_{i:03d}",
                "doc_id": manifest_row["doc_id"],
                "title": manifest_row["title"],
                "source_url": manifest_row["resolved_url"],
                "chunk_index": i,
                "text": text,
                "word_count": len(chunk),
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=RAW_TEXT_DIR)
    parser.add_argument("--manifest", type=Path, default=SOURCES_MANIFEST_CSV)
    parser.add_argument("--output", type=Path, default=CHUNKS_JSONL)
    parser.add_argument("--chunk_size", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=40)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)

    all_records = []
    for _, row in manifest.iterrows():
        path = args.input_dir / f"{row['doc_id']}.txt"
        if not path.exists():
            log.info(f"skipping {row['doc_id']}: {path} not found")
            continue
        records = process_file(path, row.to_dict(), args.chunk_size, args.overlap)
        all_records.extend(records)
        log.info(f"{row['doc_id']}: {len(records)} chunks")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info(f"wrote {len(all_records)} chunks to {args.output}")


if __name__ == "__main__":
    main()
