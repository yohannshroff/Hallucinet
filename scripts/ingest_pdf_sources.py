"""Extract text from PDFs in data/raw/ and prepare them for the rest of the
pipeline (replaces the old Wikipedia-API fetch step).

Each PDF is split into fixed-size page groups (default 4 pages/group) --
long enough to give clean_and_chunk.py's word-window chunker real
paragraphs to work with, short enough to keep citations reasonably precise.
Each group becomes one .txt file in data/raw/text/ plus one row in
data/raw/sources_manifest.csv (doc_id, title, resolved_url, fetched_at,
word_count) -- the same manifest schema the old Wikipedia fetcher produced,
so clean_and_chunk.py and everything downstream needs no changes.

Usage:
    python scripts/ingest_pdf_sources.py
    python scripts/ingest_pdf_sources.py --pdf_dir data/raw --pages_per_doc 4
"""

import argparse
import csv
import datetime
import re
from pathlib import Path

from pypdf import PdfReader

from common import RAW_DIR, RAW_TEXT_DIR, REPO_ROOT, SOURCES_MANIFEST_CSV, get_logger

log = get_logger("ingest_pdf_sources")

LEADING_PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*\n")
HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n\s*(\w)")
WHITESPACE_RE = re.compile(r"[ \t]+")
HAS_DIGIT_RE = re.compile(r"\d")

# Back-of-book indexes ("Name, 123; Other, 456...") are dense with page-number
# citations -- a much higher fraction of digit-bearing tokens than narrative
# prose. Pages above this fraction are dropped as index/TOC noise, not text.
INDEX_PAGE_DIGIT_FRACTION = 0.15


def is_index_like_page(raw: str) -> bool:
    tokens = raw.split()
    if not tokens:
        return False
    digit_frac = sum(1 for w in tokens if HAS_DIGIT_RE.search(w)) / len(tokens)
    return digit_frac > INDEX_PAGE_DIGIT_FRACTION


def clean_page_text(raw: str) -> str:
    """Undo PDF-extraction artifacts: a lone printed page number at the top
    of the page, words hyphenated across a line break, and the leftover
    line breaks and doubled spaces from justified text."""
    raw = LEADING_PAGE_NUMBER_RE.sub("", raw)
    raw = HYPHEN_LINEBREAK_RE.sub(r"\1\2", raw)
    raw = raw.replace("\n", " ")
    raw = WHITESPACE_RE.sub(" ", raw)
    return raw.strip()


def book_title_from_filename(path: Path) -> str:
    """'The Great Mutiny - India 1857 by Christopher Hibbert.pdf' ->
    'The Great Mutiny - India 1857 by Christopher Hibbert'."""
    return path.stem


def portable_path(path: Path) -> str:
    """Repo-relative path, so citations survive moving the project folder."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return slug[:40]


def ingest_pdf(path: Path, outdir: Path, pages_per_doc: int) -> list:
    """Extract one PDF into page-group .txt files. Returns manifest rows."""
    reader = PdfReader(str(path))
    n_pages = len(reader.pages)
    book_title = book_title_from_filename(path)
    book_slug = slugify(book_title)
    fetched_at = datetime.date.today().isoformat()

    rows = []
    skipped_index_pages = 0
    for start in range(0, n_pages, pages_per_doc):
        end = min(start + pages_per_doc, n_pages)  # exclusive
        page_texts = []
        for i in range(start, end):
            raw_page = reader.pages[i].extract_text() or ""
            if is_index_like_page(raw_page):
                skipped_index_pages += 1
                continue
            page_texts.append(clean_page_text(raw_page))
        text = "\n\n".join(t for t in page_texts if t)
        word_count = len(text.split())
        if word_count == 0:
            continue  # blank/plate pages with no extractable text

        first_page, last_page = start + 1, end  # 1-indexed, inclusive
        doc_id = f"{book_slug}_p{first_page:04d}-{last_page:04d}"
        title = f"{book_title} (pp. {first_page}-{last_page})"
        resolved_url = f"{portable_path(path)}#page={first_page}"

        (outdir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        rows.append(
            {
                "doc_id": doc_id,
                "title": title,
                "resolved_url": resolved_url,
                "fetched_at": fetched_at,
                "word_count": word_count,
            }
        )
    if skipped_index_pages:
        log.info(f"  skipped {skipped_index_pages} index/TOC-like pages")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf_dir", type=Path, default=RAW_DIR, help="Directory to scan for *.pdf files")
    parser.add_argument("--outdir", type=Path, default=RAW_TEXT_DIR)
    parser.add_argument("--manifest", type=Path, default=SOURCES_MANIFEST_CSV)
    parser.add_argument("--pages_per_doc", type=int, default=4, help="PDF pages grouped into one chunkable doc")
    args = parser.parse_args()

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        log.info(f"no PDFs found in {args.pdf_dir}")
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for pdf_path in pdfs:
        log.info(f"reading {pdf_path.name} ...")
        rows = ingest_pdf(pdf_path, args.outdir, args.pages_per_doc)
        all_rows.extend(rows)
        log.info(f"  {len(PdfReader(str(pdf_path)).pages)} pages -> {len(rows)} docs")

    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "title", "resolved_url", "fetched_at", "word_count"])
        writer.writeheader()
        writer.writerows(all_rows)

    total_words = sum(r["word_count"] for r in all_rows)
    log.info(f"wrote {len(all_rows)} docs ({total_words} words total) from {len(pdfs)} PDF(s) to {args.manifest}")


if __name__ == "__main__":
    main()
