"""Fetch plain-text extracts of the Wikipedia articles listed in
data/raw/source_list.csv, using the Wikipedia API (no HTML parsing needed).

Writes one cleaned .txt file per article to data/raw/text/, and records
provenance (resolved URL, fetch date, word count) in
data/raw/sources_manifest.csv so every KG fact can be traced back to a real,
dated source.

Usage:
    python scripts/fetch_wikipedia_sources.py
    python scripts/fetch_wikipedia_sources.py --list data/raw/source_list.csv --outdir data/raw/text
"""

import argparse
import csv
import datetime
import time
from pathlib import Path

import pandas as pd
import requests

from common import RAW_TEXT_DIR, SOURCE_LIST_CSV, SOURCES_MANIFEST_CSV, get_logger

log = get_logger("fetch_wikipedia_sources")

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "HalluciNet-student-project/1.0 (educational use; contact via GitHub)"
REQUEST_DELAY_SECONDS = 1.5  # be polite to the API and avoid 429s
MAX_RETRIES = 4
SHORT_ARTICLE_WORD_THRESHOLD = 50  # likely a disambiguation page, not a real article


def fetch_article_extract(title: str) -> dict:
    """Fetch a plain-text extract for a Wikipedia article title.

    Returns {"text": str, "resolved_url": str, "resolved_title": str}.
    Raises RuntimeError if the page can't be found.
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
        "format": "json",
    }

    resp = None
    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.get(WIKIPEDIA_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
        if resp.status_code == 429 and attempt < MAX_RETRIES:
            wait = 2 ** attempt  # 2s, 4s
            log.info(f"  rate-limited, retrying in {wait}s ({attempt}/{MAX_RETRIES})")
            time.sleep(wait)
            continue
        break
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        raise RuntimeError(f"no pages returned for title '{title}'")

    page = next(iter(pages.values()))
    if "missing" in page:
        raise RuntimeError(f"Wikipedia page not found for title '{title}'")

    text = page.get("extract", "")
    resolved_title = page.get("title", title)
    resolved_url = "https://en.wikipedia.org/wiki/" + resolved_title.replace(" ", "_")

    return {"text": text, "resolved_url": resolved_url, "resolved_title": resolved_title}


def save_article(doc_id: str, data: dict, outdir: Path) -> int:
    """Write the article text to <outdir>/<doc_id>.txt. Returns word count."""
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{doc_id}.txt"
    out_path.write_text(data["text"], encoding="utf-8")
    return len(data["text"].split())


def load_existing_manifest(path: Path) -> dict:
    """Return {doc_id: row_dict} for articles already fetched successfully."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {row["doc_id"]: row.to_dict() for _, row in df.iterrows()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", type=Path, default=SOURCE_LIST_CSV)
    parser.add_argument("--outdir", type=Path, default=RAW_TEXT_DIR)
    parser.add_argument("--manifest", type=Path, default=SOURCES_MANIFEST_CSV)
    parser.add_argument(
        "--force", action="store_true", help="Re-fetch articles even if already present in the manifest"
    )
    args = parser.parse_args()

    sources = pd.read_csv(args.list)
    existing = {} if args.force else load_existing_manifest(args.manifest)

    manifest_rows = list(existing.values())
    failures = []
    skipped = 0

    for _, row in sources.iterrows():
        doc_id = row["doc_id"]
        title = row["title"]

        if doc_id in existing:
            skipped += 1
            continue

        log.info(f"fetching '{title}' -> {doc_id}.txt")
        try:
            data = fetch_article_extract(title)
            word_count = save_article(doc_id, data, args.outdir)
            if word_count < SHORT_ARTICLE_WORD_THRESHOLD:
                log.info(f"  WARNING: only {word_count} words -- likely a disambiguation page, check the title")
            manifest_rows.append(
                {
                    "doc_id": doc_id,
                    "title": data["resolved_title"],
                    "resolved_url": data["resolved_url"],
                    "fetched_at": datetime.date.today().isoformat(),
                    "word_count": word_count,
                }
            )
            log.info(f"  ok: {word_count} words")
        except Exception as exc:  # noqa: BLE001 - keep going through the rest of the list
            log.info(f"  FAILED: {exc}")
            failures.append((doc_id, title, str(exc)))
        finally:
            time.sleep(REQUEST_DELAY_SECONDS)

    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "title", "resolved_url", "fetched_at", "word_count"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    log.info(f"{len(manifest_rows)} total articles ({skipped} already had, {len(manifest_rows) - skipped} newly fetched), {len(failures)} failures")
    if failures:
        log.info("failed titles -- re-run this script to retry them, or check spelling / try an alternate Wikipedia title:")
        for doc_id, title, err in failures:
            log.info(f"  - {doc_id} ('{title}'): {err}")


if __name__ == "__main__":
    main()
