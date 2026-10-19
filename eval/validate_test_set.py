"""Check the evaluation question set (data/eval/qa_test_set.csv).

Always checks the structure (columns, unique ids, allowed values, that every
answerable question names real page groups and every unanswerable one has
none). When the processed corpus is available locally it also checks the
facts: every `evidence_terms` entry of an answerable question must occur in
its `gold_pages` page groups, and every term of an unanswerable question's
notes-claim is re-checkable by hand -- so a gold answer can never point at
pages that don't contain it.

Usage:
    python eval/validate_test_set.py
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kg"))
from common import CHUNKS_JSONL, REPO_ROOT, SOURCES_MANIFEST_CSV, get_logger  # noqa: E402

log = get_logger("validate_test_set")

QA_TEST_SET_CSV = REPO_ROOT / "data" / "eval" / "qa_test_set.csv"
COLUMNS = ["id", "question", "gold_answer", "expected_entities", "gold_pages", "evidence_terms", "hop_type", "category", "answerable", "notes"]
HOP_TYPES = {"single_hop", "multi_hop", "temporal", "unanswerable"}


def page_groups_in_manifest() -> set:
    manifest = pd.read_csv(SOURCES_MANIFEST_CSV)
    return {m.group(1) for t in manifest["title"] if (m := re.search(r"\(pp\. ([\d-]+)\)", t))}


def validate(df: pd.DataFrame, check_text: bool = True) -> list:
    problems = []
    if list(df.columns) != COLUMNS:
        problems.append(f"columns are {list(df.columns)}, expected {COLUMNS}")
        return problems
    if df["id"].duplicated().any():
        problems.append("duplicate ids")
    valid_groups = page_groups_in_manifest()

    texts = {}
    if check_text and CHUNKS_JSONL.exists():
        from book_text import reconstruct_documents

        for d in reconstruct_documents(CHUNKS_JSONL):
            texts[re.search(r"\(pp\. ([\d-]+)\)", d["title"]).group(1)] = d["text"].lower()
    elif check_text:
        log.info("processed corpus not found -- skipping the evidence-text check")

    for _, r in df.iterrows():
        rid = r["id"]
        if r["hop_type"] not in HOP_TYPES:
            problems.append(f"{rid}: unknown hop_type {r['hop_type']!r}")
        if r["answerable"] not in ("yes", "no"):
            problems.append(f"{rid}: answerable must be yes/no")
        pages = [p for p in r["gold_pages"].split(";") if p]
        terms = [t for t in r["evidence_terms"].split(";") if t]
        if r["answerable"] == "no":
            if pages or terms or r["expected_entities"]:
                problems.append(f"{rid}: unanswerable questions must have no gold pages / evidence / expected entities")
            if not r["notes"]:
                problems.append(f"{rid}: unanswerable question needs a note saying why the book cannot answer it")
            continue
        if not pages or not terms or not r["expected_entities"]:
            problems.append(f"{rid}: answerable question needs gold_pages, evidence_terms and expected_entities")
        if r["hop_type"] == "multi_hop" and len(pages) < 2:
            problems.append(f"{rid}: multi_hop question should cite at least two page groups")
        for p in pages:
            if p not in valid_groups:
                problems.append(f"{rid}: {p!r} is not a page group of the corpus")
        if texts:
            union = " ".join(texts.get(p, "") for p in pages)
            for t in terms:
                if t.lower() not in union:
                    problems.append(f"{rid}: evidence term {t!r} not found in pages {pages}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QA_TEST_SET_CSV)
    args = parser.parse_args()
    df = pd.read_csv(args.questions, keep_default_na=False)
    problems = validate(df)
    counts = df["hop_type"].value_counts().to_dict()
    log.info(f"{len(df)} questions {counts}")
    for p in problems:
        log.info(f"  PROBLEM: {p}")
    log.info("test set is consistent" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
