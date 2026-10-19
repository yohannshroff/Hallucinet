"""Human validation of the automatic scorer (two annotators, Cohen's kappa).

    python eval/annotation.py export --n 60      # writes one blank sheet per annotator
    python eval/annotation.py kappa              # after both sheets are filled in
    python eval/annotation.py gold               # sheet for checking the gold answers against the book

Each annotator opens their own sheet, reads the question, the gold answer and
the model's answer, and fills in ONE column, `human_correct` (1 or 0):
  1 = the answer states what the gold answer says (or, for an unanswerable
      question, it correctly says the book does not say);
  0 = it is wrong, invents something, or refuses an answerable question.
They must not look at each other's sheet or at the automatic verdict (hidden).
`kappa` then reports agreement between the two annotators and each
annotator's agreement with the automatic `correct` verdict.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from rag_stats import cohen_kappa  # noqa: E402

RUNS = ROOT / "data" / "eval" / "runs" / "generation.jsonl"
QA = ROOT / "data" / "eval" / "qa_test_set.csv"
OUT_DIR = ROOT / "data" / "eval" / "annotation"
KEY = OUT_DIR / "key_automatic_verdicts.csv"
SHEETS = {"a": OUT_DIR / "sheet_annotator_1.csv", "b": OUT_DIR / "sheet_annotator_2.csv"}


def export(n: int, seed: int):
    runs = pd.DataFrame([json.loads(l) for l in RUNS.read_text().splitlines()])
    qa = pd.read_csv(QA, keep_default_na=False).set_index("id")
    runs = runs[runs["run"] == 1]
    # Stratified: equal share per system so every system's scorer is checked.
    per_mode = max(1, n // runs["mode"].nunique())
    sample = pd.concat([d.sample(min(per_mode, len(d)), random_state=seed) for _, d in runs.groupby("mode")])
    sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle, so order does not hint at the system
    sample["item"] = range(1, len(sample) + 1)
    sheet = pd.DataFrame({
        "item": sample["item"],
        "question": sample["id"].map(qa["question"]),
        "gold_answer": sample["id"].map(qa["gold_answer"]),
        "answerable": sample["id"].map(qa["answerable"]),
        "model_answer": sample["answer"],
        "human_correct": "",
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in SHEETS.values():
        sheet.to_csv(path, index=False)
    sample[["item", "id", "mode", "run", "correct"]].to_csv(KEY, index=False)
    print(f"wrote {len(sheet)} items to {', '.join(str(p) for p in SHEETS.values())}")


def gold_sheet():
    """One row per question for checking the gold answers against the book."""
    qa = pd.read_csv(QA, keep_default_na=False)
    sheet = qa[["id", "question", "gold_answer", "gold_pages", "evidence_terms", "answerable"]].copy()
    sheet["book_says_this_1_or_0"] = ""
    sheet["notes_or_correction"] = ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "gold_answer_check.csv"
    sheet.to_csv(path, index=False)
    print(f"wrote {len(sheet)} rows to {path}")


def kappa():
    a = pd.read_csv(SHEETS["a"]).set_index("item")["human_correct"]
    b = pd.read_csv(SHEETS["b"]).set_index("item")["human_correct"]
    key = pd.read_csv(KEY).set_index("item")["correct"].astype(int)
    if a.isna().any() or b.isna().any():
        sys.exit("both sheets must have every human_correct filled in with 1 or 0")
    a, b = a.astype(int), b.astype(int)
    print(f"items: {len(a)}")
    print(f"annotator 1 vs 2   kappa = {cohen_kappa(a, b):.3f}  (raw agreement {(a == b).mean():.2f})")
    agreed = a[a == b]
    print(f"automatic vs humans (items both annotators agree on, n={len(agreed)}): "
          f"kappa = {cohen_kappa(agreed, key[agreed.index]):.3f}  (raw agreement {(agreed == key[agreed.index]).mean():.2f})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--n", type=int, default=60)
    e.add_argument("--seed", type=int, default=7)
    sub.add_parser("kappa")
    sub.add_parser("gold")
    args = p.parse_args()
    {"export": lambda: export(args.n, args.seed), "kappa": kappa, "gold": gold_sheet}[args.cmd]()


if __name__ == "__main__":
    main()
