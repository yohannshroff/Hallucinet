"""Pre-run the 10 sample questions through the full pipeline and cache the
results as JSON -- a fallback for ui/app.py if live Ollama inference is too
slow or fails during a demo (per the master plan's risk register: "Local
LLM too slow for live demo -> cached/pre-run fallback").

Usage:
    python api/build_demo_cache.py
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import ENTITIES_CSV, REPO_ROOT, get_logger  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from response import build_ask_response  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generation"))
from ollama_client import is_reachable  # noqa: E402

log = get_logger("build_demo_cache")

SAMPLE_QUESTIONS_CSV = REPO_ROOT / "data" / "eval" / "sample_questions.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "eval" / "demo_cache.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=SAMPLE_QUESTIONS_CSV)
    parser.add_argument("--entities", type=Path, default=ENTITIES_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not is_reachable():
        log.info("Ollama not reachable -- see docs/manual_setup_ollama.md")
        sys.exit(1)

    questions_df = pd.read_csv(args.questions, keep_default_na=False)
    entities_df = pd.read_csv(args.entities, keep_default_na=False)

    cache = []
    for _, row in questions_df.iterrows():
        log.info(f"{row['id']}: {row['question']}")
        cache.append(build_ask_response(row["question"], entities_df))

    args.out.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"wrote {len(cache)} cached answers to {args.out}")


if __name__ == "__main__":
    main()
