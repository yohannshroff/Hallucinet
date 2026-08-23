"""Part 7 ablation: run every question in data/eval/qa_test_set.csv through
vector-only, graph-only, and hybrid retrieval, generate an answer for each,
and classify each answer as correct / partial / wrong / hallucinated.

Scoring methodology (documented here since it's a deliberate departure from
the master plan's literal "score by hand" suggestion -- see
docs/part7_notes.md for the full rationale):

The master plan's own risk register flags LLM-grading-its-own-answer as
circular. Scoring 90 answers (30 questions x 3 modes) by a human for real
would be the gold standard, but isn't something this pipeline can do
unattended. Instead we use two independent, non-LLM signals already built
in this project:

1. Entity match: does the answer contain (a case-insensitive substring of)
   any of the question's `expected_entities`? This is the same technique
   retrieval/entity_extraction.py already uses to find entities in text,
   applied here in reverse -- it's a factual-correctness proxy, not a
   judgment call by any LLM.
2. Trust score (eval/trust_score.py): the fraction of the answer's claims
   entailed by the evidence it was actually given, via the independent NLI
   model from Part 6.

Classification (deterministic, no LLM involved):
    correct       -- expected entity found AND trust_score >= 0.5
    partial        -- expected entity found AND trust_score < 0.5
                       (right answer, but under-cited/weakly grounded)
    wrong          -- expected entity NOT found AND trust_score >= 0.5
                       (grounded in *something*, just not the right fact)
    hallucinated   -- expected entity NOT found AND trust_score < 0.5
                       (unsupported claims, and wrong)
    refused        -- answer is the prompt template's refusal phrase
                       (counted as "wrong" for accuracy purposes -- the
                       question IS answerable from our sources -- but kept
                       as a distinct verdict since refusing is a very
                       different failure mode from confidently inventing
                       an answer)

This is an automated proxy, not a replacement for human judgment. The
master plan's risk register also recommends a 2-rater manual agreement
check on a sample for credibility -- do that against this script's output
before reporting these numbers in the final write-up.

Usage:
    python eval/ablation.py
    python eval/ablation.py --questions data/eval/qa_test_set.csv --out docs/part7_ablation_results.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generation"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
from common import ENTITIES_CSV, REPO_ROOT, get_logger  # noqa: E402

from generate_answer import answer_question  # noqa: E402
from hybrid_retrieve import RETRIEVAL_MODES  # noqa: E402
from ollama_client import is_reachable  # noqa: E402
from prompt_template import REFUSAL_PHRASE  # noqa: E402
from trust_score import compute_trust_score  # noqa: E402

log = get_logger("ablation")

QA_TEST_SET_CSV = REPO_ROOT / "data" / "eval" / "qa_test_set.csv"
DEFAULT_OUT = REPO_ROOT / "docs" / "part7_ablation_results.csv"
TRUST_SCORE_THRESHOLD = 0.5


def entity_match(answer: str, expected_entities: str) -> bool:
    """True if any of the ';'-separated expected_entities appears in the
    answer (case-insensitive substring match)."""
    answer_lower = answer.lower()
    return any(e.strip().lower() in answer_lower for e in expected_entities.split(";") if e.strip())


def classify(answer: str, expected_entities: str, trust_score) -> str:
    if answer.strip().rstrip(".") == REFUSAL_PHRASE.rstrip("."):
        return "refused"

    found = entity_match(answer, expected_entities)
    grounded = (trust_score is not None) and (trust_score >= TRUST_SCORE_THRESHOLD)

    if found and grounded:
        return "correct"
    if found and not grounded:
        return "partial"
    if not found and grounded:
        return "wrong"
    return "hallucinated"


def run_one(question_row: dict, entities_df: pd.DataFrame, mode: str) -> dict:
    result = answer_question(question_row["question"], entities_df, mode=mode)
    scoring = compute_trust_score(result["answer"], result["bundle"])
    verdict = classify(result["answer"], question_row["expected_entities"], scoring["trust_score"])

    return {
        "id": question_row["id"],
        "question": question_row["question"],
        "category": question_row["category"],
        "mode": mode,
        "answer": result["answer"],
        "trust_score": scoring["trust_score"],
        "n_claims": scoring["n_claims"],
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QA_TEST_SET_CSV)
    parser.add_argument("--entities", type=Path, default=ENTITIES_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--modes", nargs="+", choices=RETRIEVAL_MODES, default=list(RETRIEVAL_MODES))
    parser.add_argument("--ids", nargs="+", default=None, help="Only re-run these question ids (e.g. after fixing a test-set row)")
    args = parser.parse_args()

    if not is_reachable():
        log.info("Ollama not reachable -- see docs/manual_setup_ollama.md")
        sys.exit(1)

    questions_df = pd.read_csv(args.questions, keep_default_na=False)
    if args.ids:
        questions_df = questions_df[questions_df["id"].isin(args.ids)]
    entities_df = pd.read_csv(args.entities, keep_default_na=False)

    rows = []
    total = len(questions_df) * len(args.modes)
    done = 0
    for _, q_row in questions_df.iterrows():
        for mode in args.modes:
            done += 1
            log.info(f"[{done}/{total}] {q_row['id']} ({mode}): {q_row['question']}")
            rows.append(run_one(q_row.to_dict(), entities_df, mode))

    results_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.out, index=False)
    log.info(f"wrote {len(results_df)} results to {args.out}")

    summary = results_df.groupby(["mode", "verdict"]).size().unstack(fill_value=0)
    log.info(f"summary:\n{summary}")


if __name__ == "__main__":
    main()
