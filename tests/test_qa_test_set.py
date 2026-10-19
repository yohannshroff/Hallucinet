"""The evaluation question set must stay consistent with the corpus
(structure always; evidence text when the processed corpus is present)."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for sub in ("scripts", "eval", "kg"):
    sys.path.insert(0, str(ROOT / sub))

from validate_test_set import QA_TEST_SET_CSV, validate  # noqa: E402


def test_question_set_is_consistent():
    df = pd.read_csv(QA_TEST_SET_CSV, keep_default_na=False)
    assert validate(df) == []


def test_question_set_covers_every_question_type():
    df = pd.read_csv(QA_TEST_SET_CSV, keep_default_na=False)
    counts = df["hop_type"].value_counts()
    assert len(df) >= 50
    for hop in ("single_hop", "multi_hop", "temporal", "unanswerable"):
        assert counts.get(hop, 0) >= 10, f"need at least 10 {hop} questions"


def test_validator_catches_a_bad_page_group():
    df = pd.read_csv(QA_TEST_SET_CSV, keep_default_na=False)
    df.loc[0, "gold_pages"] = "9999-9999"
    assert any("not a page group" in p for p in validate(df, check_text=False))
