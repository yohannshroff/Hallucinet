"""Tests for eval/ablation.py's deterministic classify() rubric -- pure
logic, no live services needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generation"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from ablation import classify, entity_match  # noqa: E402
from prompt_template import REFUSAL_PHRASE  # noqa: E402


def test_entity_match_finds_any_of_multiple_candidates():
    assert entity_match("Henry Havelock led the column.", "Henry Havelock;James Outram")
    assert entity_match("James Outram was also present.", "Henry Havelock;James Outram")
    assert not entity_match("Colin Campbell led it.", "Henry Havelock;James Outram")


def test_entity_match_case_insensitive():
    assert entity_match("rani lakshmibai ruled jhansi", "Rani Lakshmibai")


def test_classify_correct():
    assert classify("Rani Lakshmibai ruled Jhansi.", "Rani Lakshmibai", 0.8) == "correct"


def test_classify_partial_right_entity_low_trust():
    assert classify("Rani Lakshmibai ruled Jhansi.", "Rani Lakshmibai", 0.2) == "partial"


def test_classify_wrong_entity_but_grounded():
    assert classify("Hugh Rose ruled Jhansi.", "Rani Lakshmibai", 0.8) == "wrong"


def test_classify_hallucinated_wrong_entity_and_ungrounded():
    assert classify("Napoleon ruled Jhansi.", "Rani Lakshmibai", 0.1) == "hallucinated"


def test_classify_refused():
    assert classify(REFUSAL_PHRASE, "Rani Lakshmibai", None) == "refused"


def test_classify_handles_none_trust_score_as_ungrounded():
    """trust_score is None when the answer had no checkable claims at all
    (distinct from the refusal-phrase case, e.g. an empty/degenerate
    answer) -- should not be treated as 'grounded'."""
    assert classify("Some answer with no claims.", "Rani Lakshmibai", None) in ("hallucinated", "wrong")
