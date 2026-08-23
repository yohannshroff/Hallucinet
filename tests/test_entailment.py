"""Tests for eval/entailment.py's NLI classification, using clear-cut
sentence pairs so verdicts are unambiguous. Downloads the (small)
cross-encoder/nli-MiniLM2-L6-H768 model on first run, same pattern as the
Week 2 embedding tests."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

pytest.importorskip("sentence_transformers")
from entailment import classify_claim  # noqa: E402


EVIDENCE = ["Rani Lakshmibai was the ruler of the princely state of Jhansi."]


def test_classify_claim_entailed():
    result = classify_claim("Rani Lakshmibai ruled Jhansi.", EVIDENCE)
    assert result["verdict"] == "entailed"
    assert result["score"] > 0.5


def test_classify_claim_not_entailed_for_contradiction():
    result = classify_claim("Rani Lakshmibai ruled Delhi.", EVIDENCE)
    assert result["verdict"] != "entailed"


def test_classify_claim_picks_best_evidence_from_multiple():
    evidence = [
        "The sky is blue on a clear day.",
        "Rani Lakshmibai was the ruler of the princely state of Jhansi.",
        "Cooking rice takes about twenty minutes.",
    ]
    result = classify_claim("Rani Lakshmibai ruled Jhansi.", evidence)
    assert result["verdict"] == "entailed"
    assert "Jhansi" in result["best_evidence"]


def test_classify_claim_no_evidence():
    result = classify_claim("Rani Lakshmibai ruled Jhansi.", [])
    assert result["verdict"] == "no_evidence"
