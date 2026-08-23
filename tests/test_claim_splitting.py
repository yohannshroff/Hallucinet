"""Tests for eval/claim_splitting.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

pytest.importorskip("spacy")
from claim_splitting import split_into_claims, strip_citations  # noqa: E402


def test_strip_citations_removes_bracketed_refs():
    assert strip_citations("Rani Lakshmibai ruled Jhansi [1, 2].") == "Rani Lakshmibai ruled Jhansi ."


def test_strip_citations_removes_url_brackets():
    text = "Rani Lakshmibai ruled Jhansi. [https://en.wikipedia.org/wiki/Rani_of_Jhansi]"
    assert "https://" not in strip_citations(text)


def test_split_into_claims_returns_one_claim_per_sentence():
    answer = "Rani Lakshmibai ruled Jhansi. She fought against Hugh Rose in 1858."
    claims = split_into_claims(answer)
    assert len(claims) == 2
    assert "Rani Lakshmibai ruled Jhansi" in claims[0]


def test_split_into_claims_drops_short_fragments():
    """A trailing 'References: [1], [2], [3]' becomes just 'References:'
    after citation stripping -- too short to be a checkable claim."""
    answer = "Rani Lakshmibai ruled Jhansi and fought the British. References: [1], [2], [3]"
    claims = split_into_claims(answer)
    assert all("References" not in c for c in claims)


def test_split_into_claims_empty_for_no_content():
    assert split_into_claims("") == []
