"""Tests for eval/claim_splitting.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

pytest.importorskip("spacy")
from claim_splitting import split_into_claims, strip_citations  # noqa: E402


def test_strip_citations_removes_bracketed_refs():
    assert strip_citations("Rani Lakshmibai ruled Jhansi [1, 2].") == "Rani Lakshmibai ruled Jhansi."


@pytest.mark.parametrize("raw, clean", [
    # the citation styles the model actually produced (from UI screenshots)
    ("Colin Campbell was an ally of James Outram (data/raw/The Great Mutiny - India 1857 by Christopher Hibbert.pdf#page=393).",
     "Colin Campbell was an ally of James Outram."),
    ("Hugh Rose led the resistance. [data/raw/The Great Mutiny - India 1857 by Christopher Hibbert.pdf#page=409], [data/raw/x.pdf#page=5]",
     "Hugh Rose led the resistance."),
    ("*   **Havelock's force leaves Allahabad** (Source [The Great Mutiny - India 1857 by Christopher Hibbert (pp. 197-200)], [12]).",
     "Havelock's force leaves Allahabad."),
    ("Wickins described the socks (Source , pp. 213-216).", "Wickins described the socks."),
    ("She was shot in the back. Source: , , ,", "She was shot in the back."),
])
def test_strip_citations_removes_paths_sources_pages_and_markdown(raw, clean):
    assert strip_citations(raw) == clean


def test_split_into_claims_drops_lead_ins_and_treats_bullets_as_sentences():
    answer = "Here's what it reveals:\n* Havelock left Allahabad for Cawnpore on 7 July\n* Neill stayed behind at Allahabad (Source [2])."
    assert split_into_claims(answer) == ["Havelock left Allahabad for Cawnpore on 7 July.", "Neill stayed behind at Allahabad."]


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
