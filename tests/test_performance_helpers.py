"""Tests for the efficiency work: batched NLI gives the same answers as
one-claim-at-a-time, the FAISS index / alias index are cached, and local
PDF citations don't leak file paths into the LLM prompt. All offline."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
for sub in ("scripts", "retrieval", "eval", "api"):
    sys.path.insert(0, str(ROOT / sub))

from common import FAISS_INDEX_FILE, is_web_url  # noqa: E402
from hybrid_retrieve import format_context_only, get_alias_index  # noqa: E402


def test_is_web_url():
    assert is_web_url("https://en.wikipedia.org/wiki/Jhansi")
    assert not is_web_url("data/raw/book.pdf#page=5")


def test_local_pdf_path_is_not_shown_in_llm_context():
    bundle = {
        "graph_facts": [],
        "vector_chunks": [
            {"title": "The Great Mutiny (pp. 5-8)", "source_url": "data/raw/The Great Mutiny.pdf#page=5", "text": "Some passage."}
        ],
    }
    context = format_context_only(bundle)
    assert "The Great Mutiny, pp. 5-8" in context
    assert ".pdf" not in context


def test_alias_index_is_built_once_per_dataframe():
    df = pd.DataFrame([{"entity_id": "E1", "name": "Jhansi", "aliases": "Jhansi Fort"}])
    first = get_alias_index(df)
    assert get_alias_index(df) is first
    assert first["jhansi fort"] == "E1"


def test_batched_nli_matches_single_claim_calls():
    pytest.importorskip("sentence_transformers")
    from entailment import classify_claim, classify_claims

    evidence = [
        "Rani Lakshmibai was the ruler of the princely state of Jhansi.",
        "Delhi was captured by British forces in September 1857.",
        "Rani Lakshmibai was the ruler of the princely state of Jhansi.",  # duplicate on purpose
    ]
    claims = ["Rani Lakshmibai ruled Jhansi.", "Delhi was recaptured by the British."]
    batched = classify_claims(claims, evidence)
    assert len(batched) == 2
    for claim, result in zip(claims, batched):
        single = classify_claim(claim, evidence)
        assert result["verdict"] == single["verdict"]
        assert result["score"] == pytest.approx(single["score"], abs=1e-4)
    assert classify_claims([], evidence) == []
    assert classify_claims(claims, [])[0]["verdict"] == "no_evidence"


@pytest.mark.skipif(not FAISS_INDEX_FILE.exists(), reason="FAISS index not built")
def test_faiss_index_is_loaded_once():
    pytest.importorskip("faiss")
    from vector_search import get_index

    assert get_index() is get_index()
