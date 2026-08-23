"""Tests for eval/trust_score.py: RELATION_TEMPLATES coverage (drift guard,
same pattern as test_kg_vocab_consistency.py) and compute_trust_score's
behavior on synthetic bundles."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generation"))

from common import RELATION_VOCAB  # noqa: E402

pytest.importorskip("sentence_transformers")
from trust_score import RELATION_TEMPLATES, compute_trust_score, fact_to_sentence  # noqa: E402


def test_relation_templates_cover_full_vocab():
    missing = RELATION_VOCAB - set(RELATION_TEMPLATES.keys())
    assert not missing, f"RELATION_TEMPLATES is missing template(s) for: {missing}"


def test_fact_to_sentence_produces_readable_sentence():
    fact = {"subject": "Rani Lakshmibai", "relation": "ruled", "object": "Jhansi"}
    assert fact_to_sentence(fact) == "Rani Lakshmibai ruled Jhansi."


def test_compute_trust_score_none_for_no_claims():
    bundle = {"graph_facts": [], "vector_chunks": []}
    scoring = compute_trust_score("", bundle)
    assert scoring["trust_score"] is None
    assert scoring["n_claims"] == 0


def test_compute_trust_score_high_for_grounded_answer():
    bundle = {
        "graph_facts": [{"subject": "Rani Lakshmibai", "relation": "ruled", "object": "Jhansi", "source": "x", "notes": ""}],
        "vector_chunks": [],
    }
    scoring = compute_trust_score("Rani Lakshmibai ruled Jhansi.", bundle)
    assert scoring["trust_score"] == 1.0


def test_compute_trust_score_low_for_ungrounded_claim():
    bundle = {
        "graph_facts": [{"subject": "Rani Lakshmibai", "relation": "ruled", "object": "Jhansi", "source": "x", "notes": ""}],
        "vector_chunks": [],
    }
    # An unrelated, unsupported claim with no matching evidence.
    scoring = compute_trust_score("Napoleon invaded Russia in 1812.", bundle)
    assert scoring["trust_score"] == 0.0


def test_compute_trust_score_ignores_the_refusal_phrase():
    """The model's own 'I don't know' refusal is meta-commentary, not a
    factual claim -- it must not be scored as an unsupported claim (that
    would wrongly report 0% trust for exactly the desired behavior)."""
    from prompt_template import REFUSAL_PHRASE

    bundle = {"graph_facts": [], "vector_chunks": []}
    scoring = compute_trust_score(REFUSAL_PHRASE, bundle)
    assert scoring["trust_score"] is None
    assert scoring["n_claims"] == 0


def test_evidence_texts_from_bundle_splits_chunks_into_sentences():
    """Vector chunk text must be decomposed into individual sentences, not
    passed whole -- the NLI model is trained on single-sentence pairs and
    scores true claims far too low against a whole multi-sentence chunk."""
    from trust_score import evidence_texts_from_bundle

    bundle = {
        "graph_facts": [],
        "vector_chunks": [{"text": "The act ended Company rule. The British Crown took over administration."}],
    }
    texts = evidence_texts_from_bundle(bundle)
    assert len(texts) == 2
    assert "ended Company rule" in texts[0]
