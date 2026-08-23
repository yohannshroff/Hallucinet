"""Tests for retrieval/hybrid_retrieve.py's context formatting (pure
function, no live services needed) plus a live end-to-end smoke test that
skips automatically if Neo4j and/or the FAISS index aren't available."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kg"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))

from common import ENTITIES_CSV, FAISS_INDEX_FILE  # noqa: E402
from hybrid_retrieve import format_context_for_llm  # noqa: E402


def test_format_context_includes_graph_facts_and_citations():
    bundle = {
        "query": "Who ruled Jhansi?",
        "seed_entities": [("E023", "Jhansi")],
        "graph_facts": [
            {"subject": "Rani Lakshmibai", "relation": "ruled", "object": "Jhansi", "source": "https://example.org/x", "notes": ""}
        ],
        "vector_chunks": [],
    }
    context = format_context_for_llm(bundle)
    assert "Rani Lakshmibai --ruled--> Jhansi" in context
    assert "https://example.org/x" in context


def test_format_context_handles_empty_bundle():
    bundle = {"query": "asdf", "seed_entities": [], "graph_facts": [], "vector_chunks": []}
    context = format_context_for_llm(bundle)
    assert "no context retrieved" in context


def test_format_context_includes_vector_chunks():
    bundle = {
        "query": "Who ruled Jhansi?",
        "seed_entities": [],
        "graph_facts": [],
        "vector_chunks": [
            {"title": "Jhansi", "source_url": "https://example.org/jhansi", "text": "Jhansi was ruled by..."}
        ],
    }
    context = format_context_for_llm(bundle)
    assert "Jhansi was ruled by..." in context
    assert "https://example.org/jhansi" in context


@pytest.mark.skipif(not FAISS_INDEX_FILE.exists(), reason="FAISS index not built -- run scripts/build_faiss_index.py first")
def test_retrieve_end_to_end_smoke():
    pytest.importorskip("neo4j")
    pytest.importorskip("spacy")
    pytest.importorskip("sentence_transformers")

    from hybrid_retrieve import retrieve

    entities_df = pd.read_csv(ENTITIES_CSV, keep_default_na=False)
    bundle = retrieve("Who led the resistance at Jhansi?", entities_df, k=3)

    assert ("E023", "Jhansi") in bundle["seed_entities"]
    assert len(bundle["vector_chunks"]) > 0


@pytest.mark.skipif(not FAISS_INDEX_FILE.exists(), reason="FAISS index not built -- run scripts/build_faiss_index.py first")
def test_retrieve_modes_isolate_sources():
    """Week 7's ablation depends on vector/graph modes actually excluding
    the other source -- verify that directly rather than trusting it."""
    pytest.importorskip("neo4j")
    pytest.importorskip("spacy")
    pytest.importorskip("sentence_transformers")

    from hybrid_retrieve import retrieve

    entities_df = pd.read_csv(ENTITIES_CSV, keep_default_na=False)
    query = "Who led the resistance at Jhansi?"

    vector_only = retrieve(query, entities_df, k=3, mode="vector")
    assert vector_only["graph_facts"] == []
    assert len(vector_only["vector_chunks"]) > 0

    graph_only = retrieve(query, entities_df, k=3, mode="graph")
    assert graph_only["vector_chunks"] == []
    assert len(graph_only["graph_facts"]) > 0
