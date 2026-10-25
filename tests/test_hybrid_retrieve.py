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
    assert "[1] Knowledge graph, https://example.org/x: Rani Lakshmibai ruled Jhansi." in context


def test_format_context_numbers_facts_then_passages_without_file_paths():
    bundle = {
        "query": "q",
        "graph_facts": [
            {"subject": "Hodson", "relation": "captured", "object": "Bahadur Shah", "source": "data/raw/book.pdf#page=393", "notes": ""},
            {"subject": "Jhansi", "relation": "associated_with", "object": "Hugh Rose", "source": "data/raw/book.pdf#page=405", "notes": ""},
        ],
        "vector_chunks": [],
        "passages": [{"title": "The Great Mutiny - India 1857 by Christopher Hibbert (pp. 401-404)", "source_url": "data/raw/book.pdf#page=401", "text": "The Rani..."}],
    }
    assert format_context_for_llm(bundle).splitlines()[2:] == [
        "[1] Knowledge graph, p. 393: Hodson captured Bahadur Shah.",
        "[2] Knowledge graph, p. 405: Jhansi and Hugh Rose are mentioned together in the book.",
        "[3] The Great Mutiny, pp. 401-404: The Rani...",
    ]


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

    assert "Jhansi" in [name for _, name in bundle["seed_entities"]]
    assert len(bundle["vector_chunks"]) > 0


@pytest.mark.skipif(not FAISS_INDEX_FILE.exists(), reason="FAISS index not built -- run scripts/build_faiss_index.py first")
def test_retrieve_modes_isolate_sources():
    """Part 7's ablation depends on vector/graph modes actually excluding
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

    assert all(p["via"] == "vector" for p in vector_only["passages"])

    graph_only = retrieve(query, entities_df, k=3, mode="graph")
    assert graph_only["vector_chunks"] == []
    # Jhansi's own facts are co-mentions and "Siege of Jhansi is located in Jhansi", which the
    # LLM no longer sees -- the graph's contribution here is the graph-guided passages
    assert len(graph_only["passages"]) > 0
    assert all(p["via"] == "graph" for p in graph_only["passages"])
