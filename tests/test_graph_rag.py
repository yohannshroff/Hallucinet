"""Offline tests for the graph-guided retrieval in retrieval/hybrid_retrieve.py
and retrieval/graph_search.py: reciprocal rank fusion, question-aware fact
ranking, the entity -> chunk index, and graph-guided passage selection.
Embeddings and the FAISS index are faked, so no model or service is needed."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
for sub in ("scripts", "kg", "eval", "retrieval"):
    sys.path.insert(0, str(ROOT / sub))

import hybrid_retrieve as H  # noqa: E402
from graph_search import rank_facts  # noqa: E402

ENTITIES = pd.DataFrame(
    [
        {"entity_id": "E1", "name": "Hodson", "entity_type": "Person", "aliases": "William Hodson"},
        {"entity_id": "E2", "name": "Bahadur Shah", "entity_type": "Person", "aliases": "King of Delhi"},
        {"entity_id": "E3", "name": "Delhi", "entity_type": "Location", "aliases": ""},
        {"entity_id": "E4", "name": "Rangoon", "entity_type": "Location", "aliases": ""},
    ]
)
ID_MAP = [
    {"chunk_id": "c0", "doc_id": "d", "title": "The Great Mutiny (pp. 1-4)", "source_url": "b#page=1", "text": "Hodson rode out to Humayun's tomb and took the King of Delhi."},
    {"chunk_id": "c1", "doc_id": "d", "title": "The Great Mutiny (pp. 5-8)", "source_url": "b#page=5", "text": "Delhi was quiet that week."},
    {"chunk_id": "c2", "doc_id": "d", "title": "The Great Mutiny (pp. 9-12)", "source_url": "b#page=9", "text": "Bahadur Shah was sent to Rangoon, where he died."},
    {"chunk_id": "c3", "doc_id": "d", "title": "The Great Mutiny (pp. 13-16)", "source_url": "b#page=13", "text": "Hodson wrote home about the heat."},
]


def fake_encode_factory(vectors: dict):
    """encode(texts) -> unit vectors looked up by text; unknown texts get [0, 0, 1]."""

    def encode(texts):
        out = np.array([vectors.get(t, [0.0, 0.0, 1.0]) for t in texts], dtype="float32")
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    return encode


class FakeIndex:
    def __init__(self, vecs):
        self.vecs = np.asarray(vecs, dtype="float32")

    def reconstruct(self, i):
        return self.vecs[i]


def test_rrf_merge_rewards_chunks_found_by_both_sources():
    vector = [{"chunk_id": "a", "title": "", "text": "", "via": "vector"}, {"chunk_id": "b", "title": "", "text": "", "via": "vector"}]
    graph = [{"chunk_id": "b", "title": "", "text": "", "via": "graph"}, {"chunk_id": "c", "title": "", "text": "", "via": "graph"}]
    merged = H.rrf_merge([vector, graph], k=3)
    assert [c["chunk_id"] for c in merged] == ["b", "a", "c"]
    assert merged[0]["via"] == "graph+vector"
    assert len(H.rrf_merge([vector, graph], k=2)) == 2


def test_rank_facts_puts_paths_first_then_the_most_relevant_and_caps():
    q = "Who captured the King?"
    facts = [
        {"subject": "Hodson", "relation": "associated_with", "object": "Delhi"},
        {"subject": "Hodson", "relation": "captured", "object": "Bahadur Shah"},
        {"subject": "Bahadur Shah", "relation": "ruled", "object": "Delhi", "via": "path"},
        {"subject": "Hodson", "relation": "associated_with", "object": "Rangoon"},
    ]
    encode = fake_encode_factory({
        q: [1, 0, 0],
        "Hodson captured Bahadur Shah.": [0.9, 0.1, 0],
        "Hodson and Delhi are mentioned together in the book.": [0.5, 0.5, 0],
        "Bahadur Shah ruled Delhi.": [0, 1, 0],
    })
    ranked = rank_facts(q, facts, cap=3, encode=encode)
    assert [(f["subject"], f["relation"], f["object"]) for f in ranked] == [
        ("Bahadur Shah", "ruled", "Delhi"),  # path facts always first
        ("Hodson", "captured", "Bahadur Shah"),
        ("Hodson", "associated_with", "Delhi"),
    ]
    assert rank_facts(q, [], cap=3, encode=encode) == []


def test_entity_chunk_rows_matches_names_and_aliases_case_insensitively():
    df = ENTITIES.copy()
    rows = H.entity_chunk_rows(df, ID_MAP)
    assert rows["E1"] == {0, 3}  # Hodson
    assert rows["E2"] == {0, 2}  # "King of Delhi" alias + "Bahadur Shah"
    assert rows["E3"] == {1}  # in c0 "Delhi" is part of "King of Delhi" (longest match wins), i.e. Bahadur Shah
    assert rows["E4"] == {2}


def test_graph_passages_prefers_chunks_naming_a_seed_with_a_neighbour(monkeypatch):
    df = ENTITIES.copy()
    q = "What did Hodson do?"
    monkeypatch.setattr(H, "get_index", lambda index_dir: (FakeIndex([[1, 0, 0], [0, 1, 0], [0.8, 0.2, 0], [0.99, 0.01, 0]]), ID_MAP))
    monkeypatch.setattr(H, "encode", fake_encode_factory({q: [1, 0, 0]}))
    facts = [{"subject": "Hodson", "relation": "captured", "object": "Bahadur Shah"}]

    passages = H.graph_passages(q, ["E1"], facts, df, Path("."), k=5)
    # c0 names Hodson and his neighbour Bahadur Shah; c3 names Hodson alone, so it is left out
    assert [p["chunk_id"] for p in passages] == ["c0"]
    assert passages[0]["via"] == "graph"


def test_graph_passages_falls_back_to_any_seed_mention_without_graph_facts(monkeypatch):
    df = ENTITIES.copy()
    q = "Tell me about Hodson"
    monkeypatch.setattr(H, "get_index", lambda index_dir: (FakeIndex([[0.5, 0.5, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]]), ID_MAP))
    monkeypatch.setattr(H, "encode", fake_encode_factory({q: [1, 0, 0]}))

    passages = H.graph_passages(q, ["E1"], [], df, Path("."), k=5)
    assert [p["chunk_id"] for p in passages] == ["c3", "c0"]  # both name Hodson; ranked by similarity
    assert H.graph_passages(q, [], [], df, Path("."), k=5) == []


def test_widen_passages_adds_neighbouring_chunks_without_the_overlap(monkeypatch):
    monkeypatch.setattr(H, "CHUNK_OVERLAP_WORDS", 2)
    id_map = [
        {"chunk_id": "a0", "doc_id": "a", "text": "w1 w2 w3 w4"},
        {"chunk_id": "a1", "doc_id": "a", "text": "w3 w4 w5 w6"},
        {"chunk_id": "a2", "doc_id": "a", "text": "w5 w6 w7 w8"},
        {"chunk_id": "b0", "doc_id": "b", "text": "x1 x2 x3 x4"},
    ]
    wide = H.widen_passages([{"chunk_id": "a1", "text": "w3 w4 w5 w6", "title": "t"}], id_map, window=1)
    assert wide[0]["text"] == "w1 w2 w3 w4 w5 w6 w7 w8"
    edge = H.widen_passages([{"chunk_id": "a2", "text": "w5 w6 w7 w8"}], id_map, window=1)
    assert edge[0]["text"] == "w3 w4 w5 w6 w7 w8"  # stops at the page-group boundary
    unknown = {"chunk_id": None, "text": "web snippet"}
    assert H.widen_passages([unknown], id_map) == [unknown]


def test_distinct_top_skips_neighbours_of_chosen_passages():
    id_map = [{"chunk_id": f"c{i}", "doc_id": "d", "text": ""} for i in range(10)]
    ranked = [{"chunk_id": c} for c in ("c4", "c5", "c6", "c7", "c0", "c9")]
    # c5, c6 overlap c4 once widened by one chunk each side; c7 does not
    assert [p["chunk_id"] for p in H.distinct_top(ranked, id_map, 3, window=1)] == ["c4", "c7", "c0"]


@pytest.mark.parametrize("title, expected", [
    ("The Great Mutiny - India 1857 by Christopher Hibbert (pp. 401-404)", True),
    ("The Great Mutiny - India 1857 by Christopher Hibbert (pp. 417-420)", True),
    ("The Great Mutiny - India 1857 by Christopher Hibbert (pp. 425-428)", False),  # chronological table
    ("The Great Mutiny - India 1857 by Christopher Hibbert (pp. 9-12)", False),  # contents
    ("Some Other Book (pp. 489-492)", True),  # no entry: searched in full
    ("Jhansi", True),  # not a book page group (e.g. a web source)
])
def test_is_narrative(title, expected):
    from common import is_narrative

    assert is_narrative(title) is expected


def test_retrieve_reranks_the_pool_and_skips_back_matter(monkeypatch):
    book = "The Great Mutiny - India 1857 by Christopher Hibbert"
    pool = [
        {"chunk_id": "c1", "title": f"{book} (pp. 425-428)", "source_url": "", "text": "1857 June 25 ...", "score": 0.9},
        {"chunk_id": "c2", "title": f"{book} (pp. 197-200)", "source_url": "", "text": "the November battle", "score": 0.8},
        {"chunk_id": "c3", "title": f"{book} (pp. 213-216)", "source_url": "", "text": "the June siege", "score": 0.7},
    ]
    monkeypatch.setattr(H, "extract_entities", lambda query, alias_index: [])
    monkeypatch.setattr(H, "vector_search", lambda query, index_dir, k: pool[:k])
    monkeypatch.setattr(H, "RERANK", True)
    monkeypatch.setattr(H, "rerank", lambda query, passages, k: sorted(passages, key=lambda p: "June" not in p["text"])[:k])

    bundle = H.retrieve("What happened during the siege?", ENTITIES.copy(), k=2, mode="vector", index_dir=Path("no-index"))
    assert [p["chunk_id"] for p in bundle["passages"]] == ["c3", "c2"]  # reranked; the chronology never appears
    assert [c["chunk_id"] for c in bundle["vector_chunks"]] == ["c2", "c3"]  # pool order kept for per-source metrics


def test_context_shows_the_merged_passages():
    bundle = {
        "query": "q", "seed_entities": [], "graph_facts": [], "vector_chunks": [],
        "passages": [{"title": "The Great Mutiny (pp. 9-12)", "source_url": "b#page=9", "text": "Bahadur Shah was sent to Rangoon."}],
    }
    assert "Bahadur Shah was sent to Rangoon." in H.format_context_only(bundle)


@pytest.mark.parametrize("bundle, expected", [
    ({"passages": [{"title": "x (pp. 9-12)"}], "vector_chunks": [{"title": "x (pp. 1-4)"}]}, [9]),
    ({"vector_chunks": [{"title": "x (pp. 1-4)"}]}, [1]),  # bundles cached before `passages` existed
])
def test_retrieval_metrics_score_what_the_llm_saw(bundle, expected):
    import rag_metrics as M

    assert M.ranked_groups(bundle) == expected
