"""Integration test for retrieval/graph_search.py against a live Neo4j
instance (loaded by kg/load_graph.py). Skipped automatically if Neo4j isn't
reachable."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kg"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))

pytest.importorskip("neo4j")

from graph_search import get_driver, one_hop_facts, search  # noqa: E402

JHANSI_ENTITY_ID = "E023"


@pytest.fixture(scope="module")
def driver():
    d = get_driver()
    try:
        d.verify_connectivity()
    except Exception:
        pytest.skip("Neo4j is not reachable -- run kg/load_graph.py against a running instance first")
    yield d
    d.close()


def test_one_hop_facts_for_jhansi_includes_rani_lakshmibai(driver):
    with driver.session() as session:
        facts = one_hop_facts(session, JHANSI_ENTITY_ID)

    subjects_and_relations = {(f["subject"], f["relation"]) for f in facts}
    assert ("Rani Lakshmibai", "ruled") in subjects_and_relations


def test_search_deduplicates_across_seed_entities(driver):
    with driver.session() as session:
        # Jhansi and Rani Lakshmibai share the "ruled" edge -- searching
        # both as seeds should not produce that fact twice.
        facts = search(session, [JHANSI_ENTITY_ID, "E002"])

    keys = [(f["subject"], f["relation"], f["object"]) for f in facts]
    assert len(keys) == len(set(keys))
