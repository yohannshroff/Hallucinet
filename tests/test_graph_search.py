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



def entity_id(name):
    """IDs change whenever the graph is rebuilt, so tests look entities up by name."""
    import pandas as pd
    from common import ENTITIES_CSV

    df = pd.read_csv(ENTITIES_CSV, keep_default_na=False)
    return df.loc[df["name"] == name, "entity_id"].iloc[0]


@pytest.fixture(scope="module")
def driver():
    d = get_driver()
    try:
        d.verify_connectivity()
    except Exception:
        pytest.skip("Neo4j is not reachable -- run kg/load_graph.py against a running instance first")
    yield d
    d.close()


def test_one_hop_facts_for_delhi_include_its_siege(driver):
    with driver.session() as session:
        facts = one_hop_facts(session, entity_id("Delhi"))

    subjects_and_relations = {(f["subject"], f["relation"]) for f in facts}
    assert ("Siege of Delhi", "located_in") in subjects_and_relations


def test_search_deduplicates_across_seed_entities(driver):
    with driver.session() as session:
        # Delhi and the Siege of Delhi share a "located_in" edge -- searching
        # both as seeds should not produce that fact twice.
        facts = search(session, [entity_id("Delhi"), entity_id("Siege of Delhi")])

    keys = [(f["subject"], f["relation"], f["object"]) for f in facts]
    assert len(keys) == len(set(keys))
