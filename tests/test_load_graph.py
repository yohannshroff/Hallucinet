"""Integration test for kg/load_graph.py against a live Neo4j instance.

Requires Neo4j to be running and reachable via the config in .env (see
docs/manual_setup_neo4j.md) -- skipped automatically if it isn't, so
`pytest tests/` still runs cleanly without a database up.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kg"))

pytest.importorskip("neo4j")

from common import ENTITIES_CSV, RELATIONSHIPS_CSV  # noqa: E402
from load_graph import get_driver  # noqa: E402


@pytest.fixture(scope="module")
def driver():
    d = get_driver()
    try:
        d.verify_connectivity()
    except Exception:
        pytest.skip("Neo4j is not reachable -- start it per docs/manual_setup_neo4j.md to run this test")
    yield d
    d.close()


def test_entity_node_count_matches_csv(driver):
    """Assumes kg/load_graph.py has already been run against this database."""
    entities = pd.read_csv(ENTITIES_CSV, keep_default_na=False)
    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN count(e) AS n")
        graph_count = result.single()["n"]
    assert graph_count == len(entities)


def test_relationship_count_matches_csv(driver):
    relationships = pd.read_csv(RELATIONSHIPS_CSV, keep_default_na=False)
    with driver.session() as session:
        result = session.run("MATCH ()-[r]->() RETURN count(r) AS n")
        graph_count = result.single()["n"]
    assert graph_count == len(relationships)
