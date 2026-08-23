"""Integration tests for api/main.py using FastAPI's in-process TestClient
(no separate server process needed). Skipped automatically if Neo4j, the
FAISS index, or Ollama aren't available -- same pattern as the other
live-service tests in this suite."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generation"))

from common import FAISS_INDEX_FILE  # noqa: E402

pytestmark = pytest.mark.skipif(not FAISS_INDEX_FILE.exists(), reason="FAISS index not built -- run scripts/build_faiss_index.py first")


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("neo4j")
    pytest.importorskip("fastapi")
    from ollama_client import is_reachable

    if not is_reachable():
        pytest.skip("Ollama is not reachable -- start it per docs/manual_setup_ollama.md")

    from fastapi.testclient import TestClient
    from main import app

    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["ollama_reachable"] is True


def test_ask_endpoint_returns_expected_shape(client):
    resp = client.post("/ask", json={"question": "Who led the resistance at Jhansi?", "k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {
        "question", "answer", "trust_score", "n_claims", "n_entailed", "seed_entities", "sources",
    }
    assert len(data["answer"]) > 0
    assert "Jhansi" in data["seed_entities"] or any("Jhansi" in s for s in data["seed_entities"])


def test_ask_endpoint_respects_mode_param(client):
    """Source isolation between modes is already verified at the retrieve()
    level in test_hybrid_retrieve.py::test_retrieve_modes_isolate_sources --
    this just checks the API surface accepts and threads the parameter
    through without erroring."""
    resp = client.post("/ask", json={"question": "Who led the resistance at Jhansi?", "mode": "vector", "k": 3})
    assert resp.status_code == 200
    assert len(resp.json()["answer"]) > 0
