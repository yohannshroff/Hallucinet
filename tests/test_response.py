"""Test for api/response.py's build_ask_response -- the shared logic used
by both the FastAPI endpoint and the demo-cache builder. Skipped
automatically if Neo4j, the FAISS index, or Ollama aren't available."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generation"))

from common import ENTITIES_CSV, FAISS_INDEX_FILE  # noqa: E402

pytestmark = pytest.mark.skipif(not FAISS_INDEX_FILE.exists(), reason="FAISS index not built -- run scripts/build_faiss_index.py first")


@pytest.fixture(scope="module", autouse=True)
def require_live_services():
    pytest.importorskip("neo4j")
    from ollama_client import is_reachable

    if not is_reachable():
        pytest.skip("Ollama is not reachable -- start it per docs/manual_setup_ollama.md")


def test_build_ask_response_shape():
    from response import build_ask_response

    entities_df = pd.read_csv(ENTITIES_CSV, keep_default_na=False)
    result = build_ask_response("Who led the resistance at Jhansi?", entities_df, k=3)

    assert set(result.keys()) == {
        "question", "answer", "trust_score", "n_claims", "n_entailed", "seed_entities", "sources",
    }
    assert len(result["sources"]) > 0
    assert all({"title", "url"} == set(s.keys()) for s in result["sources"])
