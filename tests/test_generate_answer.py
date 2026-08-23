"""End-to-end test for generation/generate_answer.py against live Neo4j +
FAISS index + Ollama. Skipped automatically if any of those aren't
available. Because LLM output isn't perfectly deterministic even at low
temperature, assertions are intentionally loose (structural + a light
content check), not exact-match."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generation"))

from common import ENTITIES_CSV, FAISS_INDEX_FILE  # noqa: E402
from ollama_client import is_reachable  # noqa: E402

pytestmark = pytest.mark.skipif(not FAISS_INDEX_FILE.exists(), reason="FAISS index not built -- run scripts/build_faiss_index.py first")


@pytest.fixture(scope="module", autouse=True)
def require_live_services():
    pytest.importorskip("neo4j")
    if not is_reachable():
        pytest.skip("Ollama is not reachable -- start it per docs/manual_setup_ollama.md")


@pytest.fixture(scope="module")
def entities_df():
    return pd.read_csv(ENTITIES_CSV, keep_default_na=False)


def test_answer_question_returns_expected_structure(entities_df):
    from generate_answer import answer_question

    result = answer_question("Who led the resistance at Jhansi?", entities_df, k=3)

    assert set(result.keys()) == {"question", "answer", "context", "bundle"}
    assert isinstance(result["answer"], str) and len(result["answer"]) > 0
    assert "Rani Lakshmibai" in result["context"]  # graph fact should be in the context fed to the model


def test_answer_grounded_in_context_for_in_scope_question(entities_df):
    from generate_answer import answer_question

    result = answer_question("Who ruled Jhansi before the revolt?", entities_df, k=3)
    assert "Lakshmibai" in result["answer"] or "Rani" in result["answer"]


def test_answer_declines_out_of_scope_question(entities_df):
    """Napoleon has nothing to do with our KG/sources -- the model should
    say it can't answer rather than inventing a connection."""
    from generate_answer import answer_question

    result = answer_question("What was Napoleon's role in the Indian Rebellion of 1857?", entities_df, k=3)
    answer_lower = result["answer"].lower()
    assert "napoleon" not in answer_lower or "not" in answer_lower or "no information" in answer_lower or "does not contain" in answer_lower
