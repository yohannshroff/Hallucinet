"""Integration tests for generation/ollama_client.py against a live Ollama
instance. Skipped automatically if Ollama isn't reachable or the
configured model hasn't been pulled."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generation"))

from common import OLLAMA_MODEL  # noqa: E402
from ollama_client import chat, is_reachable  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def require_ollama():
    if not is_reachable():
        pytest.skip("Ollama is not reachable -- start it per docs/manual_setup_ollama.md")


def test_chat_returns_nonempty_response():
    reply = chat([{"role": "user", "content": "Reply with exactly the word: PONG"}])
    assert isinstance(reply, str)
    assert len(reply.strip()) > 0


def test_chat_respects_system_message():
    messages = [
        {"role": "system", "content": "No matter what is asked, reply with exactly: SYSTEM_OVERRIDE_TEST"},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    reply = chat(messages)
    assert "SYSTEM_OVERRIDE_TEST" in reply
