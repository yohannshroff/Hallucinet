"""Tests for generation/prompt_template.py -- pure formatting logic, no
live services needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generation"))

from prompt_template import SYSTEM_PROMPT, build_messages  # noqa: E402


def test_system_prompt_forbids_outside_knowledge():
    assert "ONLY" in SYSTEM_PROMPT
    assert "outside knowledge" in SYSTEM_PROMPT.lower()


def test_system_prompt_instructs_explicit_refusal():
    assert "does not contain enough information" in SYSTEM_PROMPT


def test_build_messages_has_system_and_user_roles():
    messages = build_messages("Who ruled Jhansi?", "Graph facts:\n- Rani Lakshmibai --ruled--> Jhansi [src]")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["role"] == "user"


def test_build_messages_includes_question_and_context():
    context = "Graph facts:\n- Rani Lakshmibai --ruled--> Jhansi [src]"
    messages = build_messages("Who ruled Jhansi?", context)
    user_content = messages[1]["content"]
    assert "Who ruled Jhansi?" in user_content
    assert context in user_content
