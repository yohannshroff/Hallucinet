"""Tests for scripts/validate_kg_csv.py's validation logic, using small
in-memory fixtures rather than the full seed CSVs."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from validate_kg_csv import build_alias_index, validate  # noqa: E402


def make_entities():
    return pd.DataFrame(
        [
            {"entity_id": "E001", "name": "Rani Lakshmibai", "entity_type": "Person", "aliases": "Rani of Jhansi;Lakshmi Bai", "period_tag": "1857_revolt", "description": "..."},
            {"entity_id": "E002", "name": "Jhansi", "entity_type": "Location", "aliases": "", "period_tag": "1857_revolt", "description": "..."},
        ]
    )


def test_build_alias_index_includes_name_and_aliases():
    entities = make_entities()
    index = build_alias_index(entities)
    assert index["rani lakshmibai"] == "E001"
    assert index["rani of jhansi"] == "E001"
    assert index["lakshmi bai"] == "E001"
    assert index["jhansi"] == "E002"


def test_validate_flags_unresolved_subject():
    entities = make_entities()
    relationships = pd.DataFrame(
        [
            {"subject": "Rani Lakshmibai", "relation": "ruled", "object": "Jhansi", "source": "x", "period_tag": "1857_revolt", "notes": ""},
            {"subject": "Rani Lakshmibi", "relation": "ruled", "object": "Jhansi", "source": "x", "period_tag": "1857_revolt", "notes": ""},  # typo'd subject
        ]
    )
    problems = validate(entities, relationships)
    assert any("unresolved subject" in p for p in problems)
    # the correctly-spelled row should not itself trigger a problem
    assert not any("Rani Lakshmibai'" in p for p in problems)


def test_validate_allows_free_text_object():
    """Objects that are short phrases (not entities) should not fail validation."""
    entities = make_entities()
    relationships = pd.DataFrame(
        [
            {"subject": "Rani Lakshmibai", "relation": "fought_against", "object": "British forces", "source": "x", "period_tag": "1857_revolt", "notes": ""},
        ]
    )
    problems = validate(entities, relationships)
    assert problems == []


def test_validate_flags_duplicate_entity_id():
    entities = pd.concat([make_entities(), make_entities().iloc[[0]]], ignore_index=True)
    relationships = pd.DataFrame(columns=["subject", "relation", "object", "source", "period_tag", "notes"])
    problems = validate(entities, relationships)
    assert any("duplicate entity_id" in p for p in problems)
