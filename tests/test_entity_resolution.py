"""Tests for kg/entity_resolution.py's exact + fuzzy matching."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kg"))

from common import build_alias_index  # noqa: E402

pytest.importorskip("fuzzywuzzy")
from entity_resolution import resolve_entity  # noqa: E402


def make_alias_index():
    entities = pd.DataFrame(
        [
            {"entity_id": "E001", "name": "Rani Lakshmibai", "aliases": "Rani of Jhansi;Lakshmi Bai"},
            {"entity_id": "E002", "name": "Jhansi", "aliases": ""},
        ]
    )
    return build_alias_index(entities)


def test_resolve_exact_match():
    index = make_alias_index()
    entity_id, method, score = resolve_entity("Rani of Jhansi", index)
    assert entity_id == "E001"
    assert method == "exact"
    assert score == 100


def test_resolve_exact_match_is_case_insensitive():
    index = make_alias_index()
    entity_id, method, score = resolve_entity("JHANSI", index)
    assert entity_id == "E002"
    assert method == "exact"


def test_resolve_fuzzy_match_for_typo():
    index = make_alias_index()
    # "Rani Laxmibai" is a common alternate spelling/typo of "Rani Lakshmibai"
    entity_id, method, score = resolve_entity("Rani Laxmibai", index)
    assert entity_id == "E001"
    assert method == "fuzzy"
    assert score >= 85


def test_resolve_returns_none_for_unrelated_text():
    index = make_alias_index()
    entity_id, method, score = resolve_entity("completely unrelated phrase xyz", index)
    assert entity_id is None
    assert method is None
