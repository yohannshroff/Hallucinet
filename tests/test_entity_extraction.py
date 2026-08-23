"""Tests for retrieval/entity_extraction.py, using the real entities.csv
(small enough to load directly) and canned questions rather than a live
Neo4j/FAISS dependency -- extraction only needs the alias index + spaCy."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kg"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))

from common import ENTITIES_CSV, build_alias_index  # noqa: E402

pytest.importorskip("spacy")
from entity_extraction import direct_substring_matches, extract_entities  # noqa: E402


@pytest.fixture(scope="module")
def alias_index():
    entities_df = pd.read_csv(ENTITIES_CSV, keep_default_na=False)
    return build_alias_index(entities_df)


def test_direct_substring_match_finds_exact_mention(alias_index):
    matches = direct_substring_matches("Who ruled Jhansi before the revolt?", alias_index)
    assert "E023" in matches  # Jhansi


def test_extract_entities_finds_multiple_known_entities(alias_index):
    entity_ids = extract_entities("Why did Mangal Pandey attack officers at Barrackpore?", alias_index)
    assert "E003" in entity_ids  # Mangal Pandey
    assert "E025" in entity_ids  # Barrackpore


def test_extract_entities_handles_split_noun_chunk(alias_index):
    """'Doctrine of Lapse' gets split across two noun chunks by spaCy --
    the direct substring scan should still catch it."""
    entity_ids = extract_entities("What was the impact of the Doctrine of Lapse?", alias_index)
    assert "E070" in entity_ids  # Doctrine of Lapse


def test_extract_entities_empty_for_unrelated_query(alias_index):
    entity_ids = extract_entities("What is the weather like today?", alias_index)
    assert entity_ids == []
