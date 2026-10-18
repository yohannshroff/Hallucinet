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
def entities_df():
    return pd.read_csv(ENTITIES_CSV, keep_default_na=False)


@pytest.fixture(scope="module")
def alias_index(entities_df):
    return build_alias_index(entities_df)


def id_of(entities_df, name):
    """The knowledge graph is rebuilt from the book, so IDs aren't stable -- look them up by name."""
    rows = entities_df.loc[entities_df["name"] == name, "entity_id"]
    assert len(rows) == 1, f"expected exactly one entity named {name!r}, found {len(rows)}"
    return rows.iloc[0]


def test_direct_substring_match_finds_exact_mention(alias_index, entities_df):
    matches = direct_substring_matches("Who ruled Jhansi before the revolt?", alias_index)
    assert id_of(entities_df, "Jhansi") in matches


def test_extract_entities_finds_multiple_known_entities(alias_index, entities_df):
    entity_ids = extract_entities("Why did Nana Sahib attack the garrison at Cawnpore?", alias_index)
    assert id_of(entities_df, "Nana Sahib") in entity_ids
    assert id_of(entities_df, "Cawnpore") in entity_ids


def test_extract_entities_handles_multi_word_event_name(alias_index, entities_df):
    """'Siege of Delhi' gets split across noun chunks by spaCy -- the direct
    substring scan should still catch the whole name."""
    entity_ids = extract_entities("What happened during the Siege of Delhi?", alias_index)
    assert id_of(entities_df, "Siege of Delhi") in entity_ids


def test_extract_entities_empty_for_unrelated_query(alias_index):
    entity_ids = extract_entities("What is the weather like today?", alias_index)
    assert entity_ids == []
