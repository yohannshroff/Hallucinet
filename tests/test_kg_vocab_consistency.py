"""Guards against drift between the controlled vocabularies in
scripts/common.py, docs/schema.md, and what's actually used in the seed
CSVs -- catches e.g. a new relation type added to relationships.csv without
also adding it to RELATION_VOCAB (which would make kg/load_graph.py raise)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from common import ENTITIES_CSV, ENTITY_TYPES, RELATION_VOCAB, RELATIONSHIPS_CSV  # noqa: E402


def test_all_entity_types_are_known():
    entities = pd.read_csv(ENTITIES_CSV, keep_default_na=False)
    unknown = set(entities["entity_type"]) - ENTITY_TYPES
    assert not unknown, f"entities.csv uses entity_type(s) not in ENTITY_TYPES: {unknown}"


def test_all_relations_are_known():
    relationships = pd.read_csv(RELATIONSHIPS_CSV, keep_default_na=False)
    unknown = set(relationships["relation"]) - RELATION_VOCAB
    assert not unknown, f"relationships.csv uses relation(s) not in RELATION_VOCAB: {unknown}"
