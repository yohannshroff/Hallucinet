"""Offline tests for the book-to-graph build (kg/consolidate_triples.py,
kg/extract_facts.py): name merging, the relation type table, and grammar
extraction on a synthetic sentence."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for sub in ("scripts", "kg", "eval"):
    sys.path.insert(0, str(ROOT / sub))

from common import RELATION_VOCAB  # noqa: E402
from consolidate_triples import TYPE_RULES, merge_name_variants  # noqa: E402


def test_every_relation_has_a_type_rule():
    assert set(TYPE_RULES) == RELATION_VOCAB


def test_surname_merges_into_the_only_matching_full_name():
    names = {"Person": ["James Outram", "Outram"]}
    canon = merge_name_variants(names, {"James Outram": 15, "Outram": 40})
    assert canon["Outram"] == canon["James Outram"] == "James Outram"


def test_surname_not_absorbed_by_a_rare_namesake():
    # "Canning" (39 mentions) is mostly the Governor-General, not the one rare "Charlotte Canning"
    names = {"Person": ["Charlotte Canning", "Canning"]}
    canon = merge_name_variants(names, {"Charlotte Canning": 4, "Canning": 39})
    assert canon["Canning"] == "Canning"


def test_lady_is_not_stripped_so_lady_canning_stays_distinct_from_canning():
    names = {"Person": ["Lady Canning", "Canning"]}
    canon = merge_name_variants(names, {"Lady Canning": 15, "Canning": 39})
    assert canon["Lady Canning"] != canon["Canning"]


def test_event_names_with_different_places_are_never_fuzzy_merged():
    names = {"Event": ["Mutiny at Indore", "Mutiny at Dinapore"]}
    canon = merge_name_variants(names, {"Mutiny at Indore": 3, "Mutiny at Dinapore": 3})
    assert canon["Mutiny at Indore"] != canon["Mutiny at Dinapore"]


def test_spelling_variant_of_a_person_is_merged():
    names = {"Person": ["Wheeler", "Wheler"]}
    canon = merge_name_variants(names, {"Wheeler": 21, "Wheler": 3})
    assert canon["Wheler"] == "Wheeler"


def test_verb_facts_active_and_passive():
    spacy = pytest.importorskip("spacy")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy model en_core_web_sm not installed")
    from extract_facts import verb_facts

    def facts_for(text, surfaces):
        doc = nlp(text)
        roots = {}
        for surface in surfaces:
            start = text.index(surface)
            span = doc.char_span(start, start + len(surface), alignment_mode="expand")
            roots[span.root.i] = surface
        return {(s, r, o) for s, r, o, _ in verb_facts(next(doc.sents), roots)}

    assert ("Hugh Rose", "besieged", "Jhansi") in facts_for("Hugh Rose besieged Jhansi.", ["Hugh Rose", "Jhansi"])
    assert ("Nicholson", "captured", "Delhi") in facts_for("Delhi was captured by Nicholson.", ["Delhi", "Nicholson"])
