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


# --- hand review of drafted facts (kg/draft_facts.py) and its use in consolidation ---


def test_reply_schema_only_allows_typed_relations_and_the_sentences_names():
    from draft_facts import NONE, reply_schema

    item = reply_schema(["Delhi", "Nicholson"])["properties"]["facts"]["items"]["properties"]
    assert set(item["relation"]["enum"]) - {NONE} == RELATION_VOCAB - {"associated_with"}
    assert item["subject"]["enum"] == item["object"]["enum"] == ["Delhi", "Nicholson"]


def test_draft_sentence_keeps_only_well_formed_facts_from_the_llm():
    import json

    from draft_facts import draft_sentence

    def fake_chat(messages, **kwargs):
        assert "Delhi was captured by Nicholson." in messages[1]["content"]
        return json.dumps(
            {
                "facts": [
                    {"pair": 1, "subject": "Nicholson", "relation": "captured", "object": "Delhi", "evidence": "captured by Nicholson"},
                    {"pair": 1, "subject": "Nicholson", "relation": "none", "object": "Delhi", "evidence": ""},
                    {"pair": 2, "subject": "Nicholson", "relation": "killed", "object": "Delhi", "evidence": ""},  # no pair 2
                    {"pair": 1, "subject": "Nicholson", "relation": "associated_with", "object": "Delhi", "evidence": ""},
                ]
            }
        )

    facts = draft_sentence("Delhi was captured by Nicholson.", [("Delhi", "Nicholson")], fake_chat)
    assert [(f["subject"], f["relation"], f["object"]) for f in facts] == [("Nicholson", "captured", "Delhi")]


def test_export_turns_a_fix_into_a_rejection_plus_the_corrected_fact():
    from draft_facts import export_rows

    base = {"source": "book.pdf#page=1", "title": "The Great Mutiny (pp. 1-4)", "note": "", "fixed_subject": "", "fixed_relation": "", "fixed_object": ""}
    review = [
        {**base, "review_id": "R1", "subject": "Hodson", "relation": "captured", "object": "Bahadur Shah", "decision": "accept"},
        {**base, "review_id": "R2", "subject": "Campbell", "relation": "administered", "object": "Oudh", "decision": "reject"},
        {**base, "review_id": "R3", "subject": "Delhi", "relation": "besieged", "object": "Archdale Wilson", "decision": "fix",
         "fixed_subject": "Archdale Wilson", "fixed_object": "Delhi"},
        {**base, "review_id": "R4", "subject": "A", "relation": "led", "object": "B", "decision": ""},
        {**base, "review_id": "R5", "subject": "A", "relation": "led", "object": "B", "decision": "maybe"},
    ]
    rows, counts, problems = export_rows(review)
    got = {(r["subject"], r["relation"], r["object"], r["decision"]) for r in rows}
    assert got == {
        ("Hodson", "captured", "Bahadur Shah", "accept"),
        ("Campbell", "administered", "Oudh", "reject"),
        ("Delhi", "besieged", "Archdale Wilson", "reject"),
        ("Archdale Wilson", "besieged", "Delhi", "accept"),
    }
    assert counts["unreviewed"] == 1 and len(problems) == 1


def test_verified_facts_are_read_as_accepted_and_rejected(tmp_path):
    from consolidate_triples import read_verified_facts

    path = tmp_path / "verified_facts.csv"
    path.write_text(
        "subject,relation,object,source,title,decision,note\n"
        "Hodson,captured,Bahadur Shah,book.pdf#page=1,The Great Mutiny (pp. 1-4),accept,\n"
        "Campbell,ruled,Oudh,book.pdf#page=2,The Great Mutiny (pp. 5-8),reject,a different Campbell\n"
        "Nobody,led,Delhi,book.pdf#page=3,The Great Mutiny (pp. 9-12),accept,\n",
        encoding="utf-8",
    )
    lexicon = {"Hodson": "Person", "Bahadur Shah": "Person", "Campbell": "Person", "Oudh": "Location", "Delhi": "Location"}
    accepted, rejected = read_verified_facts(path, lexicon)
    assert [(a[0], a[2], a[3], a[5][0]) for a in accepted] == [("Hodson", "captured", "Bahadur Shah", "verified")]
    assert rejected == {("Campbell", "ruled", "Oudh")}


def test_rani_is_renamed_and_findable_by_her_other_names():
    from consolidate_triples import EXTRA_ALIASES, RENAMES

    assert RENAMES["Rani"] == "Rani of Jhansi"
    assert {"Rani", "Lakshmibai"} <= set(EXTRA_ALIASES["Rani of Jhansi"])


def test_event_names_are_cut_back_to_the_known_place_or_dropped():
    from consolidate_triples import tidy_event_name

    lexicon = {"Lucknow": "Location"}

    def cand(subject, place):
        return {"rule": "event_name", "subject": subject, "relation": "located_in", "object": place}

    fixed = tidy_event_name(cand("Relief of Lucknow Havelock", "Lucknow Havelock"), lexicon)
    assert (fixed["subject"], fixed["object"]) == ("Relief of Lucknow", "Lucknow")
    assert tidy_event_name(cand("Massacre of Firinghis'.14 But", "Firinghis'.14 But"), lexicon) is None
    assert tidy_event_name(cand("Mutiny at Mhow", "Mhow"), lexicon)["subject"] == "Mutiny at Mhow"  # unknown but clean: kept


def test_officials_administer_rather_than_rule():
    from extract_facts import TITLE_RELATION

    assert TITLE_RELATION["chief commissioner"] == TITLE_RELATION["commissioner"] == "administered"
    assert TITLE_RELATION["king"] == "ruled"
