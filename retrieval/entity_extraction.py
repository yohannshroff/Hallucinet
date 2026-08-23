"""Extract known KG entities mentioned in a user query.

Generic spaCy NER is unreliable on this niche domain out of the box (it
mislabels "Barrackpore" as PERSON, misses "Mangal Pandey" as an entity
entirely, and splits "Doctrine of Lapse" into two separate noun chunks --
verified empirically, see docs/week4_notes.md). Since our KG only has ~50
entities, we combine two complementary signals instead of relying on
spaCy's NER labels alone:

1. Direct substring scan -- every known entity name/alias, checked as a
   case-insensitive substring of the query. Cheap at this scale and catches
   exact/alias mentions with no false negatives from NER mislabeling.
2. spaCy noun chunks + named entities as candidate spans, each resolved via
   kg.entity_resolution.resolve_entity (exact + fuzzy). Catches paraphrases
   or minor spelling variants that don't appear verbatim in entities.csv.

Usage:
    python retrieval/entity_extraction.py "Who led the resistance at Jhansi?"
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kg"))
from common import ENTITIES_CSV, build_alias_index, get_logger  # noqa: E402
from entity_resolution import resolve_entity  # noqa: E402

log = get_logger("entity_extraction")

SPACY_MODEL_NAME = "en_core_web_sm"
MIN_SUBSTRING_ALIAS_LENGTH = 4  # skip very short aliases to avoid noisy substring matches

_nlp = None  # lazy-loaded singleton, spaCy model load is slow (~1s)


def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load(SPACY_MODEL_NAME)
    return _nlp


def direct_substring_matches(query: str, alias_index: dict) -> set:
    query_lower = query.lower()
    matches = set()
    for alias, entity_id in alias_index.items():
        if len(alias) >= MIN_SUBSTRING_ALIAS_LENGTH and alias in query_lower:
            matches.add(entity_id)
    return matches


def candidate_spans(query: str) -> list:
    """spaCy noun chunks + named entities as candidate spans for fuzzy resolution."""
    doc = get_nlp()(query)
    spans = set()
    for ent in doc.ents:
        spans.add(ent.text)
    for chunk in doc.noun_chunks:
        spans.add(chunk.text)
    return list(spans)


def extract_entities(query: str, alias_index: dict, fuzzy_threshold: int = 90) -> list:
    """Return a de-duplicated list of entity_ids mentioned in `query`.

    fuzzy_threshold is set higher than entity_resolution's default (85)
    because query text is noisier than a curated relationships.csv row --
    a stricter threshold here avoids resolving unrelated question words to
    an unrelated entity by coincidence.
    """
    found = direct_substring_matches(query, alias_index)

    for span in candidate_spans(query):
        entity_id, method, score = resolve_entity(span, alias_index, threshold=fuzzy_threshold)
        if entity_id is not None:
            found.add(entity_id)

    return list(found)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="User question to extract entities from")
    parser.add_argument("--entities", type=Path, default=ENTITIES_CSV)
    args = parser.parse_args()

    entities_df = pd.read_csv(args.entities, keep_default_na=False)
    alias_index = build_alias_index(entities_df)

    entity_ids = extract_entities(args.query, alias_index)

    log.info(f"query: {args.query!r}")
    if not entity_ids:
        log.info("no known entities found")
        return
    for entity_id in entity_ids:
        name = entities_df.loc[entities_df["entity_id"] == entity_id, "name"].iloc[0]
        log.info(f"  {entity_id}: {name}")


if __name__ == "__main__":
    main()
