"""Resolve a free-text mention (e.g. from a relationships.csv row, or later
a user query in Week 4) to a known entity_id.

Two-stage resolution:
1. Exact match against the name/alias index built from entities.csv
   (common.build_alias_index) -- handles the common case cheaply.
2. Fuzzy fallback via fuzzywuzzy for near-misses (typos, minor spelling
   variants) that don't hit the exact index, e.g. "Rani Lakshmibai" vs
   "Rani Laxmibai".

Usage:
    python kg/entity_resolution.py "Rani Laxmibai"
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import ENTITIES_CSV, build_alias_index, get_logger  # noqa: E402

log = get_logger("entity_resolution")

FUZZY_MATCH_THRESHOLD = 85  # 0-100; fuzzywuzzy score below this is not considered a match


def resolve_entity(mention: str, alias_index: dict, threshold: int = FUZZY_MATCH_THRESHOLD):
    """Resolve `mention` to an entity_id, or None if no confident match.

    Returns (entity_id, method, score) where method is "exact" or "fuzzy",
    and score is 100 for exact matches or the fuzzy match score otherwise.
    """
    key = mention.strip().lower()
    if key in alias_index:
        return alias_index[key], "exact", 100

    from fuzzywuzzy import process

    match = process.extractOne(key, alias_index.keys())
    if match is None:
        return None, None, 0
    matched_key, score = match
    if score >= threshold:
        return alias_index[matched_key], "fuzzy", score
    return None, None, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mention", help="Free-text name to resolve, e.g. 'Rani Laxmibai'")
    parser.add_argument("--entities", type=Path, default=ENTITIES_CSV)
    parser.add_argument("--threshold", type=int, default=FUZZY_MATCH_THRESHOLD)
    args = parser.parse_args()

    entities_df = pd.read_csv(args.entities, keep_default_na=False)
    alias_index = build_alias_index(entities_df)

    entity_id, method, score = resolve_entity(args.mention, alias_index, threshold=args.threshold)

    if entity_id is None:
        log.info(f"no confident match for '{args.mention}' (best fuzzy score: {score})")
        sys.exit(1)

    name = entities_df.loc[entities_df["entity_id"] == entity_id, "name"].iloc[0]
    log.info(f"'{args.mention}' -> {entity_id} ({name})  [{method} match, score={score}]")


if __name__ == "__main__":
    main()
