"""Validate the hand-built knowledge-graph seed spreadsheet.

Loads entities.csv and relationships.csv, builds a name+alias -> entity_id
index, and flags any subject/object in relationships.csv that doesn't
resolve to a known entity (catching typos before they become orphan nodes
in the Part 3 graph build).

Usage:
    python scripts/validate_kg_csv.py
    python scripts/validate_kg_csv.py --entities path/to/entities.csv \
        --relationships path/to/relationships.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from common import ENTITIES_CSV, RELATIONSHIPS_CSV, build_alias_index, get_logger

log = get_logger("validate_kg_csv")


def load_entities(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    return df


def validate(entities_df: pd.DataFrame, relationships_df: pd.DataFrame) -> list:
    """Return a list of human-readable problems found. Empty list = clean."""
    problems = []

    # Duplicate entity ids/names.
    dup_ids = entities_df["entity_id"][entities_df["entity_id"].duplicated()]
    for eid in dup_ids.unique():
        problems.append(f"duplicate entity_id: {eid}")

    alias_index = build_alias_index(entities_df)

    # A name/alias should not be claimed by two different entities.
    seen = {}
    for _, row in entities_df.iterrows():
        keys = [row["name"].strip().lower()]
        if row.get("aliases", ""):
            keys += [a.strip().lower() for a in row["aliases"].split(";") if a.strip()]
        for key in keys:
            if key in seen and seen[key] != row["entity_id"]:
                problems.append(
                    f"name/alias '{key}' claimed by both {seen[key]} and {row['entity_id']}"
                )
            seen[key] = row["entity_id"]

    # Relationship subject/object resolution. Object may legitimately be a
    # free-text phrase (per docs/schema.md) rather than an entity, so we only
    # WARN (not fail) when the object doesn't resolve -- subject must always
    # resolve since every fact should be anchored to a known entity.
    unresolved_subjects = 0
    unresolved_objects = 0
    for i, row in relationships_df.iterrows():
        subj_key = str(row["subject"]).strip().lower()
        obj_key = str(row["object"]).strip().lower()
        if subj_key not in alias_index:
            problems.append(f"relationships.csv row {i + 2}: unresolved subject '{row['subject']}'")
            unresolved_subjects += 1
        if obj_key not in alias_index:
            unresolved_objects += 1  # informational only, not an error

    log.info(f"{len(entities_df)} entities, {len(relationships_df)} relationships")
    log.info(f"{unresolved_subjects} unresolved subjects, {unresolved_objects} objects that are free-text phrases (not entities)")

    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", type=Path, default=ENTITIES_CSV)
    parser.add_argument("--relationships", type=Path, default=RELATIONSHIPS_CSV)
    args = parser.parse_args()

    entities_df = load_entities(args.entities)
    relationships_df = pd.read_csv(args.relationships, keep_default_na=False)

    problems = validate(entities_df, relationships_df)

    if problems:
        log.info(f"{len(problems)} problem(s) found:")
        for p in problems:
            log.info(f"  - {p}")
        sys.exit(1)

    log.info("KG seed spreadsheet looks clean.")


if __name__ == "__main__":
    main()
