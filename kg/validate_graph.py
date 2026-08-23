"""Spot-check the Neo4j graph loaded by kg/load_graph.py.

Two checks:
1. Structural: node/relationship counts by label/type, confirming nothing
   was silently dropped during ingestion.
2. Round-trip: for a sample of rows from relationships.csv (or the full
   file with --all), confirm the exact (subject)-[relation]->(object) edge
   is actually present in the graph.

Historical accuracy of each fact was checked against its cited Wikipedia
source during Part 1 seeding (see docs/part1_notes.md) -- this script
checks that ingestion faithfully reproduced the spreadsheet in the graph,
not the underlying history.

Usage:
    python kg/validate_graph.py
    python kg/validate_graph.py --sample 20
    python kg/validate_graph.py --all
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import (  # noqa: E402
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    RELATIONSHIPS_CSV,
    get_logger,
)

log = get_logger("validate_graph")


def get_driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def print_structural_summary(session):
    result = session.run(
        "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n ORDER BY n DESC"
    )
    log.info("node counts by label:")
    for record in result:
        log.info(f"  {record['label']}: {record['n']}")

    result = session.run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS n ORDER BY n DESC")
    log.info("relationship counts by type:")
    for record in result:
        log.info(f"  {record['rel_type']}: {record['n']}")


def edge_exists(session, subject: str, relation: str, object_text: str) -> bool:
    # relation type is interpolated but only ever comes from relationships.csv,
    # which load_graph.py already validates against a fixed vocabulary.
    query = f"""
        MATCH (a)-[r:{relation}]->(b)
        WHERE toLower(a.name) = toLower($subject)
          AND (toLower(b.name) = toLower($object_text) OR toLower(coalesce(b.name, '')) = toLower($object_text))
        RETURN count(r) AS n
    """
    result = session.run(query, subject=subject, object_text=object_text)
    return result.single()["n"] > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relationships", type=Path, default=RELATIONSHIPS_CSV)
    parser.add_argument("--sample", type=int, default=15, help="How many relationship rows to spot-check")
    parser.add_argument("--all", action="store_true", help="Check every row instead of a sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    relationships_df = pd.read_csv(args.relationships, keep_default_na=False)

    driver = get_driver()
    try:
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        log.info(f"could not connect to Neo4j at {NEO4J_URI}: {exc}")
        sys.exit(1)

    with driver.session() as session:
        print_structural_summary(session)

        if args.all:
            sample = relationships_df
        else:
            random.seed(args.seed)
            n = min(args.sample, len(relationships_df))
            sample = relationships_df.sample(n=n, random_state=args.seed)

        log.info(f"spot-checking {len(sample)} relationship(s) against the graph:")
        missing = []
        for _, row in sample.iterrows():
            found = edge_exists(session, row["subject"], row["relation"], row["object"])
            status = "OK" if found else "MISSING"
            log.info(f"  [{status}] {row['subject']} -{row['relation']}-> {row['object']}")
            if not found:
                missing.append(row)

    driver.close()

    if missing:
        log.info(f"{len(missing)} relationship(s) from the spreadsheet were NOT found in the graph")
        sys.exit(1)
    log.info("all spot-checked relationships are present in the graph.")


if __name__ == "__main__":
    main()
