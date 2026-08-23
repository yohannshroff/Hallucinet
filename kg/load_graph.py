"""Load the hand-built KG seed spreadsheet (entities.csv + relationships.csv)
into Neo4j.

Design decisions (see docs/schema.md and docs/week3_notes.md):
- Every entity becomes a :Entity node, plus a second label matching its
  entity_type (:Person / :Location / :Event / :Organization / :Cause) so
  Week 4 Cypher queries can filter by type cheaply.
- A relationship's `object` usually resolves to another entity, but per the
  schema some objects are legitimate free-text phrases (e.g. "rebel
  sepoys"). Rather than dropping those facts, they become :Concept nodes
  keyed by their text -- nothing from the curated spreadsheet is lost, and
  a Concept can later be promoted to a real Entity by adding it to
  entities.csv and re-running this script.
- Both node and relationship loads use MERGE, so re-running this script
  after editing the CSVs is safe and idempotent.

Usage:
    python kg/load_graph.py
    python kg/load_graph.py --entities data/kg_seed/entities.csv \
        --relationships data/kg_seed/relationships.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import (  # noqa: E402
    ENTITIES_CSV,
    ENTITY_TYPES,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    RELATION_VOCAB,
    RELATIONSHIPS_CSV,
    build_alias_index,
    get_logger,
)

log = get_logger("load_graph")


def get_driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def ensure_constraints(session):
    session.run("CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE")
    session.run("CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")


def load_entities(session, entities_df: pd.DataFrame) -> int:
    count = 0
    for _, row in entities_df.iterrows():
        entity_type = row["entity_type"]
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"entity {row['entity_id']} has unknown entity_type '{entity_type}'")

        aliases = [a.strip() for a in str(row.get("aliases", "")).split(";") if a.strip()]

        # entity_type is validated against a fixed vocabulary above, so it's
        # safe to interpolate as a Cypher label here.
        query = f"""
            MERGE (e:Entity:{entity_type} {{entity_id: $entity_id}})
            SET e.name = $name,
                e.aliases = $aliases,
                e.period_tag = $period_tag,
                e.description = $description
        """
        session.run(
            query,
            entity_id=row["entity_id"],
            name=row["name"],
            aliases=aliases,
            period_tag=row["period_tag"],
            description=row.get("description", ""),
        )
        count += 1
    return count


def load_relationships(session, relationships_df: pd.DataFrame, alias_index: dict) -> dict:
    stats = {"entity_to_entity": 0, "entity_to_concept": 0, "skipped_unresolved_subject": 0}

    for i, row in relationships_df.iterrows():
        relation = row["relation"]
        if relation not in RELATION_VOCAB:
            raise ValueError(f"relationships.csv row {i + 2}: unknown relation '{relation}' -- add it to RELATION_VOCAB in scripts/common.py and docs/schema.md")

        subject_key = str(row["subject"]).strip().lower()
        object_key = str(row["object"]).strip().lower()

        subject_id = alias_index.get(subject_key)
        if subject_id is None:
            log.info(f"  skipping row {i + 2}: unresolved subject '{row['subject']}' (run validate_kg_csv.py first)")
            stats["skipped_unresolved_subject"] += 1
            continue

        object_id = alias_index.get(object_key)

        # relation is validated against a fixed vocabulary above, so it's
        # safe to interpolate as a Cypher relationship type here.
        if object_id is not None:
            query = f"""
                MATCH (a:Entity {{entity_id: $subject_id}})
                MATCH (b:Entity {{entity_id: $object_id}})
                MERGE (a)-[r:{relation}]->(b)
                SET r.source = $source, r.period_tag = $period_tag, r.notes = $notes
            """
            session.run(
                query,
                subject_id=subject_id,
                object_id=object_id,
                source=row.get("source", ""),
                period_tag=row.get("period_tag", ""),
                notes=row.get("notes", ""),
            )
            stats["entity_to_entity"] += 1
        else:
            query = f"""
                MATCH (a:Entity {{entity_id: $subject_id}})
                MERGE (b:Concept {{name: $object_text}})
                MERGE (a)-[r:{relation}]->(b)
                SET r.source = $source, r.period_tag = $period_tag, r.notes = $notes
            """
            session.run(
                query,
                subject_id=subject_id,
                object_text=str(row["object"]).strip(),
                source=row.get("source", ""),
                period_tag=row.get("period_tag", ""),
                notes=row.get("notes", ""),
            )
            stats["entity_to_concept"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", type=Path, default=ENTITIES_CSV)
    parser.add_argument("--relationships", type=Path, default=RELATIONSHIPS_CSV)
    args = parser.parse_args()

    entities_df = pd.read_csv(args.entities, keep_default_na=False)
    relationships_df = pd.read_csv(args.relationships, keep_default_na=False)
    alias_index = build_alias_index(entities_df)

    driver = get_driver()
    try:
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        log.info(f"could not connect to Neo4j at {NEO4J_URI}: {exc}")
        log.info("see docs/manual_setup_neo4j.md -- is the database running?")
        sys.exit(1)

    with driver.session() as session:
        ensure_constraints(session)
        entity_count = load_entities(session, entities_df)
        log.info(f"loaded {entity_count} entity nodes")

        stats = load_relationships(session, relationships_df, alias_index)
        log.info(
            f"loaded {stats['entity_to_entity']} entity-to-entity relationships, "
            f"{stats['entity_to_concept']} entity-to-concept relationships "
            f"({stats['skipped_unresolved_subject']} rows skipped)"
        )

    driver.close()
    log.info("done -- run kg/validate_graph.py to spot-check the loaded graph")


if __name__ == "__main__":
    main()
