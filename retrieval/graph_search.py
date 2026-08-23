"""Cypher graph search: given a set of seed entity_ids (from
entity_extraction.py), fetch their neighborhood from Neo4j as a list of
human-readable, cited facts.

Usage:
    python retrieval/graph_search.py E023          # by entity_id
    python retrieval/graph_search.py --query "Who led the resistance at Jhansi?"
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kg"))
from common import (  # noqa: E402
    ENTITIES_CSV,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    build_alias_index,
    get_logger,
)

log = get_logger("graph_search")

DEFAULT_HOPS_LIMIT = 20  # cap facts per seed entity so a hub node doesn't flood the context


def get_driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def one_hop_facts(session, entity_id: str, limit: int = DEFAULT_HOPS_LIMIT) -> list:
    """Return facts for both directions of every edge touching this entity.

    Each fact: {subject, relation, object, direction, source, notes}
    direction is "outgoing" (entity_id is the subject) or "incoming".
    """
    query = """
        MATCH (e:Entity {entity_id: $entity_id})-[r]-(n)
        RETURN e.name AS seed_name,
               n.name AS neighbor_name,
               type(r) AS relation,
               r.source AS source,
               r.notes AS notes,
               startNode(r) = e AS is_outgoing
        LIMIT $limit
    """
    results = session.run(query, entity_id=entity_id, limit=limit)

    facts = []
    for record in results:
        if record["is_outgoing"]:
            subject, obj, direction = record["seed_name"], record["neighbor_name"], "outgoing"
        else:
            subject, obj, direction = record["neighbor_name"], record["seed_name"], "incoming"
        facts.append(
            {
                "subject": subject,
                "relation": record["relation"],
                "object": obj,
                "direction": direction,
                "source": record["source"],
                "notes": record["notes"],
            }
        )
    return facts


def search(session, entity_ids: list, limit_per_entity: int = DEFAULT_HOPS_LIMIT) -> list:
    """Fetch 1-hop facts for every seed entity, de-duplicated."""
    seen = set()
    all_facts = []
    for entity_id in entity_ids:
        for fact in one_hop_facts(session, entity_id, limit=limit_per_entity):
            key = (fact["subject"], fact["relation"], fact["object"])
            if key not in seen:
                seen.add(key)
                all_facts.append(fact)
    return all_facts


def format_fact(fact: dict) -> str:
    return f"{fact['subject']} --{fact['relation']}--> {fact['object']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entity_ids", nargs="*", help="Entity IDs to expand, e.g. E023")
    parser.add_argument("--query", help="Alternatively, a natural-language query to extract entities from first")
    parser.add_argument("--entities", type=Path, default=ENTITIES_CSV)
    parser.add_argument("--limit", type=int, default=DEFAULT_HOPS_LIMIT)
    args = parser.parse_args()

    entity_ids = list(args.entity_ids)
    if args.query:
        from entity_extraction import extract_entities

        entities_df = pd.read_csv(args.entities, keep_default_na=False)
        alias_index = build_alias_index(entities_df)
        entity_ids += extract_entities(args.query, alias_index)

    if not entity_ids:
        log.info("no entity_ids given (pass some, or --query to extract them)")
        sys.exit(1)

    driver = get_driver()
    try:
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        log.info(f"could not connect to Neo4j at {NEO4J_URI}: {exc}")
        sys.exit(1)

    with driver.session() as session:
        facts = search(session, entity_ids, limit_per_entity=args.limit)

    driver.close()

    log.info(f"{len(facts)} fact(s) from {len(entity_ids)} seed entit(y/ies):")
    for fact in facts:
        print(f"  {format_fact(fact)}  [{fact['source']}]")


if __name__ == "__main__":
    main()
