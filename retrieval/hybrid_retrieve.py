"""Hybrid retrieval: entity extraction -> graph search + vector search (run
independently, not dependent on each other) -> merged context.

Merge/rank policy (kept simple on purpose, per the master plan -- this is
not meant to be a sophisticated reranker):
- Graph facts come first: they're precise, structured, and directly tied to
  the entities the query is actually about.
- Vector chunks come second: broader supporting context/prose, ranked by
  cosine similarity, capped at `k`.
- No cross-source scoring/fusion -- Week 7's evaluation (vector-only vs
  graph-only vs hybrid) is exactly what tells us whether this simple
  concatenation is good enough or needs a real fusion strategy.

Usage:
    python retrieval/hybrid_retrieve.py "Who led the resistance at Jhansi?"
    python retrieval/hybrid_retrieve.py "Who led the resistance at Jhansi?" --k 5 --graph_limit 20
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import ENTITIES_CSV, FAISS_INDEX_DIR, build_alias_index, get_logger  # noqa: E402

from entity_extraction import extract_entities  # noqa: E402
from graph_search import DEFAULT_HOPS_LIMIT, format_fact, get_driver as get_neo4j_driver, search as graph_search  # noqa: E402
from vector_search import search as vector_search  # noqa: E402

log = get_logger("hybrid_retrieve")


def retrieve(
    query: str,
    entities_df: pd.DataFrame,
    k: int = 5,
    graph_limit: int = DEFAULT_HOPS_LIMIT,
    index_dir: Path = FAISS_INDEX_DIR,
) -> dict:
    """Run entity extraction, graph search, and vector search; return a
    merged context bundle. Returns None for graph_facts/seed_entities if
    Neo4j isn't reachable, so callers can still get vector-only results.
    """
    alias_index = build_alias_index(entities_df)
    seed_entity_ids = extract_entities(query, alias_index)

    graph_facts = []
    if seed_entity_ids:
        driver = get_neo4j_driver()
        try:
            driver.verify_connectivity()
            with driver.session() as session:
                graph_facts = graph_search(session, seed_entity_ids, limit_per_entity=graph_limit)
        except Exception as exc:  # noqa: BLE001
            log.info(f"graph search skipped (Neo4j unreachable: {exc})")
        finally:
            driver.close()

    vector_chunks = vector_search(query, index_dir=index_dir, k=k)

    seed_names = [
        entities_df.loc[entities_df["entity_id"] == eid, "name"].iloc[0] for eid in seed_entity_ids
    ]

    return {
        "query": query,
        "seed_entities": list(zip(seed_entity_ids, seed_names)),
        "graph_facts": graph_facts,
        "vector_chunks": vector_chunks,
    }


def format_context_only(bundle: dict) -> str:
    """Render just the retrieved context (graph facts + passages), with no
    query line -- used as the context block inside an LLM prompt (Week 5),
    where the question is already provided separately."""
    lines = []

    if bundle["graph_facts"]:
        lines.append("Graph facts:")
        for fact in bundle["graph_facts"]:
            lines.append(f"- {format_fact(fact)} [{fact['source']}]")
        lines.append("")

    if bundle["vector_chunks"]:
        lines.append("Retrieved passages:")
        for i, chunk in enumerate(bundle["vector_chunks"], start=1):
            lines.append(f"[{i}] {chunk['title']} ({chunk['source_url']})")
            lines.append(f"    {chunk['text']}")
        lines.append("")

    if not bundle["graph_facts"] and not bundle["vector_chunks"]:
        lines.append("(no context retrieved)")

    return "\n".join(lines)


def format_context_for_llm(bundle: dict) -> str:
    """Like format_context_only, but with a leading query line -- used for
    the CLI's own human-readable display."""
    return f"Query: {bundle['query']}\n\n" + format_context_only(bundle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5, help="Top-k vector chunks")
    parser.add_argument("--graph_limit", type=int, default=DEFAULT_HOPS_LIMIT, help="Max facts per seed entity")
    parser.add_argument("--entities", type=Path, default=ENTITIES_CSV)
    parser.add_argument("--index_dir", type=Path, default=FAISS_INDEX_DIR)
    args = parser.parse_args()

    entities_df = pd.read_csv(args.entities, keep_default_na=False)
    bundle = retrieve(args.query, entities_df, k=args.k, graph_limit=args.graph_limit, index_dir=args.index_dir)

    log.info(f"seed entities: {bundle['seed_entities']}")
    log.info(f"{len(bundle['graph_facts'])} graph fact(s), {len(bundle['vector_chunks'])} vector chunk(s)")
    print()
    print(format_context_for_llm(bundle))


if __name__ == "__main__":
    main()
