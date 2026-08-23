# kg/ (Week 3)

Entity resolution and Neo4j ingestion.

- `entity_resolution.py` — resolves a free-text mention to an `entity_id`:
  exact match against `data/kg_seed/entities.csv`'s name/alias index first,
  then a fuzzy (`fuzzywuzzy`) fallback for typos/spelling variants. Reused
  by `load_graph.py` now and by Week 4's query-time entity extraction.
- `load_graph.py` — loads `entities.csv`/`relationships.csv` into Neo4j.
  Entities become `:Entity` nodes (plus a type label: `:Person`,
  `:Location`, `:Event`, `:Organization`, `:Cause`). Relationships whose
  `object` resolves to an entity become normal entity-to-entity edges;
  relationships whose `object` is a free-text phrase (per `docs/schema.md`)
  become edges to a `:Concept` node instead, so no row from the curated
  spreadsheet is dropped. Idempotent (`MERGE`-based) — safe to re-run after
  editing the CSVs.
- `validate_graph.py` — structural summary (node/relationship counts) plus
  a round-trip spot-check that sampled (or, with `--all`, every) row from
  `relationships.csv` actually made it into the graph correctly.

See [docs/week3_notes.md](../docs/week3_notes.md) for what was run and
verified, and [docs/manual_setup_neo4j.md](../docs/manual_setup_neo4j.md)
for getting a Neo4j instance running.
