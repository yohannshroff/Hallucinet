# kg/ (Part 3)

Building the knowledge graph from the book, and loading it into Neo4j. The
build pipeline (`find_candidates.py` -> `type_entities.py` -> `extract_facts.py`
-> `draft_facts.py` (LLM-drafted typed facts, reviewed by hand and exported to
`data/kg_seed/verified_facts.csv`) -> `consolidate_triples.py`, with
`book_text.py` as a helper) is described in
[docs/kg_build.md](../docs/kg_build.md); the loader and resolution code:

- `entity_resolution.py` — resolves a free-text mention to an `entity_id`:
  exact match against `data/kg_seed/entities.csv`'s name/alias index first,
  then a fuzzy (`fuzzywuzzy`) fallback for typos/spelling variants. Reused
  by `load_graph.py` now and by Part 4's query-time entity extraction.
- `load_graph.py` — loads `entities.csv`/`relationships.csv` into Neo4j.
  Entities become `:Entity` nodes (plus a type label: `:Person`,
  `:Location`, `:Event`, `:Organization`, `:Cause`). Relationships whose
  `object` resolves to an entity become normal entity-to-entity edges;
  relationships whose `object` is a free-text phrase (per `docs/schema.md`)
  become edges to a `:Concept` node instead, so no row from the curated
  spreadsheet is dropped. Idempotent (`MERGE`-based); use `--reset` to wipe the
  existing graph first when the CSVs were rebuilt (node ids change).
- `validate_graph.py` — structural summary (node/relationship counts) plus
  a round-trip spot-check that sampled (or, with `--all`, every) row from
  `relationships.csv` actually made it into the graph correctly.

See [docs/part3_notes.md](../docs/part3_notes.md) for what was run and
verified, and [docs/manual_setup_neo4j.md](../docs/manual_setup_neo4j.md)
for getting a Neo4j instance running.
