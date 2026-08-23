# Week 3 notes

Goals: load the hand-built KG seed into Neo4j, add alias normalization, and
validate the graph.

- Neo4j: installed locally via Homebrew (`docs/manual_setup_neo4j.md`,
  Option A), running as a `brew services` background daemon on the
  standard bolt port (`bolt://localhost:7687`).
- Entity resolution: `kg/entity_resolution.py` — exact name/alias match
  first (via `common.build_alias_index`, shared with
  `scripts/validate_kg_csv.py`), fuzzy (`fuzzywuzzy`) fallback for
  spelling variants like "Rani Laxmibai" vs "Rani Lakshmibai".
- Graph load: `kg/load_graph.py` loaded all 47 entities and all 50
  relationships from the Week 1 seed spreadsheet with zero skipped rows —
  47 as entity-to-entity edges, 3 as entity-to-Concept edges (the
  free-text-object rows: "rebel sepoys", "rebel forces in Delhi", "sons of
  Bahadur Shah II").
- Validation: `kg/validate_graph.py --all` confirmed all 50
  relationships round-tripped correctly into the graph (spreadsheet ->
  Neo4j), exceeding the master plan's "spot-check 15-20" target by
  checking every row.
- Historical accuracy of each fact was already checked against its cited
  Wikipedia source during Week 1 seeding — this week's validation is about
  ingestion fidelity (nothing lost/mangled going into Neo4j), which is a
  different, complementary check.

## Design decisions worth knowing about

- Entities get a second Neo4j label matching `entity_type`
  (`:Person`/`:Location`/`:Event`/`:Organization`/`:Cause`) so Week 4's
  hybrid retrieval can filter by type cheaply in Cypher.
- Free-text relationship objects (no matching entity) become `:Concept`
  nodes rather than being dropped — see `docs/schema.md`.
- `RELATION_VOCAB` and `ENTITY_TYPES` now live in `scripts/common.py` as
  the single source of truth; `tests/test_kg_vocab_consistency.py` checks
  the seed CSVs never drift from them.

## Open items / things to double check

- As the team keeps expanding `entities.csv`/`relationships.csv`, re-run
  `python scripts/validate_kg_csv.py && python kg/load_graph.py && python
  kg/validate_graph.py --all` — all three are idempotent and cheap.
- The 3 `:Concept` nodes are candidates for promotion to real entities if
  Week 4 retrieval needs to traverse through them (e.g. "rebel sepoys" ->
  which regiments specifically).
