# Knowledge graph seed schema

The schema is deliberately generic (no "1857-only" field names) so it can
extend to the wider freedom struggle later — extending scope should only
ever mean adding new `period_tag` *values*, never new columns.

## `data/kg_seed/entities.csv`

| column | meaning |
|---|---|
| `entity_id` | short stable id, e.g. `E001` |
| `name` | canonical display name |
| `entity_type` | one of: `Person`, `Location`, `Event`, `Organization`, `Cause` |
| `aliases` | semicolon-separated alternate names/spellings (used for fuzzy alias matching in Part 3, e.g. "Rani Lakshmibai" vs "Rani of Jhansi") |
| `period_tag` | which historical period this belongs to, e.g. `1857_revolt`. Future scope expansion adds new tag values here, not new columns. |
| `description` | one-sentence summary |

## `data/kg_seed/relationships.csv`

| column | meaning |
|---|---|
| `subject` | entity `name` (must resolve to a row in entities.csv, or an alias of one) |
| `relation` | controlled vocabulary, see below |
| `object` | entity `name`, or a short phrase when no single entity fits (e.g. "rebel sepoys") |
| `source` | URL the fact was drawn from — should match a `resolved_url` in `data/raw/sources_manifest.csv` |
| `period_tag` | same convention as entities.csv |
| `notes` | free-text context (dates, caveats) |

### Relation controlled vocabulary

`led`, `ruled`, `fought_against`, `allied_with`, `killed`, `executed`,
`captured`, `besieged`, `located_in`, `part_of`, `caused_by`, `triggered`,
`succeeded_by`, `appointed_by`, `commanded`, `declared_emperor_by`,
`died_during`, `member_of`, `disbanded_by`.

Adding a new relation type is fine — just add it to this list (and to
`RELATION_VOCAB` in `scripts/common.py`) so `kg/load_graph.py` and the
validation scripts stay in sync with what's actually used in the data.

## Neo4j graph shape (Part 3)

`kg/load_graph.py` loads this spreadsheet into Neo4j:

- Every `entities.csv` row becomes an `:Entity` node, plus a second label
  matching its `entity_type` (`:Person`, `:Location`, `:Event`,
  `:Organization`, `:Cause`) for cheap type-filtered Cypher queries.
- A relationship whose `object` resolves to a known entity becomes a normal
  `(:Entity)-[:RELATION]->(:Entity)` edge.
- A relationship whose `object` is a free-text phrase (no matching entity)
  becomes `(:Entity)-[:RELATION]->(:Concept {name: "..."})` instead of
  being dropped. A `Concept` can be promoted to a full `Entity` later by
  adding it to `entities.csv` and re-running the loader.

## Validation

`scripts/validate_kg_csv.py` builds a name+alias → `entity_id` index from
`entities.csv` and flags any `subject`/`object` in `relationships.csv` that
doesn't resolve, so typos in the hand-built spreadsheet get caught early
instead of silently producing orphan graph nodes in Part 3.
