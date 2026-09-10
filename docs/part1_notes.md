# Part 1 notes

Goals: finalize entity/relationship list, collect source documents, stand up
the repo + Python environment.

- Entities/relationships: `data/kg_seed/entities.csv` and
  `data/kg_seed/relationships.csv`, schema documented in
  [schema.md](schema.md). Seeded with ~40-60 entities and ~50+ relationships
  covering the people/locations/events/organizations/causes named in the
  master plan.
- Sources: `data/raw/source_list.csv` lists the Wikipedia articles to fetch;
  `scripts/fetch_wikipedia_sources.py` pulls plain-text extracts into
  `data/raw/text/` and records provenance in `sources_manifest.csv`.
  **Update:** the corpus later switched from Wikipedia articles to PDF
  source books — see the note at the bottom of this file.
- Validate the KG spreadsheet with `scripts/validate_kg_csv.py` before
  moving on — it catches typo'd entity names in the relationships table.
- Environment: `scripts/setup_env.sh` sets up the venv Python deps. Neo4j
  and Ollama are documented separately (`manual_setup_neo4j.md`,
  `manual_setup_ollama.md`) and aren't needed until Part 3 / Part 5.

## Open items / things to double check by hand

- Spot-check the seeded relationships against 2+ real sources per fact
  before trusting them for the graph build in Part 3 (per the risk
  register in the master plan).
- Expand entities/relationships beyond the initial seed rows as the team
  reads more source material — the CSVs are meant to keep growing through
  Part 1.

## Update: switched to PDF source books

The Wikipedia API fetch step (`scripts/fetch_wikipedia_sources.py`) was
replaced with `scripts/ingest_pdf_sources.py`, which extracts text from any
PDF dropped in `data/raw/` (e.g. a scanned history book) instead of pulling
Wikipedia article extracts. It writes the same `sources_manifest.csv`
schema (`doc_id, title, resolved_url, fetched_at, word_count`), so Part 2
onward (`clean_and_chunk.py`, embeddings, FAISS) needed no changes. Each
PDF is split into small page-group documents (default 4 pages) rather than
one document per article; citations now point to a local
`<pdf path>#page=N` instead of a Wikipedia URL. The KG seed CSVs
(`entities.csv` / `relationships.csv`) are unaffected — they're
hand-curated, not derived from this text corpus.
