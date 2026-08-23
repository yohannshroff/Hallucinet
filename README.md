# HalluciNet

RAG + GraphRAG system for reducing LLM hallucination on Indian historical
Q&A. v1 scope: the Revolt of 1857 (~May 1857 – 1859).

> **Hypothesis:** combining graph-based and vector-based retrieval reduces
> hallucination on Indian historical Q&A compared to vector retrieval alone,
> measured on a curated 1857 Revolt test set.

## Status

Weeks 1–3 of an 8-week build: knowledge-graph seed data, source document
collection, a working vector retrieval pipeline (chunking → embeddings →
FAISS), and that KG seed loaded into Neo4j with entity resolution. Weeks
4–8 (hybrid retrieval, LLM integration, trust scoring, evaluation, full
app) are stubbed out — see the `README.md` in each of `retrieval/`, `api/`,
`ui/`, `eval/`.

## Setup

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

This installs everything Week 1–3 code needs (`requests`, `pandas`,
`sentence-transformers`, `faiss-cpu`, `neo4j`, `fuzzywuzzy`, etc.). Neo4j
itself and Ollama are **not** installed by this script — see
[docs/manual_setup_neo4j.md](docs/manual_setup_neo4j.md) and
[docs/manual_setup_ollama.md](docs/manual_setup_ollama.md) (Ollama isn't
needed until Week 5). Copy `.env.example` to `.env` and fill in your Neo4j
credentials before running anything in `kg/`.

## Pipeline (Weeks 1–3)

```bash
# Week 1: fetch source documents and validate the hand-built KG seed
python scripts/fetch_wikipedia_sources.py
python scripts/validate_kg_csv.py

# Week 2: chunk -> embed -> index -> query
python scripts/clean_and_chunk.py
python scripts/build_embeddings.py
python scripts/build_faiss_index.py
python scripts/query_index.py "Who led the resistance at Jhansi?" --k 5

# Week 3: load the KG seed into Neo4j and validate it
python kg/load_graph.py
python kg/validate_graph.py --all
```

Run the test suite with:

```bash
pytest tests/
```

(The `kg/` integration tests skip automatically if Neo4j isn't running.)

## Repo layout

```
data/           source docs, KG seed CSVs, processed chunks/embeddings/index
scripts/        Week 1-2 pipeline scripts (see docs/week1_notes.md, week2_notes.md)
kg/             Week 3: Neo4j ingestion + entity resolution (see docs/week3_notes.md)
docs/           schema reference, manual setup guides, weekly notes
tests/          pytest suite for the Week 1-3 pipeline
retrieval/      [stub] Week 4: hybrid vector+graph retrieval
api/            [stub] Week 5: FastAPI backend
ui/             [stub] Week 6: Streamlit frontend
eval/           [stub] Week 7-8: hallucination evaluation harness
```

See [docs/schema.md](docs/schema.md) for the entity/relationship CSV schema.
