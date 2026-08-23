# HalluciNet

RAG + GraphRAG system for reducing LLM hallucination on Indian historical
Q&A. v1 scope: the Revolt of 1857 (~May 1857 – 1859).

> **Hypothesis:** combining graph-based and vector-based retrieval reduces
> hallucination on Indian historical Q&A compared to vector retrieval alone,
> measured on a curated 1857 Revolt test set.

## Status

Weeks 1–5 of an 8-week build: knowledge-graph seed data, source document
collection, a working vector retrieval pipeline (chunking → embeddings →
FAISS), that KG seed loaded into Neo4j with entity resolution, hybrid
retrieval (query-time entity extraction + graph search + vector search,
merged into a single cited context), and a local LLM (Ollama) generating
grounded answers from that context. Weeks 6–8 (trust scoring, evaluation,
full app) are stubbed out — see the `README.md` in each of `ui/`, `eval/`.

## Setup

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python -m spacy download en_core_web_sm
```

This installs everything Week 1–5 code needs (`requests`, `pandas`,
`sentence-transformers`, `faiss-cpu`, `neo4j`, `fuzzywuzzy`, `spacy`, etc.).
Neo4j and Ollama are **not** installed by `setup_env.sh` — see
[docs/manual_setup_neo4j.md](docs/manual_setup_neo4j.md) and
[docs/manual_setup_ollama.md](docs/manual_setup_ollama.md). Copy
`.env.example` to `.env` and fill in your Neo4j/Ollama config before
running anything in `kg/`, `retrieval/`, or `generation/`.

## Pipeline (Weeks 1–5)

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

# Week 4: hybrid retrieval (entity extraction + graph search + vector search)
python retrieval/hybrid_retrieve.py "Who led the resistance at Jhansi?"

# Week 5: grounded answer generation via Ollama
python generation/generate_answer.py "Who led the resistance at Jhansi?"
python generation/run_sample_questions.py   # runs the 10-question sample set
```

Run the test suite with:

```bash
pytest tests/
```

(The `kg/`, `retrieval/`, and `generation/` integration tests skip
automatically if Neo4j, the FAISS index, or Ollama aren't available.)

## Repo layout

```
data/           source docs, KG seed CSVs, processed chunks/embeddings/index, eval questions
scripts/        Week 1-2 pipeline scripts (see docs/week1_notes.md, week2_notes.md)
kg/             Week 3: Neo4j ingestion + entity resolution (see docs/week3_notes.md)
retrieval/      Week 4: hybrid vector+graph retrieval (see docs/week4_notes.md)
generation/     Week 5: grounding prompt template + Ollama generation (see docs/week5_notes.md)
docs/           schema reference, manual setup guides, weekly notes
tests/          pytest suite for the Week 1-5 pipeline
ui/             [stub] Week 6: Streamlit frontend
eval/           [stub] Week 7: hallucination evaluation harness
api/            [stub] Week 8: FastAPI backend wiring
```

See [docs/schema.md](docs/schema.md) for the entity/relationship CSV schema.
