# HalluciNet

RAG + GraphRAG system for reducing LLM hallucination on Indian historical
Q&A. v1 scope: the Revolt of 1857 (~May 1857 – 1859).

> **Hypothesis:** combining graph-based and vector-based retrieval reduces
> hallucination on Indian historical Q&A compared to vector retrieval alone,
> measured on a curated 1857 Revolt test set.

## Status

All 8 weeks of the build plan are complete: KG seed data + source
documents (1), vector retrieval (2), Neo4j graph (3), hybrid retrieval
(4), grounded generation via a local LLM (5), NLI-based trust scoring (6),
a vector/graph/hybrid ablation over 30 questions (7), and a FastAPI +
Streamlit app with a demo-question cache (8).

**Start here:** [docs/final_report.md](docs/final_report.md) summarizes
the whole project — architecture, results (with an honest read of their
caveats), and what to fix before presenting the numbers as final. Each
`docs/week*_notes.md` has the detailed weekly log.

## Setup

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python -m spacy download en_core_web_sm
```

This installs all Python dependencies (`requests`, `pandas`,
`sentence-transformers`, `faiss-cpu`, `neo4j`, `fuzzywuzzy`, `spacy`,
`matplotlib`, `fastapi`, `uvicorn`, `streamlit`). Neo4j and Ollama are
**not** installed by `setup_env.sh` — see
[docs/manual_setup_neo4j.md](docs/manual_setup_neo4j.md) and
[docs/manual_setup_ollama.md](docs/manual_setup_ollama.md). Copy
`.env.example` to `.env` and fill in your Neo4j/Ollama config.

## Pipeline

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

# Week 6: claim splitting + NLI entailment -> trust score
python eval/trust_score.py "Who led the resistance at Jhansi?"
python eval/run_trust_scores.py   # trust scores for the 10-question sample set

# Week 7: vector vs graph vs hybrid ablation over 30 questions
python eval/ablation.py
python eval/chart.py   # renders docs/week7_accuracy_by_mode.png

# Week 8: run the app
python api/build_demo_cache.py     # pre-cache the 10 sample questions as a demo fallback
uvicorn api.main:app --port 8000   # in one terminal
streamlit run ui/app.py            # in another
```

Run the test suite with:

```bash
pytest tests/
```

(Integration tests across `kg/`, `retrieval/`, `generation/`, `eval/`, and
`api/` skip automatically if Neo4j, the FAISS index, or Ollama aren't
available.)

## Repo layout

```
data/           source docs, KG seed CSVs, eval question sets, processed chunks/embeddings/index
scripts/        Week 1-2 pipeline scripts (see docs/week1_notes.md, week2_notes.md)
kg/             Week 3: Neo4j ingestion + entity resolution (see docs/week3_notes.md)
retrieval/      Week 4: hybrid vector+graph retrieval (see docs/week4_notes.md)
generation/     Week 5: grounding prompt template + Ollama generation (see docs/week5_notes.md)
eval/           Weeks 6-7: trust score + vector/graph/hybrid ablation (see docs/week6_notes.md, week7_notes.md)
api/            Week 8: FastAPI backend (see docs/week8_notes.md)
ui/             Week 8: Streamlit frontend
docs/           final report, schema reference, manual setup guides, weekly notes
tests/          pytest suite for the whole pipeline
```

See [docs/schema.md](docs/schema.md) for the entity/relationship CSV schema.
