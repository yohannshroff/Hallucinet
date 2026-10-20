# HalluciNet

RAG + GraphRAG system for reducing LLM hallucination on Indian historical
Q&A. The Revolt of 1857 is the first case study used to build and test the
pipeline; the dataset and knowledge graph are meant to grow to cover more
events and periods (every entity already carries a `period_tag`).

> **Hypothesis:** combining graph-based and vector-based retrieval reduces
> hallucination on Indian historical Q&A compared to vector retrieval alone,
> measured on a curated test set (currently drawn from the 1857 case study).

## Status

All 8 parts of the build plan are complete: KG seed data + source
documents (1), vector retrieval (2), Neo4j graph (3), hybrid retrieval
(4), grounded generation via a local LLM (5), NLI-based trust scoring (6),
a vector/graph/hybrid ablation over 30 questions (7), and a FastAPI +
Streamlit app with a demo-question cache (8).

**Start here:** [docs/final_report.md](docs/final_report.md) summarizes
the whole project — architecture, results (with an honest read of their
caveats), and what to fix before presenting the numbers as final. Each
`docs/part*_notes.md` has the detailed per-part log.

## Source

All answers come from one book: Christopher Hibbert, *The Great Mutiny: India 1857* (New York: The Viking Press, 1978). The book is copyrighted and is not included in this repository; page citations in the app and in the evaluation refer to the PDF's own page numbering. To rebuild everything, supply your own copy as a PDF in `data/raw/` (see the pipeline below).

## Setup

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python -m spacy download en_core_web_sm
```

This installs all Python dependencies (`pypdf`, `pandas`, `requests`,
`sentence-transformers`, `faiss-cpu`, `neo4j`, `fuzzywuzzy`, `spacy`,
`matplotlib`, `fastapi`, `uvicorn`, `streamlit`). Neo4j and Ollama are
**not** installed by `setup_env.sh` — see
[docs/manual_setup_neo4j.md](docs/manual_setup_neo4j.md) and
[docs/manual_setup_ollama.md](docs/manual_setup_ollama.md). Copy
`.env.example` to `.env` and fill in your Neo4j/Ollama config.

## Pipeline

```bash
# Part 1: extract text from PDF source books (drop PDFs in data/raw/)
python scripts/ingest_pdf_sources.py

# Part 2: chunk -> embed -> index -> query
python scripts/clean_and_chunk.py
python scripts/build_embeddings.py
python scripts/build_faiss_index.py
python scripts/query_index.py "Who led the resistance at Jhansi?" --k 5

# Part 3: build the knowledge graph from the book (details: docs/kg_build.md), load and validate it
python kg/find_candidates.py        # names + rule-based facts (~40 s)
python kg/type_entities.py          # LLM types/cleans the names (~20 min, needs Ollama; cached)
python kg/extract_facts.py          # grammar facts + NLI check + co-mentions (~45 s)
python kg/consolidate_triples.py --min_cooc 2   # -> data/kg_seed/*.csv (seconds)
python kg/load_graph.py --reset
python kg/validate_graph.py --all

# Part 4: hybrid retrieval (entity extraction + graph search + vector search)
python retrieval/hybrid_retrieve.py "Who led the resistance at Jhansi?"

# Part 5: grounded answer generation via Ollama
python generation/generate_answer.py "Who led the resistance at Jhansi?"
python generation/run_sample_questions.py   # runs the 10-question sample set

# Part 6: claim splitting + NLI entailment -> trust score
python eval/trust_score.py "Who led the resistance at Jhansi?"
python eval/run_trust_scores.py   # trust scores for the 10-question sample set

# Part 7: vector vs graph vs hybrid ablation over the question set (docs/test_set.md)
python eval/validate_test_set.py   # checks every gold page against the book text
python eval/ablation.py
python eval/rag_eval.py retrieval        # Recall@k / MRR against the gold pages (no LLM)
python eval/chunk_sensitivity.py          # chunk-size sensitivity of vector retrieval (no LLM)
python eval/rag_eval.py generate --runs 3 # closed-book / vector / graph / hybrid answers (needs Ollama, ~3 h)
python eval/rag_eval.py report            # tables with mean +- SD, CIs, McNemar -> docs/eval_report.md

# Part 8: run the app
python api/build_demo_cache.py     # pre-cache the 10 sample questions as a demo fallback
uvicorn api.main:app --port 8000   # in one terminal (endpoints: /ask /health /stats /kg)
streamlit run ui/app.py            # in another: Ask, Compare search modes, Knowledge base
```

Run the test suite with:

```bash
pytest tests/
```

(Integration tests across `kg/`, `retrieval/`, `generation/`, `eval/`, and
`api/` skip automatically if Neo4j, the FAISS index, or Ollama aren't
available.)

## Running it on a small laptop (8 GB Mac)

Memory, not CPU, is the bottleneck: the language model alone is ~3 GB. What
the project does about it:

- **Models are loaded once and reused.** The API loads the search index,
  embedding model, claim-checking model and spaCy at startup (about 20 s,
  once) instead of per question; the Neo4j connection and the FAISS index
  are shared across requests.
- **Small helper models run on CPU with 2 threads** (`HALLUCINET_DEVICE`,
  `HALLUCINET_THREADS`). Measured here: the Apple GPU wasn't faster for
  them and would compete with Ollama for GPU memory.
- **All claims are checked in one batch** rather than one model call per
  claim — same scores, much less overhead.
- **The language model unloads itself after 5 idle minutes**
  (`OLLAMA_KEEP_ALIVE`) and replies are capped at 400 tokens
  (`OLLAMA_NUM_PREDICT`).
- **Cap Neo4j's memory.** Its defaults claim a big share of RAM; for this
  47-node graph add these to `$(brew --prefix neo4j)/libexec/conf/neo4j.conf`
  and restart it:

  ```
  server.memory.heap.initial_size=192m
  server.memory.heap.max_size=384m
  server.memory.pagecache.size=64m
  ```

Typical timings (warm): ~15 s per answer (language model) plus ~2 s for the
trust check. The first question after the model has unloaded takes longer.

## Repo layout

```
data/           source docs, KG seed CSVs, eval question sets, processed chunks/embeddings/index
scripts/        Part 1-2 pipeline scripts (see docs/part1_notes.md, part2_notes.md)
kg/             Part 3: Neo4j ingestion + entity resolution (see docs/part3_notes.md)
retrieval/      Part 4: hybrid vector+graph retrieval (see docs/part4_notes.md)
generation/     Part 5: grounding prompt template + Ollama generation (see docs/part5_notes.md)
eval/           Parts 6-7: trust score + vector/graph/hybrid ablation (see docs/part6_notes.md, part7_notes.md)
api/            Part 8: FastAPI backend (see docs/part8_notes.md)
ui/             Part 8: Streamlit frontend
docs/           final report, schema reference, manual setup guides, per-part notes
tests/          pytest suite for the whole pipeline
```

See [docs/schema.md](docs/schema.md) for the entity/relationship CSV schema.
