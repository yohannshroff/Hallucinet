# api/ (Part 8)

FastAPI backend wrapping the hybrid retriever + grounded Ollama generation
(Parts 4-5) and NLI trust scoring (Part 6) behind an HTTP API.

- `main.py` — the FastAPI app:
  - `POST /ask` takes `{question, k, mode}` (`mode`: `hybrid`, `graph` or
    `vector`) and returns the answer, `trust_score`, `n_claims` /
    `n_entailed`, the per-claim checks (`claims`), the graph facts and book
    passages used (in the same numbered order the answer cites), the
    recognised entities and timings.
  - `GET /health` — are Ollama, Neo4j and the search index available?
  - `GET /stats` — what's in the knowledge base (live Neo4j counts + corpus size).
  - `GET /kg` — the full entity and relationship tables.

  At startup it loads the helper models (spaCy, embeddings, NLI, reranker)
  and the FAISS index once, so the first question isn't slow; the language
  model itself is loaded by Ollama on the first question.
- `response.py` — the shared response-building logic (`build_ask_response`)
  used by both `main.py`'s endpoint and `build_demo_cache.py`, so the two
  never drift out of sync.
- `build_demo_cache.py` — pre-runs the sample questions and caches the
  results to `data/eval/demo_cache.json`, which `ui/app.py` falls back to
  if the live API/Ollama is unreachable or too slow during a demo.

Run with:

```bash
uvicorn api.main:app --port 8000
```

A question takes roughly 30 s to a few minutes on a laptop CPU (the first
one longer, while the model loads). Requests are answered one at a time; a
request the UI gave up on still runs to the end, so the next one waits.

See [docs/part8_notes.md](../docs/part8_notes.md).
