# api/ (Week 8)

FastAPI backend wrapping the hybrid retriever + grounded Ollama generation
(Weeks 4-5) and NLI trust scoring (Week 6) behind an HTTP endpoint.

- `main.py` — the FastAPI app. `POST /ask` takes `{question, k, mode}` and
  returns `{question, answer, trust_score, n_claims, n_entailed,
  seed_entities, sources}`. `GET /health` reports whether Ollama is
  reachable.
- `response.py` — the shared response-building logic (`build_ask_response`)
  used by both `main.py`'s endpoint and `build_demo_cache.py`, so the two
  never drift out of sync.
- `build_demo_cache.py` — pre-runs the 10 Week 5 sample questions and
  caches the results to `data/eval/demo_cache.json`, which `ui/app.py`
  falls back to if the live API/Ollama is unreachable or too slow during a
  demo (per the master plan's risk register).

Run with:

```bash
uvicorn api.main:app --reload --port 8000
```

See [docs/week8_notes.md](../docs/week8_notes.md).
