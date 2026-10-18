# Part 8 notes

Goals: wire FastAPI + Streamlit end-to-end, cache demo questions as a
fallback, finalize the report.

- `api/main.py`: FastAPI app, `POST /ask` (`{question, k, mode}` ->
  `{question, answer, trust_score, n_claims, n_entailed, seed_entities,
  sources}`), `GET /health`. Verified live: `uvicorn api.main:app --port
  8000`, `curl -X POST localhost:8000/ask -d '{"question": "Who led the
  resistance at Jhansi?"}'` returned a correct, cited, 100%-trust answer.
- `api/response.py`: response-shaping logic factored out so `main.py`'s
  endpoint and `build_demo_cache.py` share it rather than duplicating it.
- `ui/app.py`: Streamlit frontend -- question box, mode selector,
  answer/trust-score/sources display, falls back to
  `data/eval/demo_cache.json` if the API call fails. Verified live in a
  browser: typed "Who ruled Jhansi before the revolt?", got "Rani
  Lakshmibai ruled Jhansi" at 100% trust with two cited sources rendered
  correctly end-to-end through the actual UI (not just the API).
- `api/build_demo_cache.py`: pre-ran the 10 Part 5 sample questions
  through the full pipeline, cached to `data/eval/demo_cache.json` --
  the master plan's risk-register mitigation for "local LLM too slow for
  live demo."

## Verification performed

1. `curl localhost:8000/health` -> `{"ollama_reachable": true}`
2. `curl -X POST localhost:8000/ask ...` -> correct, cited, grounded
   answer with a real trust score
3. Streamlit UI driven through an actual browser (not just curl): typed a
   question, clicked Ask, confirmed the answer/trust score/sources render
   correctly
4. Full `pytest tests/` suite green (see final count in the Part 8 commit)

## What this project deliberately did not build

Per the master plan's "what NOT to build" list: no bias analysis, no
real-time web crawling, no multi-agent orchestration, no cloud deployment,
no enterprise GraphRAG framework, no large-scale infra, no advanced
visualization beyond the one comparison chart. The FastAPI/Streamlit
wiring here is intentionally minimal -- a working local demo, not a
production service.

## Open items for the actual report/viva prep

- Run the 2-rater manual agreement check (flagged in `docs/part7_notes.md`)
  on a sample of the ablation results before quoting the Part 7 numbers as
  final.
- Consider fixing the T10 Concept-node graph gap (`docs/part7_notes.md`)
  if there's time before presenting.
- Rehearse with the demo cache as the fallback path in mind -- if live
  Ollama inference stalls during the viva, switch talking points to "the
  system falls back to pre-verified cached answers" rather than waiting.

## Update: richer app + laptop efficiency pass

- **API:** `/ask` now also returns per-claim verdicts, graph facts, book
  passages (with snippets and match scores) and timings; new `/stats`
  (live Neo4j counts + corpus size) and `/kg` (entity/relationship tables)
  endpoints; `/health` also reports Neo4j and index status. Models and the
  FAISS index load once at startup.
- **UI:** tabs for Ask (example chips, trust banner, sources, claim-by-claim
  check), Compare search modes (same question, three retrieval modes side by
  side), Knowledge base (counts, charts, searchable tables) and How it works.
- **Efficiency:** shared Neo4j connection, cached FAISS/alias indexes,
  batched claim checking, CPU-pinned helper models with 2 threads, Ollama
  `keep_alive`/length caps, and Neo4j heap/page-cache caps -- see the
  "Running it on a small laptop" section of the README.
- **Citations:** book passages are cited by title and page range (the repo
  stores a relative `data/raw/<file>.pdf#page=N`, not a machine-specific
  absolute path), and the prompt no longer asks the model to cite file paths.
- `data/eval/demo_cache.json` is still from the old Wikipedia-era corpus;
  rebuild it with `python api/build_demo_cache.py` (needs Ollama running).
