# Week 8 notes

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
- `api/build_demo_cache.py`: pre-ran the 10 Week 5 sample questions
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
4. Full `pytest tests/` suite green (see final count in the Week 8 commit)

## What this project deliberately did not build

Per the master plan's "what NOT to build" list: no bias analysis, no
real-time web crawling, no multi-agent orchestration, no cloud deployment,
no enterprise GraphRAG framework, no large-scale infra, no advanced
visualization beyond the one comparison chart. The FastAPI/Streamlit
wiring here is intentionally minimal -- a working local demo, not a
production service.

## Open items for the actual report/viva prep

- Run the 2-rater manual agreement check (flagged in `docs/week7_notes.md`)
  on a sample of the ablation results before quoting the Week 7 numbers as
  final.
- Consider fixing the T10 Concept-node graph gap (`docs/week7_notes.md`)
  if there's time before presenting.
- Rehearse with the demo cache as the fallback path in mind -- if live
  Ollama inference stalls during the viva, switch talking points to "the
  system falls back to pre-verified cached answers" rather than waiting.
