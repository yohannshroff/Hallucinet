# ui/ (Part 8)

Streamlit frontend for the FastAPI backend in `api/`.

- **Ask** — chat-style questions; each answer shows its `[n]` citations
  linked to a numbered source list (knowledge-graph facts and book pages),
  the trust score, and a claim-by-claim check.
- **Compare modes** — the same question answered by hybrid, graph-only and
  passages-only retrieval side by side (three answers in a row, so it can
  take several minutes).
- **Knowledge base** — graph and corpus statistics and the entity /
  relationship tables.
- **How it works** — a plain-language description of the pipeline.

The sidebar holds the retrieval mode, top-k and backend status. If the
backend or the language model is unavailable, saved answers from
`data/eval/demo_cache.json` (built with `python api/build_demo_cache.py`)
are shown instead.

Styling lives in `styles.css` (base colours and fonts in
`.streamlit/config.toml`): no shadows or rounded corners except the
question box and the question bubbles.

Run with (in a separate terminal from the API):

```bash
uvicorn api.main:app --port 8000
streamlit run ui/app.py
```

See [docs/part8_notes.md](../docs/part8_notes.md).
