# ui/ (Part 8)

Streamlit frontend: question box, retrieval-mode selector, grounded
answer, trust score, and cited sources — calls the FastAPI backend in
`api/`, with a pre-cached fallback (`data/eval/demo_cache.json`) if the
live API is unreachable or too slow during a demo.

Run with (in a separate terminal from the API):

```bash
uvicorn api.main:app --port 8000
streamlit run ui/app.py
```

See [docs/part8_notes.md](../docs/part8_notes.md).
