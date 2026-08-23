# Part 2 notes

Goals: clean + chunk the fetched documents, embed the chunks, build a FAISS
index, and confirm vector search works end-to-end.

- Chunking: `scripts/clean_and_chunk.py`, 180-word windows with 40-word
  overlap (stride 140), chosen to stay under MiniLM's ~256-subtoken limit
  after word→subtoken expansion. Output: `data/processed/chunks.jsonl`.
- Embeddings: `scripts/build_embeddings.py`, `all-MiniLM-L6-v2` via
  `sentence-transformers`, normalized vectors. Output:
  `data/processed/embeddings/chunk_embeddings.npy` (shape `[N, 384]`).
- Index: `scripts/build_faiss_index.py`, `faiss.IndexFlatIP` (cosine sim via
  inner product on normalized vectors — fine at this corpus size). Output:
  `data/processed/faiss_index/index.faiss` + `id_map.json`.
- Query: `scripts/query_index.py "<question>" --k 5` for a manual smoke
  test.

## Pipeline order

```
fetch_wikipedia_sources.py -> clean_and_chunk.py -> build_embeddings.py -> build_faiss_index.py -> query_index.py
```

## Open items

- If retrieval quality looks weak on the smoke-test query, revisit chunk
  size/overlap before Part 4 (hybrid retrieval) rather than after.
