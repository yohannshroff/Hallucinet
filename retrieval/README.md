# retrieval/ (Part 4)

Finds the evidence for a question: knowledge-graph facts from Neo4j and
passages from the book.

- `entity_extraction.py` — finds the entities a question mentions: alias
  scan over `data/kg_seed/entities.csv`, plus spaCy spans resolved with
  exact and fuzzy matching.
- `graph_search.py` — 1-hop facts around those entities, plus the shortest
  paths between them (path facts); facts are ranked by similarity to the
  question (`rank_facts`).
- `vector_search.py` — FAISS search over the 60-word chunks
  (`all-MiniLM-L6-v2` embeddings).
- `rerank.py` — cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
  that reads the question and each passage together. `HALLUCINET_RERANK=0`
  turns it off.
- `hybrid_retrieve.py` — `retrieve(question, entities_df, k, mode)` for
  `vector`, `graph` or `hybrid`:
  1. graph facts, capped at 6 and ranked by the question; plain co-mention
     (`associated_with`) facts are only shown when they lie on a path
     between two question entities;
  2. a pool of 20 candidate passages per source (vector search and/or
     graph-guided passages that name a question entity and its neighbour),
     limited to the book's narrative pages (`common.NARRATIVE_PAGES`, so the
     index and chronology at the back are skipped);
  3. the reranker picks the best k, skipping passages that overlap one
     already chosen, and each is widened by one neighbouring chunk on each side.

  `format_context_only` renders the result as one numbered source list
  (`[1] Knowledge graph, p. 393: ...`, `[3] The Great Mutiny, pp. 401-404: ...`)
  that the language model cites by number.

```bash
python retrieval/hybrid_retrieve.py "Who led the resistance at Jhansi?" --mode hybrid
python retrieval/rerank.py "Who led the resistance at Jhansi?"
```

See [docs/part4_notes.md](../docs/part4_notes.md) for the original design
and [docs/final_report.md](../docs/final_report.md) for current results.
