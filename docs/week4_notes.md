# Week 4 notes

Goals: extract entities from a user query, run graph + vector search, merge
into a single context.

- Entity extraction: `retrieval/entity_extraction.py`. Verified empirically
  that generic spaCy (`en_core_web_sm`) NER is unreliable on this niche
  domain — it mislabels "Barrackpore" as PERSON, misses "Mangal Pandey"
  entirely, and splits "Doctrine of Lapse" across two noun chunks. Since
  the KG only has ~50 entities, extraction combines two signals instead of
  trusting NER alone:
  1. a direct case-insensitive substring scan of every known name/alias
     against the query (cheap, catches exact/alias mentions with no NER
     mislabeling risk)
  2. spaCy noun chunks + named entities as candidate spans, resolved via
     `kg/entity_resolution.py` (exact then fuzzy, threshold raised to 90
     since query text is noisier than a curated spreadsheet row)
- Graph search: `retrieval/graph_search.py` — 1-hop Cypher expansion from
  each seed entity (both edge directions), capped per entity, de-duplicated
  across seeds.
- Vector search: `retrieval/vector_search.py` — thin wrapper around
  `scripts/query_index.py`'s existing FAISS search, reused rather than
  duplicated.
- Merge: `retrieval/hybrid_retrieve.py` runs graph and vector search
  independently (graph search degrades gracefully to empty if Neo4j is
  unreachable) and concatenates: graph facts first (precise, entity-tied),
  vector chunks second (broader supporting prose), each cited. No
  cross-source scoring/fusion — deliberately simple; Week 7's vector-only
  vs graph-only vs hybrid ablation is what tells us whether this is good
  enough.

## Verified manually

- `"Who led the resistance at Jhansi?"` → seed entity Jhansi; graph facts
  surfaced Rani Lakshmibai (`ruled`) and Hugh Rose (`besieged`) directly;
  vector chunks corroborated with prose from the Siege of Jhansi and
  Indian Rebellion of 1857 articles.
- `"Why did Mangal Pandey attack officers at Barrackpore?"` → both entities
  extracted correctly (despite spaCy's NER mislabeling), graph facts and
  passages both centered correctly on the Barrackpore incident.

## Open items

- The fuzzy fallback can occasionally resolve a loosely-related phrase to
  an entity (e.g. "Indian rulers" → Princely States) — not wrong exactly,
  but worth watching as the query set grows in Week 7's evaluation.
- 1-hop graph expansion only; if Week 7 evaluation shows multi-hop
  reasoning questions need more, revisit before Week 5 locks in the prompt
  template.
