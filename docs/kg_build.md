# How the knowledge graph is built from the book

The graph is built once from the same text the vector index searches
(`data/processed/chunks.jsonl`, i.e. Hibbert's *The Great Mutiny*). It
replaces the earlier hand-curated, Wikipedia-sourced graph.

## Result

| | |
|---|---|
| Entities | 278 (123 people, 91 places, 48 events, 16 groups) |
| Typed relationships | 57: `located_in` 47, `ruled` 7, `member_of` 2, `fought_against` 1 |
| `associated_with` edges | 357 (entities mentioned together in 2+ sentences) |
| Citations | every relationship names the book pages it came from |

## Pipeline

Run in order; each stage caches its output so nothing slow is repeated.

| Stage | Script | Uses | Time | Output (`data/kg_build/`) |
|---|---|---|---|---|
| 1. Find names | `kg/find_candidates.py` | spaCy | ~40 s | `mentions.json`, `candidates.jsonl` (gitignored: quote book sentences) |
| 2. Type and clean names | `kg/type_entities.py` | local LLM (`gemma3:4b`) | ~20 min | `entity_types.json` |
| 3. Extract facts | `kg/extract_facts.py` | spaCy parse + NLI model | ~45 s | `extracted.jsonl`, `cooccurrence.jsonl` |
| 4. Consolidate | `kg/consolidate_triples.py` | none | seconds | `data/kg_seed/entities.csv`, `relationships.csv` |
| 5. Load | `kg/load_graph.py --reset` | Neo4j | seconds | the graph database |

Stages 2's slow result (`entity_types.json`) and stage 3's outputs are
committed (names, relations and page numbers only, no book text), so a fresh
clone can rebuild the CSVs and the graph from stage 4 without the PDF or a
language model. Re-tuning thresholds (`--min_cooc`, `--min_entail`, ...) only
means re-running stage 4.

What each stage does:

1. **Find names.** spaCy NER plus a rule for event names ("Siege of Delhi").
   Names mentioned 3+ times are candidates. Event names imply their location
   ("Siege of Delhi" `located_in` Delhi) and "*Person* of the *Regiment*"
   gives `member_of`.
2. **Type and clean names.** spaCy's small model mislabels many names in this
   book (it calls Meerut and Jhansi people), so the local LLM types each name
   from one example sentence (person / place / group / event / other) with its
   reply constrained to a JSON schema. "Other" removes junk such as footnote
   debris and book titles. Of 522 names: 289 people, 114 places, 40 groups, 19
   events, 60 dropped.
3. **Extract facts.** Grammar rules over the dependency parse find
   "*subject* *verb* *object*" for verbs that map to a controlled relation,
   including passives, plus "*Person*, the *title* of *place*". Each fact is
   scored by the project's NLI model against its source sentence; only facts
   with entailment >= 0.5 are kept. The same pass counts which entity pairs
   are mentioned in the same sentence.
4. **Consolidate.** Merges name variants (surname -> the one matching full
   name when that name is common enough; spelling variants of people and
   places), drops facts whose entity types make the relation impossible (the
   table in the script), and keeps `associated_with` edges for pairs seen in 2+
   sentences, at most the 6 strongest per entity.

## What did not work (and why the graph looks the way it does)

The goal was a graph of typed relations (`led`, `besieged`, ...). Four ways of
extracting them automatically on a laptop were tried; each was checked by eye
on small samples (informal, not a formal evaluation):

| Attempt | What happened |
|---|---|
| The 4B LLM, free-form triples from 250-word windows | ~22 s per window; mostly invented relations ("hit", "threw") and trivia |
| The 4B LLM, one sentence at a time, reply constrained to the allowed entities and relations | Better format, still wrong: of 16 sentences, 13 facts, roughly 2 correct; with vague relations removed, 0 of 5 correct (e.g. "Mangal Pande executed Barrackpore") |
| Grammar rules alone | Correct when they fire, but Hibbert's long, pronoun-heavy sentences rarely have a clean "named subject verb named object": 12 candidate facts in the whole book |
| NLI model as a zero-shot relation classifier | Responds to co-occurring words rather than the relation (scored "Havelock appointed_by Tatya Tope" 0.98); only a minority of the high-scoring picks sampled were correct |

So typed relations are limited to what rules extract reliably, and the bulk of
the graph is `associated_with`: "these two entities are mentioned together in
N sentences (pp. X-Y)". That statement is true by construction, but it is
weaker evidence than a typed fact. Consequences for the evaluation:

- Graph-only retrieval on this graph is expected to be weak for "who did X to
  Y" questions; the vector side supplies the content.
- A larger model, or a human-verified set of typed facts layered on top
  (checked against the book's pages), would improve it. That is the main
  future-work item for the graph.

## Re-running

```bash
python kg/find_candidates.py        # ~40 s
python kg/type_entities.py          # ~20 min, needs Ollama; resumable
python kg/extract_facts.py          # ~45 s
python kg/consolidate_triples.py --min_cooc 2
python kg/load_graph.py --reset     # wipes and reloads Neo4j
```
