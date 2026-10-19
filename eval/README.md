# eval/ (Parts 6-7)

Hallucination evaluation harness.

**Part 6 — trust scoring:**
- `claim_splitting.py` — splits an answer into sentence-level claims,
  stripping citation brackets.
- `entailment.py` — NLI entailment scoring (`cross-encoder/nli-MiniLM2-L6-H768`)
  of a claim against a pool of evidence texts.
- `trust_score.py` — ties the two together: % of an answer's claims
  entailed by the retrieved evidence. CLI: `python eval/trust_score.py
  "<question>"`.
- `run_trust_scores.py` — runs trust scoring over the 10 sample questions,
  saves a transcript to `docs/part6_trust_scores.md`.

**Part 7 — ablation:**
- `ablation.py` — runs every question in `data/eval/qa_test_set.csv`
  (30 hand-written Q&A pairs) through vector-only / graph-only / hybrid
  retrieval, generates an answer for each, and classifies it
  correct/partial/wrong/hallucinated/refused using an automated,
  non-LLM proxy (entity match + trust score — see the module docstring for
  the full rubric and its rationale). CLI: `python eval/ablation.py`.
- `chart.py` — renders the accuracy-by-mode bar chart from `ablation.py`'s
  results CSV.

See [docs/part6_notes.md](../docs/part6_notes.md) and
[docs/part7_notes.md](../docs/part7_notes.md) for what was run, verified,
and found.

**Track B evaluation (final review):**
- `rag_eval.py` — `retrieval` (Recall@k / hit@k / MRR / context precision
  against `gold_pages`, no language model needed), `generate` (closed-book,
  vector, graph, hybrid; several seeded runs; resumable), `report` (mean ± SD
  across runs, bootstrap 95% CIs, exact McNemar, breakdown by question type).
- `rag_metrics.py` — the scoring rules (unanswerable questions are correct only
  if the model refuses; claim-level hallucination is measured against the gold
  pages). `rag_stats.py` — bootstrap, McNemar, Cohen's kappa.
- `validate_test_set.py` — checks the question set against the book text.
