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
- `run_trust_scores.py` — runs trust scoring over the sample questions,
  saves a transcript to `docs/part6_trust_scores.md`.

Claims are cleaned of citations, page references, file paths and markdown
before checking, and evidence is checked one sentence and one pair of
consecutive sentences at a time (so a claim whose subject is a pronoun in the
book can still be matched).

**Evaluation (current; results in [docs/eval_report.md](../docs/eval_report.md)
and [docs/final_report.md](../docs/final_report.md)):**
- `rag_eval.py`
  - `retrieval`: hit@k / Recall@k / MRR / context precision against
    `gold_pages`, no language model needed.
  - `generate`: closed-book, vector, graph and hybrid answers to the 54
    questions in `data/eval/qa_test_set.csv`; seeded runs; resumable.
  - `citations`: re-scores citation accuracy in saved runs (no language model).
  - `report`: mean ± SD across runs, bootstrap 95% CIs, exact McNemar,
    breakdown by question type -> `docs/eval_report.md`.
- `rag_metrics.py` — the scoring rules: unanswerable questions are correct only
  if the model refuses; claim-level hallucination is measured against the gold
  pages; citation accuracy maps each `[n]` the answer cites to its source and
  checks that source's page against the gold pages. `rag_stats.py` — bootstrap,
  McNemar, Cohen's kappa.
- `annotation.py` — the two-annotator check of the automatic scorer
  ([docs/eval_annotation.md](../docs/eval_annotation.md)).
- `chunk_sensitivity.py` — vector retrieval at different chunk sizes
  ([docs/eval_chunk_sensitivity.md](../docs/eval_chunk_sensitivity.md)).
- `validate_test_set.py` — checks the question set against the book text.

**Earlier ablation (superseded):** `ablation.py` and `chart.py` ran the old
30-question set over the Wikipedia corpus; see
[docs/part7_notes.md](../docs/part7_notes.md). They are kept for the record.

See [docs/part6_notes.md](../docs/part6_notes.md) for how the trust score was
built and tested.
