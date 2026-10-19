# Evaluation question set

`data/eval/qa_test_set.csv`: 54 questions about *The Great Mutiny* (the corpus
the system searches). It replaces the earlier 30-question set, which had been
written from the old knowledge graph and so favoured graph retrieval.

| Type (`hop_type`) | Count | What it tests |
|---|---|---|
| `single_hop` | 21 | the answer is stated in one passage |
| `temporal` | 13 | a date, or which of two events came first |
| `multi_hop` | 10 | the answer needs facts from two different passages |
| `unanswerable` | 10 | the book does not contain the answer (the system should say so) |

## Columns

| Column | Meaning |
|---|---|
| `id`, `question` | |
| `gold_answer` | short answer, paraphrased from the book (no quotations) |
| `expected_entities` | `;`-separated words or names; an answer containing any of them counts as matching (used by the automatic scorer) |
| `gold_pages` | the 4-page groups of the PDF that state the answer, `;`-separated, e.g. `353-356;373-376`. A multi-hop question lists a group per fact. Blank for unanswerable questions |
| `evidence_terms` | short phrases that must appear in the gold pages, used to machine-check `gold_pages` |
| `hop_type`, `category`, `answerable` | |
| `notes` | for unanswerable questions: why the book cannot answer (e.g. the term never occurs) |

`gold_pages` use the same page groups the system cites (`pp. 353-356` etc.),
so retrieval can be scored against them directly (Recall@k, MRR). They are
PDF page numbers, which differ from the printed page numbers.

## How it was written

- **Independently of the system.** Questions were written by reading the book
  text itself: the chapter summaries (which list the key events and dates)
  and passages sampled across all parts of the narrative. No retrieval
  result and no knowledge-graph content was used to choose a question, so the
  set does not favour any retrieval mode.
- **Every gold page is machine-checked.** `python eval/validate_test_set.py`
  confirms that each answerable question's `evidence_terms` occur in its
  `gold_pages`, and that the page groups exist in the corpus.
- **Unanswerable questions were checked for absence** against the full text:
  the key terms ("Congress", "1885", "Bassein", "Treaty of", "Zafar", a Delhi
  population figure, a railway used by Havelock, a Queen Victoria visit to
  Lucknow, ...) either never occur or occur only in unrelated contexts. Some
  are false-premise or plausible-sounding traps on purpose, since the risk
  being measured is the model inventing an answer.

## Status and limits

- **Verified.** Both team members signed off every gold answer against the book (October 2026). Alongside that, the evidence check proves the key facts are in the cited pages, and a second re-read of the cited pages for all 44 answerable questions found every gold answer supported (wording differences only, e.g. "Muslims" for the book's "Mohammedans"). The gold answers were drafted with AI assistance before this verification. The two-annotator agreement the report needs is a separate exercise, on model answers (see `eval/annotation.py`).
- 54 questions give wide confidence intervals; the report should say so and
  use paired tests rather than raw percentages.
- Date and "who" questions dominate. Causal and analytical questions ("why did
  the revolt spread?") are not included because they have no single gold page
  and need human grading.
