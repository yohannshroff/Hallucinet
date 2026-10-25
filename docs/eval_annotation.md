# Human validation of the automatic scorer

Sample: 60 answers from run 1, 15 per system (closed-book, vector, graph, hybrid), drawn at
random with a fixed seed and shuffled. Two annotators labelled each answer independently
(1 = states what the gold answer says, or correctly says the book does not say; 0 = wrong,
invented, or an unwarranted refusal), without seeing each other's labels, the system name or
the automatic verdict. Sheets: `data/eval/annotation/sheet_annotator_{1,2}.csv`; automatic
verdicts: `key_automatic_verdicts.csv`. Computed with `python eval/annotation.py kappa`.

| Measure | Value |
|---|---|
| Items | 60 |
| Annotator 1 vs annotator 2: Cohen's kappa | 0.862 |
| Raw agreement between annotators | 0.93 (56 of 60) |
| Automatic scorer vs human labels, on the 56 items both annotators agree on: kappa | 0.772 |
| Raw agreement, automatic vs human | 0.89 (50 of 56) |

Annotators disagreed on 4 items (Q30 twice, Q42 twice).

Automatic verdict against the annotators' agreed label, by system (matches / items): closed-book 13/14,
graph 14/15, hybrid 10/13, vector 13/14. Of the 6 mismatches, 4 were answers the humans
judged correct and the automatic scorer marked wrong (hybrid x3, vector x1: the entity-match
rule misses some correct answers, for example date and multi-part answers), and 2 were answers
the scorer marked correct and the humans judged wrong (one graph, one closed-book).

Consequence: on this sample the automatic scorer is conservative for the retrieval systems.
Humans labelled 77% of agreed hybrid answers and 71% of vector answers correct, against 54%
and 64% by the automatic rule on the same items (about 13 to 15 items per system, so these
shares are rough). Report the automatic numbers as a lower bound for the RAG systems, and
the agreement above as the check of the scorer.
