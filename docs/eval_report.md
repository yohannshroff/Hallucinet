# Evaluation report

54 questions, 1 run(s) per system. Cells are mean ± SD across runs, then a 95% bootstrap CI over questions in brackets (the CI reflects question sampling, not run-to-run variation).

## Overall (all questions)

| system | correct | hallucinated answer | refused |
|---|---|---|---|
| closed_book | 0.167 ± 0.000 [0.07, 0.28] | 0.759 ± 0.000 [0.65, 0.87] | 0.148 ± 0.000 [0.06, 0.24] |
| vector | 0.833 ± 0.000 [0.72, 0.93] | 0.130 ± 0.000 [0.06, 0.22] | 0.185 ± 0.000 [0.09, 0.30] |
| graph | 0.685 ± 0.000 [0.56, 0.80] | 0.167 ± 0.000 [0.07, 0.28] | 0.333 ± 0.000 [0.20, 0.46] |
| hybrid | 0.852 ± 0.000 [0.76, 0.94] | 0.130 ± 0.000 [0.06, 0.22] | 0.167 ± 0.000 [0.07, 0.28] |

## Answerable questions: quality and hallucination

| system | correct | token-F1 | claim-level hallucination rate | (NLI only) | trust score | citation accuracy |
|---|---|---|---|---|---|---|
| closed_book | 0.114 ± 0.000 [0.02, 0.20] | 0.245 ± 0.000 [0.18, 0.32] | 0.775 ± 0.000 [0.60, 0.95] | 0.925 ± 0.000 [0.80, 1.00] | n/a | n/a |
| vector | 0.841 ± 0.000 [0.73, 0.93] | 0.376 ± 0.000 [0.32, 0.43] | 0.306 ± 0.000 [0.19, 0.44] | 0.640 ± 0.000 [0.51, 0.77] | 0.602 ± 0.000 [0.46, 0.73] | 0.628 ± 0.000 [0.50, 0.75] |
| graph | 0.614 ± 0.000 [0.48, 0.75] | 0.298 ± 0.000 [0.24, 0.36] | 0.270 ± 0.000 [0.14, 0.41] | 0.642 ± 0.000 [0.50, 0.78] | 0.719 ± 0.000 [0.60, 0.83] | 0.511 ± 0.000 [0.38, 0.64] |
| hybrid | 0.864 ± 0.000 [0.75, 0.95] | 0.396 ± 0.000 [0.34, 0.45] | 0.236 ± 0.000 [0.12, 0.36] | 0.585 ± 0.000 [0.45, 0.71] | 0.685 ± 0.000 [0.57, 0.80] | 0.662 ± 0.000 [0.54, 0.77] |

## By question type (correct)

| system | single_hop | multi_hop | temporal | unanswerable |
|---|---|---|---|---|
| closed_book | 0.048 ± 0.000 [0.00, 0.14] | 0.200 ± 0.000 [0.00, 0.50] | 0.154 ± 0.000 [0.00, 0.38] | 0.400 ± 0.000 [0.10, 0.70] |
| vector | 0.810 ± 0.000 [0.62, 0.95] | 0.900 ± 0.000 [0.70, 1.00] | 0.846 ± 0.000 [0.62, 1.00] | 0.800 ± 0.000 [0.50, 1.00] |
| graph | 0.429 ± 0.000 [0.24, 0.62] | 0.900 ± 0.000 [0.70, 1.00] | 0.692 ± 0.000 [0.46, 0.92] | 1.000 ± 0.000 [1.00, 1.00] |
| hybrid | 0.810 ± 0.000 [0.62, 0.95] | 1.000 ± 0.000 [1.00, 1.00] | 0.846 ± 0.000 [0.62, 1.00] | 0.800 ± 0.000 [0.50, 1.00] |

## Paired comparison (exact McNemar on per-question majority correctness; Bonferroni α = 0.01 for these 5 tests)

| A vs B | A only right | B only right | p |
|---|---|---|---|
| hybrid vs vector | 1 | 0 | 1.000 |
| vector vs closed_book | 38 | 2 | 0.000 |
| hybrid vs closed_book | 39 | 2 | 0.000 |
| graph vs vector | 3 | 11 | 0.057 |
| hybrid vs graph | 11 | 2 | 0.022 |

## Retrieval (answerable questions)

| mode | k | hit@k | recall@k | MRR | context precision | items |
|---|---|---|---|---|---|---|
| graph | 1 | 0.659 | 0.557 | 0.318 | 0.286 | 4.0 |
| graph | 3 | 0.727 | 0.670 | 0.331 | 0.210 | 5.8 |
| graph | 5 | 0.727 | 0.670 | 0.331 | 0.174 | 7.5 |
| graph | 10 | 0.795 | 0.750 | 0.340 | 0.138 | 11.5 |
| hybrid | 1 | 0.864 | 0.761 | 0.511 | 0.479 | 4.1 |
| hybrid | 3 | 0.909 | 0.841 | 0.517 | 0.309 | 6.1 |
| hybrid | 5 | 0.909 | 0.864 | 0.517 | 0.255 | 8.1 |
| hybrid | 10 | 0.955 | 0.909 | 0.520 | 0.180 | 13.1 |
| vector | 1 | 0.795 | 0.705 | 0.795 | 0.795 | 1.0 |
| vector | 3 | 0.864 | 0.761 | 0.826 | 0.386 | 3.0 |
| vector | 5 | 0.864 | 0.773 | 0.826 | 0.277 | 5.0 |
| vector | 10 | 0.909 | 0.864 | 0.832 | 0.185 | 10.0 |
