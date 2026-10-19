"""Statistics for comparing systems on the same question set: bootstrap
confidence intervals, an exact McNemar test, and Cohen's kappa.

With ~54 questions the intervals are wide; report them rather than hide them.
"""

import numpy as np
from scipy.stats import binomtest


def mean_sd(values) -> tuple:
    """Mean and sample SD (ddof=1) of one number per run."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def bootstrap_ci(per_question, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple:
    """Percentile bootstrap CI of the mean, resampling questions (not runs):
    the questions are the sample we want to generalise from."""
    v = np.asarray(per_question, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def mcnemar_exact(a_correct, b_correct) -> dict:
    """Exact (binomial) McNemar test on paired per-question correctness.

    b = A right, B wrong; c = A wrong, B right. Two-sided p-value.
    """
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue)
    return {"a_only": only_a, "b_only": only_b, "p_value": p}


def cohen_kappa(rater1, rater2) -> float:
    """Cohen's kappa for two raters' labels over the same items."""
    r1, r2 = list(rater1), list(rater2)
    if len(r1) != len(r2) or not r1:
        raise ValueError("raters must label the same, non-empty set of items")
    n = len(r1)
    po = sum(x == y for x, y in zip(r1, r2)) / n
    labels = set(r1) | set(r2)
    pe = sum((r1.count(l) / n) * (r2.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
