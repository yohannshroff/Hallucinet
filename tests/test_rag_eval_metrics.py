"""Scoring functions and statistics used by the Track B evaluation."""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("scripts", "eval"):
    sys.path.insert(0, str(ROOT / sub))

import rag_metrics as M  # noqa: E402
import rag_stats as S  # noqa: E402

BUNDLE = {
    "graph_facts": [{"source": "data/raw/x.pdf#page=59"}],
    "vector_chunks": [{"title": "Book (pp. 169-172)"}, {"title": "Book (pp. 57-60)"}],
}


def test_page_groups():
    assert M.group_start(59) == 57
    assert M.group_start(60) == 57
    assert M.group_start(61) == 61
    assert M.parse_gold_pages("169-172;217-220") == [169, 217]
    assert M.parse_gold_pages("") == []


def test_retrieval_metrics():
    groups = M.ranked_groups(BUNDLE)  # graph fact first, then chunks
    assert groups == [57, 169, 57]
    assert M.reciprocal_rank(groups, [169]) == 0.5
    assert M.recall_at_k(groups, [169, 217]) == 0.5
    assert M.hit_at_k(M.ranked_groups(BUNDLE, k=1), [169]) == 1.0
    assert abs(M.context_precision(groups, [57]) - 2 / 3) < 1e-9
    assert math.isnan(M.recall_at_k(groups, []))


def test_refusals_are_correct_only_for_unanswerable():
    refusal = "The provided context does not contain enough information to answer this question."
    unans = {"answerable": "no", "expected_entities": ""}
    ans = {"answerable": "yes", "expected_entities": "tallow;beef;pig"}
    assert M.judge(refusal, unans)["correct"] is True
    assert M.judge("It was 3,000,000.", unans)["hallucinated_answer"] is True
    assert M.judge(refusal, ans)["correct"] is False
    assert M.judge("Tallow, from beef or pig fat.", ans)["correct"] is True
    assert M.judge("Butter.", ans)["hallucinated_answer"] is True


def test_citation_stats():
    s = M.citation_stats("Yes (pp. 169-172) and (pp. 400-404).", BUNDLE, [169])
    assert s["n_cited"] == 2
    assert s["citation_valid"] == 0.5
    assert s["citation_accuracy"] == 0.5


def test_mcnemar_and_kappa():
    r = S.mcnemar_exact([1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1])
    assert (r["a_only"], r["b_only"]) == (6, 1)
    assert abs(r["p_value"] - 0.125) < 1e-9
    assert S.cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert abs(S.cohen_kappa(list("aabb"), list("abab"))) < 1e-9


def test_bootstrap_ci_contains_mean():
    lo, hi = S.bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 0], n_boot=2000)
    assert lo <= 0.625 <= hi
    m, sd = S.mean_sd([0.5, 0.6, 0.7])
    assert abs(m - 0.6) < 1e-9 and abs(sd - 0.1) < 1e-9
