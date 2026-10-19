"""Pure scoring functions for the Track B evaluation (no models, no I/O).

Retrieval is scored against `gold_pages` in data/eval/qa_test_set.csv. The
corpus is stored as 4-page groups, so a "page" here is a group, named by its
first page (the group "57-60" is page 57).

Generation is scored per question, aware of whether it is answerable:
- answerable: correct = not a refusal AND at least half of the expected
  entities appear in the answer (a deterministic proxy; validate it against
  the two-annotator labels before quoting it).
- unanswerable: correct = the answer is a refusal. Anything else is a
  hallucination by definition (the book has no answer to find).
"""

import math
import re

PAGES_PER_GROUP = 4

CHUNK_PAGES_RE = re.compile(r"pp\.\s*(\d+)\s*-\s*(\d+)")
PAGE_ANCHOR_RE = re.compile(r"#page=(\d+)")
REFUSAL_RE = re.compile(
    r"does not contain enough information|do(?:es)? not know|don't know|cannot answer|can't answer|"
    r"no (?:relevant )?information|not (?:mentioned|provided|stated)|unable to (?:answer|determine)",
    re.IGNORECASE,
)
CITED_PAGES_RE = re.compile(r"pp?\.\s*(\d+)(?:\s*[-–]\s*(\d+))?")


def group_start(page: int) -> int:
    return ((page - 1) // PAGES_PER_GROUP) * PAGES_PER_GROUP + 1


def parse_gold_pages(gold_pages: str) -> list:
    """'57-60;173-176' -> [57, 173]. Empty string (unanswerable) -> []."""
    return [int(p.split("-")[0]) for p in gold_pages.split(";") if p.strip()]


def chunk_group(chunk: dict):
    m = CHUNK_PAGES_RE.search(chunk.get("title", ""))
    return int(m.group(1)) if m else None


def fact_group(fact: dict):
    m = PAGE_ANCHOR_RE.search(fact.get("source", "") or "")
    return group_start(int(m.group(1))) if m else None


def ranked_groups(bundle: dict, k: int = None) -> list:
    """Page groups in the order the generator sees them: graph facts first,
    then vector chunks (cut to the top `k`). Duplicates are kept, since
    precision should count every retrieved item."""
    chunks = bundle.get("vector_chunks", [])
    if k is not None:
        chunks = chunks[:k]
    groups = [fact_group(f) for f in bundle.get("graph_facts", [])] + [chunk_group(c) for c in chunks]
    return [g for g in groups if g is not None]


def recall_at_k(groups: list, gold: list) -> float:
    """Fraction of the gold page groups that were retrieved."""
    return len(set(groups) & set(gold)) / len(gold) if gold else math.nan


def hit_at_k(groups: list, gold: list) -> float:
    """1 if any gold page group was retrieved (what most people mean by Recall@k for single-gold questions)."""
    return float(bool(set(groups) & set(gold))) if gold else math.nan


def reciprocal_rank(groups: list, gold: list) -> float:
    if not gold:
        return math.nan
    for rank, g in enumerate(groups, start=1):
        if g in gold:
            return 1.0 / rank
    return 0.0


def context_precision(groups: list, gold: list) -> float:
    """Fraction of retrieved items that sit on a gold page group."""
    if not gold:
        return math.nan
    return sum(g in gold for g in groups) / len(groups) if groups else 0.0


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL_RE.search(answer))


def entity_recall(answer: str, expected_entities: str) -> float:
    expected = [e.strip().lower() for e in expected_entities.split(";") if e.strip()]
    if not expected:
        return math.nan
    text = answer.lower()
    return sum(e in text for e in expected) / len(expected)


def token_f1(answer: str, gold: str) -> float:
    a = re.findall(r"\w+", answer.lower())
    g = re.findall(r"\w+", gold.lower())
    if not a or not g:
        return 0.0
    common = {}
    for t in set(a) & set(g):
        common[t] = min(a.count(t), g.count(t))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    p, r = overlap / len(a), overlap / len(g)
    return 2 * p * r / (p + r)


def judge(answer: str, row: dict) -> dict:
    """Answerable-aware verdict for one answer.

    Returns {"refused", "correct", "hallucinated_answer"}. `hallucinated_answer`
    means the model answered when it should not have (unanswerable) or answered
    with the wrong content (answerable); a refusal is never a hallucination.
    """
    refused = is_refusal(answer)
    if row["answerable"] == "no":
        return {"refused": refused, "correct": refused, "hallucinated_answer": not refused}
    recall = entity_recall(answer, row["expected_entities"])
    correct = (not refused) and recall >= 0.5
    return {"refused": refused, "correct": correct, "hallucinated_answer": (not refused) and not correct}


def citation_stats(answer: str, bundle: dict, gold: list) -> dict:
    """How the answer's page citations ("pp. 57-60") hold up.

    - cited: number of page citations found
    - valid: cited groups that were actually in the retrieved context
    - on_gold: cited groups that are gold pages
    Returns accuracy = on_gold / cited (nan if nothing cited).
    """
    cited = []
    for m in CITED_PAGES_RE.finditer(answer):
        cited.append(group_start(int(m.group(1))))
    retrieved = set(ranked_groups(bundle))
    n = len(cited)
    return {
        "n_cited": n,
        "citation_valid": sum(c in retrieved for c in cited) / n if n else math.nan,
        "citation_accuracy": sum(c in gold for c in cited) / n if n and gold else math.nan,
    }
