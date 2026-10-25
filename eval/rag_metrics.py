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
CITED_NUMBERS_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


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
    then the passages (cut to the top `k`). Duplicates are kept, since
    precision should count every retrieved item. `passages` falls back to
    `vector_chunks` for bundles saved before that field existed."""
    bundle = bundle or {}
    chunks = bundle.get("passages", bundle.get("vector_chunks", []))
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


def source_groups(bundle: dict) -> list:
    """Page group of each numbered source, in the order the LLM sees them (graph
    facts, then passages; see context_sources in retrieval/hybrid_retrieve.py).
    Unlike ranked_groups, sources without a page stay as None, so index n-1 is
    source [n]."""
    bundle = bundle or {}
    chunks = bundle.get("passages", bundle.get("vector_chunks", []))
    return [fact_group(f) for f in bundle.get("graph_facts", [])] + [chunk_group(c) for c in chunks]


def citation_stats(answer: str, bundle: dict, gold: list) -> dict:
    """How the answer's citations hold up. The model cites numbered sources
    ("[2]", "[1, 4]"); answers from older runs cite pages ("pp. 57-60"), which
    are counted too.

    - cited: number of citations found (each number in a list counts once)
    - valid: citations pointing at a source that was in the context
    - on_gold: citations whose source sits on a gold page group
    Returns accuracy = on_gold / cited (nan if nothing cited or no gold pages).
    """
    if bundle is None:
        return {"n_cited": 0, "citation_valid": math.nan, "citation_accuracy": math.nan}
    sources = source_groups(bundle)
    retrieved = {g for g in sources if g is not None}
    valid, on_gold = [], []
    for m in CITED_NUMBERS_RE.finditer(answer):
        for n in (int(x) for x in m.group(1).split(",")):
            in_range = 1 <= n <= len(sources)
            valid.append(in_range)
            on_gold.append(in_range and sources[n - 1] in gold)
    for m in CITED_PAGES_RE.finditer(answer):
        g = group_start(int(m.group(1)))
        valid.append(g in retrieved)
        on_gold.append(g in gold)
    n = len(valid)
    return {
        "n_cited": n,
        "citation_valid": sum(valid) / n if n else math.nan,
        "citation_accuracy": sum(on_gold) / n if n and gold else math.nan,
    }


STOPWORDS = frozenset(
    "a an the of in on at to for by with from and or but is was were are be been it its he she they his her their "
    "that this which who whom as not had has have do did so than then there these those into over after before "
    "also however source".split()
)


def content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1}


def lexical_support(claim: str, evidence_sentences: list) -> float:
    """Best share of the claim's content words found in any window of two
    consecutive evidence sentences. A cheap second opinion for the small NLI
    model, which scores many true paraphrases as 'neutral'."""
    words = content_words(claim)
    if not words or not evidence_sentences:
        return 0.0
    best = 0.0
    for i in range(len(evidence_sentences)):
        window = content_words(" ".join(evidence_sentences[i:i + 2]))
        best = max(best, len(words & window) / len(words))
    return best
