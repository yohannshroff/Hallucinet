"""Shared response-building logic: runs the full Part 4-6 pipeline
(retrieval -> generation -> trust score) and shapes the result into the
dict both api/main.py's /ask endpoint and the demo-cache builder need.
Kept out of main.py so it has no FastAPI dependency and can be imported
standalone by scripts/tests.
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generation"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from common import bundle_passages, is_web_url  # noqa: E402
from generate_answer import answer_question  # noqa: E402
from graph_search import format_fact  # noqa: E402
from trust_score import compute_trust_score, fact_to_sentence  # noqa: E402

SNIPPET_CHARS = 360


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "..."


def build_ask_response(question: str, entities_df: pd.DataFrame, k: int = 5, mode: str = "hybrid") -> dict:
    t0 = time.perf_counter()
    result = answer_question(question, entities_df, k=k, mode=mode)
    t1 = time.perf_counter()
    scoring = compute_trust_score(result["answer"], result["bundle"])
    t2 = time.perf_counter()

    bundle = result["bundle"]

    # `sentence` is the wording the LLM saw ("X and Y are mentioned together in the book."), so the
    # source list reads the same as the context it cited.
    graph_facts = [
        {"text": format_fact(f), "sentence": fact_to_sentence(f), "source": f["source"] or ""} for f in bundle["graph_facts"]
    ]
    passages = [
        {
            "title": c["title"],
            "snippet": _snippet(c["text"]),
            "score": round(c["score"], 3),
            "source": c["source_url"],
        }
        for c in bundle_passages(bundle)
    ]

    # Flat de-duplicated citation list (kept for older clients / the demo cache).
    # Local file references (book passages) have no clickable link, so url is "".
    sources, seen = [], set()
    for fact in bundle["graph_facts"]:
        if fact["source"] and fact["source"] not in seen:
            sources.append({"title": f"{fact['subject']} {fact['relation']} {fact['object']}", "url": fact["source"]})
            seen.add(fact["source"])
    for chunk in bundle_passages(bundle):
        if chunk["source_url"] not in seen:
            sources.append({"title": chunk["title"], "url": chunk["source_url"] if is_web_url(chunk["source_url"]) else ""})
            seen.add(chunk["source_url"])

    return {
        "question": result["question"],
        "answer": result["answer"],
        "mode": mode,
        "trust_score": scoring["trust_score"],
        "n_claims": scoring["n_claims"],
        "n_entailed": scoring["n_entailed"],
        "seed_entities": [name for _, name in bundle["seed_entities"]],
        "sources": sources,
        "claims": [
            {
                "claim": c["claim"],
                "verdict": c["verdict"],
                "score": round(c["score"], 3),
                "best_evidence": c["best_evidence"],
            }
            for c in scoring["claims"]
        ],
        "graph_facts": graph_facts,
        "passages": passages,
        "timing_ms": {
            "answer": round((t1 - t0) * 1000),
            "trust_check": round((t2 - t1) * 1000),
        },
    }
