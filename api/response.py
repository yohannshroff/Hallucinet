"""Shared response-building logic: runs the full Week 4-6 pipeline
(retrieval -> generation -> trust score) and shapes the result into the
dict both api/main.py's /ask endpoint and the demo-cache builder need.
Kept out of main.py so it has no FastAPI dependency and can be imported
standalone by scripts/tests.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generation"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from generate_answer import answer_question  # noqa: E402
from trust_score import compute_trust_score  # noqa: E402


def build_ask_response(question: str, entities_df: pd.DataFrame, k: int = 5, mode: str = "hybrid") -> dict:
    result = answer_question(question, entities_df, k=k, mode=mode)
    scoring = compute_trust_score(result["answer"], result["bundle"])

    sources = []
    seen_urls = set()
    for fact in result["bundle"]["graph_facts"]:
        if fact["source"] and fact["source"] not in seen_urls:
            sources.append({"title": f"{fact['subject']} {fact['relation']} {fact['object']}", "url": fact["source"]})
            seen_urls.add(fact["source"])
    for chunk in result["bundle"]["vector_chunks"]:
        if chunk["source_url"] not in seen_urls:
            sources.append({"title": chunk["title"], "url": chunk["source_url"]})
            seen_urls.add(chunk["source_url"])

    return {
        "question": result["question"],
        "answer": result["answer"],
        "trust_score": scoring["trust_score"],
        "n_claims": scoring["n_claims"],
        "n_entailed": scoring["n_entailed"],
        "seed_entities": [name for _, name in result["bundle"]["seed_entities"]],
        "sources": sources,
    }
