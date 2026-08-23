"""FastAPI backend wrapping the hybrid retriever + grounded Ollama
generation (Parts 4-5) and trust scoring (Part 6) behind an HTTP endpoint,
for the Streamlit frontend (ui/) to call.

Usage:
    uvicorn api.main:app --reload --port 8000
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generation"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from common import ENTITIES_CSV, get_logger  # noqa: E402

import pandas as pd  # noqa: E402

from generate_answer import answer_question  # noqa: E402
from ollama_client import is_reachable  # noqa: E402
from trust_score import compute_trust_score  # noqa: E402

log = get_logger("api")

app = FastAPI(title="HalluciNet API", description="Grounded Q&A over the 1857 Revolt with a trust score.")

_entities_df = None  # loaded once at startup, not per-request


def get_entities_df() -> pd.DataFrame:
    global _entities_df
    if _entities_df is None:
        _entities_df = pd.read_csv(ENTITIES_CSV, keep_default_na=False)
    return _entities_df


class AskRequest(BaseModel):
    question: str
    k: int = 5
    mode: str = "hybrid"  # "vector" | "graph" | "hybrid"


class Source(BaseModel):
    title: str
    url: str


class AskResponse(BaseModel):
    question: str
    answer: str
    trust_score: float | None
    n_claims: int
    n_entailed: int
    seed_entities: list
    sources: list


@app.get("/health")
def health():
    return {"ollama_reachable": is_reachable()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    entities_df = get_entities_df()
    result = answer_question(req.question, entities_df, k=req.k, mode=req.mode)
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

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        trust_score=scoring["trust_score"],
        n_claims=scoring["n_claims"],
        n_entailed=scoring["n_entailed"],
        seed_entities=[name for _, name in result["bundle"]["seed_entities"]],
        sources=sources,
    )
