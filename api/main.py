"""FastAPI backend wrapping the hybrid retriever + grounded Ollama
generation (Parts 4-5) and trust scoring (Part 6) behind an HTTP API, for
the Streamlit frontend (ui/) to call.

Endpoints:
    POST /ask     question -> grounded answer + trust score + evidence
    GET  /health  are Ollama / Neo4j / the vector index available?
    GET  /stats   what's in the knowledge base (live Neo4j counts + corpus size)
    GET  /kg      the full entity and relationship tables

Usage:
    uvicorn api.main:app --port 8000
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `response`, however the app is launched
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generation"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from common import ENTITIES_CSV, FAISS_INDEX_DIR, FAISS_INDEX_FILE, RELATIONSHIPS_CSV, SOURCES_MANIFEST_CSV, get_logger  # noqa: E402

import pandas as pd  # noqa: E402

from ollama_client import is_reachable  # noqa: E402
from response import build_ask_response  # noqa: E402

log = get_logger("api")

_entities_df = None  # loaded once, not per-request
_stats_cache = {"at": 0.0, "value": None}
STATS_TTL_SECONDS = 60


def get_entities_df() -> pd.DataFrame:
    global _entities_df
    if _entities_df is None:
        _entities_df = pd.read_csv(ENTITIES_CSV, keep_default_na=False)
    return _entities_df


def warm_up() -> None:
    """Load the small helper models and the FAISS index at startup so the
    first question isn't slow. The LLM itself is deliberately NOT preloaded:
    it's ~3 GB, and Ollama loads it on the first question and frees it after
    it's been idle (OLLAMA_KEEP_ALIVE)."""
    started = time.perf_counter()
    try:
        from entailment import get_model as get_nli
        from entity_extraction import get_nlp
        from vector_search import get_index, get_model as get_embedder

        get_entities_df()
        get_nlp()
        get_embedder()
        get_nli()
        if FAISS_INDEX_FILE.exists():
            get_index(FAISS_INDEX_DIR)
        log.info(f"warm-up finished in {time.perf_counter() - started:.1f}s")
    except Exception as exc:  # noqa: BLE001 - never block startup on warm-up
        log.info(f"warm-up skipped: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    warm_up()
    yield


app = FastAPI(
    title="HalluciNet API",
    description="Grounded Q&A over Indian history sources, with a trust score for every answer.",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str
    k: int = 5
    mode: str = "hybrid"  # "vector" | "graph" | "hybrid"


class Claim(BaseModel):
    claim: str
    verdict: str
    score: float
    best_evidence: str | None = None


class GraphFact(BaseModel):
    text: str
    source: str = ""


class Passage(BaseModel):
    title: str
    snippet: str
    score: float
    source: str = ""


class AskResponse(BaseModel):
    question: str
    answer: str
    mode: str = "hybrid"
    trust_score: float | None
    n_claims: int
    n_entailed: int
    seed_entities: list
    sources: list
    claims: list[Claim] = []
    graph_facts: list[GraphFact] = []
    passages: list[Passage] = []
    timing_ms: dict = {}


@app.get("/health")
def health():
    neo4j_ok = False
    try:
        from graph_search import get_shared_driver

        get_shared_driver().verify_connectivity()
        neo4j_ok = True
    except Exception:  # noqa: BLE001
        pass
    return {
        "ollama_reachable": is_reachable(),
        "neo4j_reachable": neo4j_ok,
        "index_ready": FAISS_INDEX_FILE.exists(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    return build_ask_response(req.question, get_entities_df(), k=req.k, mode=req.mode)


def _neo4j_counts():
    """Live counts from the graph database, or None if Neo4j is unreachable."""
    try:
        from graph_search import get_shared_driver

        with get_shared_driver().session() as session:
            node_types = {
                r["label"]: r["n"]
                for r in session.run(
                    "MATCH (n) UNWIND labels(n) AS label WITH label, count(*) AS n "
                    "WHERE label <> 'Entity' RETURN label, n ORDER BY n DESC"
                )
            }
            relations = {
                r["rel"]: r["n"]
                for r in session.run("MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY n DESC")
            }
            n_nodes = session.run("MATCH (n) RETURN count(n) AS n").single()["n"]
        return {"nodes": n_nodes, "relationships": sum(relations.values()), "node_types": node_types, "relation_types": relations}
    except Exception as exc:  # noqa: BLE001
        log.info(f"stats: Neo4j unavailable ({exc})")
        return None


def _corpus_stats() -> dict:
    stats = {"documents": 0, "words": 0, "chunks": 0, "sources": []}
    if SOURCES_MANIFEST_CSV.exists():
        manifest = pd.read_csv(SOURCES_MANIFEST_CSV)
        stats["documents"] = int(len(manifest))
        stats["words"] = int(manifest["word_count"].sum())
        # "Book Title (pp. 1-4)" -> "Book Title"
        stats["sources"] = sorted({t.rsplit(" (pp.", 1)[0] for t in manifest["title"]})
    if FAISS_INDEX_FILE.exists():
        from vector_search import get_index

        stats["chunks"] = int(get_index(FAISS_INDEX_DIR)[0].ntotal)
    return stats


@app.get("/stats")
def stats():
    now = time.time()
    if _stats_cache["value"] is not None and now - _stats_cache["at"] < STATS_TTL_SECONDS:
        return _stats_cache["value"]

    entities = get_entities_df()
    relationships = pd.read_csv(RELATIONSHIPS_CSV, keep_default_na=False)
    live = _neo4j_counts()

    value = {
        "neo4j_live": live is not None,
        "graph": live
        or {
            "nodes": int(len(entities)),
            "relationships": int(len(relationships)),
            "node_types": entities["entity_type"].value_counts().to_dict(),
            "relation_types": relationships["relation"].value_counts().to_dict(),
        },
        "periods": sorted(set(entities["period_tag"])),
        "corpus": _corpus_stats(),
    }
    _stats_cache.update(at=now, value=value)
    return value


@app.get("/kg")
def kg():
    entities = get_entities_df()
    relationships = pd.read_csv(RELATIONSHIPS_CSV, keep_default_na=False)
    return {
        "entities": entities.to_dict(orient="records"),
        "relationships": relationships.to_dict(orient="records"),
    }
