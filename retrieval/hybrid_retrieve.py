"""Hybrid retrieval: entity extraction -> graph search + vector search ->
merged context.

The first evaluation showed that bare graph triples ("A --associated_with-->
B", 18 per question, unranked) gave the LLM nothing to quote and diluted the
context. So the graph is now used two ways:
- Graph facts: the typed 1-hop facts of the question's entities plus the
  facts on short paths between them (graph_search.path_facts), ranked by
  similarity to the question and capped at GRAPH_FACT_CAP. Plain co-mention
  edges (associated_with) are left out unless they lie on such a path.
- Graph-guided passages: book chunks that name a question entity *and* one
  of its graph neighbours (or another question entity), ranked by
  similarity to the question. These carry the actual text and dates.

Passages the LLM sees (`passages` in the bundle) come from a pool of
RERANK_POOL candidates per source, limited to the book's narrative pages
(common.NARRATIVE_PAGES), and the best k by the cross-encoder reranker
(retrieval/rerank.py):
- vector: the vector-search pool
- graph:  the graph-guided pool
- hybrid: both pools merged with reciprocal rank fusion (RRF)
Passages next to one already chosen are skipped (distinct_top), and each is
then shown with its neighbouring chunks (widen_passages).
With HALLUCINET_RERANK=0 the top k of the pool are taken as they are.

Usage:
    python retrieval/hybrid_retrieve.py "Who led the resistance at Jhansi?"
    python retrieval/hybrid_retrieve.py "Who led the resistance at Jhansi?" --k 5 --graph_limit 20
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import (  # noqa: E402
    CHUNK_OVERLAP_WORDS,
    ENTITIES_CSV,
    FAISS_INDEX_DIR,
    FAISS_INDEX_FILE,
    build_alias_index,
    bundle_passages,
    get_logger,
    is_narrative,
    is_web_url,
)

from entity_extraction import MIN_SUBSTRING_ALIAS_LENGTH, extract_entities  # noqa: E402
from graph_search import (  # noqa: E402
    DEFAULT_HOPS_LIMIT,
    format_fact,
    get_shared_driver,
    path_facts,
    rank_facts,
    reset_shared_driver,
    search as graph_search,
)
from rerank import RERANK, rerank  # noqa: E402
from vector_search import get_index, get_model, search as vector_search  # noqa: E402

log = get_logger("hybrid_retrieve")

RETRIEVAL_MODES = ("vector", "graph", "hybrid")
GRAPH_FACT_CAP = 6  # facts shown to the LLM, after ranking by the question
RRF_K = 60  # standard reciprocal-rank-fusion constant
# Search matches small 60-word chunks, but the LLM reads each one with this many
# neighbouring chunks on either side (~150 words), so "she" or "the Rajah" comes
# with the sentence that says who that is.
PASSAGE_WINDOW = 1
RERANK_POOL = 20  # candidates per source handed to the cross-encoder reranker (retrieval/rerank.py)


def encode(texts: list):
    """Normalized embeddings (dot product = cosine), from the already-loaded model."""
    return get_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def entity_chunk_rows(entities_df: pd.DataFrame, id_map: list) -> dict:
    """entity_id -> set of id_map rows whose text names the entity (by name or
    alias). Built once per (entities, index) pair -- a regex pass over the
    chunks, well under a second -- and cached on the DataFrame."""
    cached = entities_df.attrs.get("entity_chunk_rows")
    if cached and cached[0] == id(id_map):
        return cached[1]
    alias_index = get_alias_index(entities_df)
    aliases = sorted((a for a in alias_index if len(a) >= MIN_SUBSTRING_ALIAS_LENGTH), key=len, reverse=True)
    pattern = re.compile(r"(?<![\w'’])(" + "|".join(re.escape(a) for a in aliases) + r")(?![\w])", re.IGNORECASE)
    rows = defaultdict(set)
    for i, entry in enumerate(id_map):
        for m in pattern.finditer(entry["text"]):
            rows[alias_index[m.group(1).lower()]].add(i)
    entities_df.attrs["entity_chunk_rows"] = (id(id_map), rows)
    return rows


def graph_passages(query: str, seed_ids: list, facts: list, entities_df: pd.DataFrame, index_dir: Path, k: int) -> list:
    """Chunks that name a question entity together with one of its graph
    neighbours or another question entity, ranked by similarity to the
    question. Falls back to chunks naming any question entity. Works without
    Neo4j (then only co-mentions of question entities count)."""
    import numpy as np

    index, id_map = get_index(index_dir)
    rows_by_entity = entity_chunk_rows(entities_df, id_map)
    alias_index = get_alias_index(entities_df)
    seeds = set(seed_ids)
    neighbours = {alias_index.get(str(name).lower()) for f in facts for name in (f["subject"], f["object"])} - {None} - seeds

    seed_hits = Counter(r for s in seeds for r in rows_by_entity.get(s, ()))
    neighbour_rows = {r for n in neighbours for r in rows_by_entity.get(n, ())}
    candidates = [r for r, n in seed_hits.items() if n >= 2 or r in neighbour_rows] or list(seed_hits)
    candidates = [r for r in candidates if is_narrative(id_map[r]["title"])]
    if not candidates:
        return []

    qvec = encode([query])[0]
    vecs = np.vstack([index.reconstruct(int(r)) for r in candidates])
    scores = vecs @ qvec
    top = sorted(range(len(candidates)), key=lambda i: -scores[i])[:k]
    out = []
    for i in top:
        entry = id_map[candidates[i]]
        out.append({
            "score": float(scores[i]), "chunk_id": entry.get("chunk_id"), "doc_id": entry["doc_id"],
            "title": entry["title"], "source_url": entry["source_url"], "text": entry["text"], "via": "graph",
        })
    return out


def rrf_merge(ranked_lists: list, k: int, rrf_k: int = RRF_K) -> list:
    """Reciprocal rank fusion: score = sum of 1 / (rrf_k + rank) over the lists
    a chunk appears in. A chunk found by both sources is marked via="graph+vector"."""
    scores, items, via = Counter(), {}, defaultdict(set)
    for lst in ranked_lists:
        for rank, chunk in enumerate(lst, start=1):
            key = chunk.get("chunk_id") or (chunk["title"], chunk["text"])
            scores[key] += 1.0 / (rrf_k + rank)
            items.setdefault(key, chunk)
            via[key].add(chunk.get("via", "vector"))
    order = sorted(scores, key=lambda key: -scores[key])[:k]
    return [{**items[key], "via": "+".join(sorted(via[key]))} for key in order]


def widen_passages(passages: list, id_map: list, window: int = PASSAGE_WINDOW) -> list:
    """Replace each passage's text with its chunk plus `window` neighbouring chunks
    on each side from the same page group, stitched without the overlap (the same
    rule as kg/book_text.py). Passages whose chunk isn't in id_map are kept as is."""
    if window <= 0 or not passages:
        return passages
    rows = id_map_rows(id_map)
    out = []
    for p in passages:
        r = rows.get(p.get("chunk_id"))
        if r is None:
            out.append(p)
            continue
        doc = id_map[r]["doc_id"]
        lo = r
        while lo > 0 and r - lo < window and id_map[lo - 1]["doc_id"] == doc:
            lo -= 1
        hi = r
        while hi + 1 < len(id_map) and hi - r < window and id_map[hi + 1]["doc_id"] == doc:
            hi += 1
        words = []
        for i in range(lo, hi + 1):
            w = id_map[i]["text"].split()
            words += w if i == lo else w[CHUNK_OVERLAP_WORDS:]
        out.append({**p, "text": " ".join(words)})
    return out


def distinct_top(passages: list, id_map: list, k: int, window: int = PASSAGE_WINDOW) -> list:
    """The first k passages, skipping any whose chunk lies within `window` chunks of
    one already taken: once widened, neighbours would show the model the same text twice."""
    rows = id_map_rows(id_map)
    taken, out = [], []
    for p in passages:
        r = rows.get(p.get("chunk_id"))
        if r is not None and any(abs(r - t) <= 2 * window for t in taken):
            continue
        out.append(p)
        if r is not None:
            taken.append(r)
        if len(out) == k:
            break
    return out


def id_map_rows(id_map: list) -> dict:
    """chunk_id -> row in id_map, built once per loaded index."""
    cached = getattr(id_map_rows, "_cache", None)
    if cached is None or cached[0] is not id_map:
        id_map_rows._cache = (id_map, {e.get("chunk_id"): i for i, e in enumerate(id_map)})
    return id_map_rows._cache[1]


def get_alias_index(entities_df: pd.DataFrame) -> dict:
    """Build the name/alias -> entity_id lookup once per DataFrame, not once per question."""
    if "alias_index" not in entities_df.attrs:
        entities_df.attrs["alias_index"] = build_alias_index(entities_df)
    return entities_df.attrs["alias_index"]


def retrieve(
    query: str,
    entities_df: pd.DataFrame,
    k: int = 5,
    graph_limit: int = DEFAULT_HOPS_LIMIT,
    index_dir: Path = FAISS_INDEX_DIR,
    mode: str = "hybrid",
) -> dict:
    """Run entity extraction, graph search, and vector search; return a
    merged context bundle. Returns [] for graph_facts if Neo4j isn't
    reachable, so callers can still get vector results (and graph-guided
    passages from co-mentions of the question's entities).

    `mode` selects which retrieval source(s) populate the bundle -- "vector"
    or "graph" alone, or "hybrid" (both) -- used by Part 7's ablation to
    compare them. `vector_chunks` only ever holds vector-search results and
    `graph_chunks` only graph-guided passages; `passages` is what the LLM
    sees (see the module docstring).
    """
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"mode must be one of {RETRIEVAL_MODES}, got {mode!r}")

    seed_entity_ids = extract_entities(query, get_alias_index(entities_df))

    all_facts, graph_facts, graph_chunks = [], [], []
    if mode in ("graph", "hybrid") and seed_entity_ids:
        try:
            with get_shared_driver().session() as session:
                paths = path_facts(session, seed_entity_ids) if len(seed_entity_ids) > 1 else []
                one_hop = graph_search(session, seed_entity_ids, limit_per_entity=graph_limit)
            seen = {(f["subject"], f["relation"], f["object"]) for f in paths}
            all_facts = paths + [f for f in one_hop if (f["subject"], f["relation"], f["object"]) not in seen]
        except Exception as exc:  # noqa: BLE001
            log.info(f"graph search skipped (Neo4j unreachable: {exc})")
            reset_shared_driver()
        # A bare co-mention ("Jhansi and Sagar are mentioned together") says nothing about
        # roles, and the model read them as such ("Sagar led the resistance"). They still pick
        # the graph-guided passages below; the LLM only sees them when they link two of the
        # question's entities (path facts).
        # "Siege of Cawnpore is located in Cawnpore" only repeats the event's name; the model
        # padded answers with such facts, so they are left out too.
        shown = [
            f for f in all_facts
            if (f["relation"] != "associated_with" or f.get("via") == "path")
            and not (f["relation"] == "located_in" and str(f["object"]) in str(f["subject"]))
        ]
        graph_facts = rank_facts(query, shown, GRAPH_FACT_CAP, encode)

    # Each source contributes a pool of candidates (more than k when reranking); the
    # reranker then picks the best k from the mode's own pool(s), so the vector / graph /
    # hybrid comparison stays fair.
    pool = RERANK_POOL if RERANK else k
    graph_pool = []
    if mode in ("graph", "hybrid") and seed_entity_ids and (Path(index_dir) / FAISS_INDEX_FILE.name).exists():
        graph_pool = graph_passages(query, seed_entity_ids, all_facts, entities_df, index_dir, pool)

    vector_pool = []
    if mode in ("vector", "hybrid"):
        hits = vector_search(query, index_dir=index_dir, k=2 * pool)  # some fall outside the narrative pages
        vector_pool = [{**c, "via": "vector"} for c in hits if is_narrative(c["title"])][:pool]

    if mode == "hybrid":
        candidates = rrf_merge([vector_pool, graph_pool], len(vector_pool) + len(graph_pool))
    else:
        candidates = vector_pool or graph_pool
    # Pure reranker order. Fusing it with the search order (RRF) fixed one question it gets
    # wrong (a "gallant resistance" passage for "who led the resistance at Jhansi?") but
    # lowered hit@1 on the test set from 0.86 to 0.75 (hybrid), so it is not done.
    ranked = rerank(query, candidates, len(candidates)) if RERANK else candidates
    passages = ranked[:k]
    vector_chunks, graph_chunks = vector_pool[:k], graph_pool[:k]
    if passages and (Path(index_dir) / FAISS_INDEX_FILE.name).exists():
        id_map = get_index(index_dir)[1]
        passages = widen_passages(distinct_top(ranked, id_map, k), id_map)

    seed_names = [
        entities_df.loc[entities_df["entity_id"] == eid, "name"].iloc[0] for eid in seed_entity_ids
    ]

    return {
        "query": query,
        "seed_entities": list(zip(seed_entity_ids, seed_names)),
        "graph_facts": graph_facts,
        "vector_chunks": vector_chunks,
        "graph_chunks": graph_chunks,
        "passages": passages,
    }


TITLE_PAGES_RE = re.compile(r"^(.*?)(?: - [^()]*)?\s*\(pp\. (\d+)-(\d+)\)$")
PAGE_ANCHOR_RE = re.compile(r"#page=(\d+)")


def short_title(title: str) -> str:
    """'The Great Mutiny - India 1857 by Christopher Hibbert (pp. 401-404)' -> 'The Great Mutiny, pp. 401-404'."""
    m = TITLE_PAGES_RE.match(title)
    return f"{m.group(1)}, pp. {m.group(2)}-{m.group(3)}" if m else title


def fact_citation(fact: dict) -> str:
    """Where a graph fact comes from, without the file path: a web link, 'p. 393', or ''."""
    source = fact.get("source") or ""
    if is_web_url(source):
        return source
    m = PAGE_ANCHOR_RE.search(source)
    return f"p. {m.group(1)}" if m else ""


def context_sources(bundle: dict) -> list:
    """The numbered sources the LLM sees, in citation order: graph facts first, then
    passages. Each is (label, text); the UI numbers its source list the same way."""
    from trust_score import fact_to_sentence  # "X and Y are mentioned together in the book." etc.

    out = []
    for fact in bundle["graph_facts"]:
        where = fact_citation(fact)
        out.append(("Knowledge graph" + (f", {where}" if where else ""), fact_to_sentence(fact)))
    for chunk in bundle_passages(bundle):
        # A web link is a useful citation; a local "data/raw/book.pdf#page=N" path is
        # noise the model would copy into its answer -- the title has the pages.
        where = f" ({chunk['source_url']})" if is_web_url(chunk["source_url"]) else ""
        out.append((short_title(chunk["title"]) + where, chunk["text"]))
    return out


def format_context_only(bundle: dict) -> str:
    """Render just the retrieved context as one numbered source list, with no
    query line -- used as the context block inside an LLM prompt (Part 5),
    where the question is already provided separately. The model cites by
    these numbers ([1], [2], ...)."""
    sources = context_sources(bundle)
    if not sources:
        return "(no context retrieved)"
    return "\n".join(f"[{i}] {label}: {text}" for i, (label, text) in enumerate(sources, start=1))


def format_context_for_llm(bundle: dict) -> str:
    """Like format_context_only, but with a leading query line -- used for
    the CLI's own human-readable display."""
    return f"Query: {bundle['query']}\n\n" + format_context_only(bundle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5, help="Top-k vector chunks")
    parser.add_argument("--graph_limit", type=int, default=DEFAULT_HOPS_LIMIT, help="Max facts per seed entity")
    parser.add_argument("--entities", type=Path, default=ENTITIES_CSV)
    parser.add_argument("--index_dir", type=Path, default=FAISS_INDEX_DIR)
    parser.add_argument("--mode", choices=RETRIEVAL_MODES, default="hybrid")
    args = parser.parse_args()

    entities_df = pd.read_csv(args.entities, keep_default_na=False)
    bundle = retrieve(args.query, entities_df, k=args.k, graph_limit=args.graph_limit, index_dir=args.index_dir, mode=args.mode)

    log.info(f"seed entities: {bundle['seed_entities']}")
    log.info(
        f"{len(bundle['graph_facts'])} graph fact(s); {len(bundle['passages'])} passage(s) "
        f"({len(bundle['vector_chunks'])} from vector search, {len(bundle['graph_chunks'])} graph-guided)"
    )
    print()
    print(format_context_for_llm(bundle))


if __name__ == "__main__":
    main()
