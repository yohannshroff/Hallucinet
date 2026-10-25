"""Cross-encoder reranking of retrieved passages.

The embedding search compares a question vector with each chunk vector
separately; a cross-encoder reads the question and the passage together,
so it can tell "the siege of Cawnpore in June" from "the battle at Cawnpore
in November". hybrid_retrieve.py fetches a larger pool of candidates and
keeps the best k by this score.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M parameters, ~90 MB), a
passage-ranking model of the same small family as the NLI model in
eval/entailment.py; ~1-2 s for 40 passages on a laptop CPU.

Set HALLUCINET_RERANK=0 to turn it off (to compare with plain similarity ranking).

Usage:
    python retrieval/rerank.py "Who led the resistance at Jhansi?"
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import MODEL_DEVICE, configure_runtime, get_logger  # noqa: E402

log = get_logger("rerank")

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK = os.environ.get("HALLUCINET_RERANK", "1") != "0"

_model = None  # lazy-loaded singleton, same pattern as vector_search's embedding model


def get_reranker():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        configure_runtime()
        _model = CrossEncoder(RERANK_MODEL, device=MODEL_DEVICE)
    return _model


def rerank(query: str, passages: list, k: int) -> list:
    """The k passages that best answer `query`, best first, each with a `rerank_score`."""
    if not passages:
        return []
    scores = get_reranker().predict([(query, p["text"]) for p in passages], batch_size=32, show_progress_bar=False)
    ranked = sorted(zip(passages, scores), key=lambda ps: -float(ps[1]))
    return [{**p, "rerank_score": float(s)} for p, s in ranked[:k]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--pool", type=int, default=20)
    args = parser.parse_args()

    from vector_search import search

    pool = search(args.query, k=args.pool)
    for p in rerank(args.query, pool, args.k):
        print(f"  rerank={p['rerank_score']:.2f}  cosine={p['score']:.3f}  {p['title']}")


if __name__ == "__main__":
    main()
