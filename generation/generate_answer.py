"""End-to-end grounded answer generation: hybrid retrieval (Part 4) -> the
grounding-only prompt template -> local Ollama generation.

Usage:
    python generation/generate_answer.py "Who led the resistance at Jhansi?"
    python generation/generate_answer.py "Who led the resistance at Jhansi?" --k 5 --graph_limit 20
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
from common import ENTITIES_CSV, FAISS_INDEX_DIR, OLLAMA_MODEL, get_logger  # noqa: E402

from hybrid_retrieve import DEFAULT_HOPS_LIMIT, RETRIEVAL_MODES, format_context_only, retrieve  # noqa: E402
from ollama_client import chat, is_reachable  # noqa: E402
from prompt_template import build_messages  # noqa: E402

log = get_logger("generate_answer")


def answer_question(
    question: str,
    entities_df: pd.DataFrame,
    k: int = 5,
    graph_limit: int = DEFAULT_HOPS_LIMIT,
    index_dir: Path = FAISS_INDEX_DIR,
    model: str = OLLAMA_MODEL,
    mode: str = "hybrid",
) -> dict:
    """Run retrieval, build the grounded prompt, and generate an answer.

    Returns {"question", "answer", "context", "bundle"} -- `bundle` is the
    raw retrieval output (Part 4's retrieve()), kept around so Part 6's
    trust scoring can re-check the answer's claims against the same
    evidence it was generated from. `mode` ("vector"/"graph"/"hybrid") is
    forwarded to retrieve() -- see Part 7's ablation.
    """
    bundle = retrieve(question, entities_df, k=k, graph_limit=graph_limit, index_dir=index_dir, mode=mode)
    context = format_context_only(bundle)

    messages = build_messages(question, context)
    answer = chat(messages, model=model)

    return {"question": question, "answer": answer, "context": context, "bundle": bundle}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--graph_limit", type=int, default=DEFAULT_HOPS_LIMIT)
    parser.add_argument("--entities", type=Path, default=ENTITIES_CSV)
    parser.add_argument("--index_dir", type=Path, default=FAISS_INDEX_DIR)
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--mode", choices=RETRIEVAL_MODES, default="hybrid")
    args = parser.parse_args()

    if not is_reachable():
        log.info("Ollama not reachable -- see docs/manual_setup_ollama.md")
        sys.exit(1)

    entities_df = pd.read_csv(args.entities, keep_default_na=False)
    result = answer_question(
        args.question,
        entities_df,
        k=args.k,
        graph_limit=args.graph_limit,
        index_dir=args.index_dir,
        model=args.model,
        mode=args.mode,
    )

    log.info(f"seed entities: {result['bundle']['seed_entities']}")
    print(f"\nQ: {result['question']}\n")
    print(f"A: {result['answer']}\n")


if __name__ == "__main__":
    main()
