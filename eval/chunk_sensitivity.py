"""Chunk-size sensitivity for vector retrieval (retrieval metrics only; no language model).

Takes the book text already in data/processed/chunks.jsonl, re-cuts it into
chunks of different sizes, embeds them with the same model, and measures
hit@k / Recall@k / MRR / context precision against the gold pages of the
answerable questions -- the same metrics as `rag_eval.py retrieval`. The
production index is untouched: everything is built in memory.

    python eval/chunk_sensitivity.py
    python eval/chunk_sensitivity.py --sizes 60:15 120:30 180:40 360:80

Note the embedding model reads at most `max_seq_length` word pieces, so a chunk
much longer than ~180 words is partly ignored when embedded; the script prints
that limit.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for sub in ("scripts", "kg", "eval"):
    sys.path.insert(0, str(ROOT / sub))
from common import CHUNKS_JSONL, EMBEDDING_MODEL_NAME, MODEL_DEVICE, REPO_ROOT, configure_runtime, get_logger  # noqa: E402

import rag_metrics as M  # noqa: E402
from book_text import reconstruct_documents  # noqa: E402
from clean_and_chunk import chunk_words  # noqa: E402

log = get_logger("chunk_sensitivity")

QA_CSV = REPO_ROOT / "data" / "eval" / "qa_test_set.csv"
OUT_CSV = REPO_ROOT / "docs" / "eval_chunk_sensitivity.csv"
OUT_MD = REPO_ROOT / "docs" / "eval_chunk_sensitivity.md"
KS = (1, 3, 5, 10)


def rechunk(docs: list, size: int, overlap: int) -> list:
    chunks = []
    for d in docs:
        for window in chunk_words(d["text"].split(), chunk_size=size, overlap=overlap):
            chunks.append({"title": d["title"], "text": " ".join(window)})
    return chunks


def evaluate(chunks: list, questions: pd.DataFrame, model) -> pd.DataFrame:
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True, convert_to_numpy=True,
                       batch_size=32, show_progress_bar=False).astype("float32")
    qvecs = model.encode(list(questions["question"]), normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    order = np.argsort(-(qvecs @ emb.T), axis=1)[:, : max(KS)]
    groups = [M.chunk_group(c) for c in chunks]
    rows = []
    for qi, (_, q) in enumerate(questions.iterrows()):
        gold = M.parse_gold_pages(q["gold_pages"])
        ranked = [groups[j] for j in order[qi]]
        for k in KS:
            top = [g for g in ranked[:k] if g is not None]
            rows.append({"id": q["id"], "hop_type": q["hop_type"], "k": k,
                         "hit": M.hit_at_k(top, gold), "recall": M.recall_at_k(top, gold),
                         "mrr": M.reciprocal_rank(top, gold), "precision": M.context_precision(top, gold)})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", nargs="+", default=["60:15", "120:30", "180:40", "360:80"], help="size:overlap pairs, in words")
    args = p.parse_args()

    from sentence_transformers import SentenceTransformer

    configure_runtime()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=MODEL_DEVICE)
    log.info(f"embedding model reads at most {model.max_seq_length} word pieces per chunk")
    docs = reconstruct_documents(CHUNKS_JSONL)
    qa = pd.read_csv(QA_CSV, keep_default_na=False)
    qa = qa[qa["answerable"] == "yes"]

    frames = []
    for spec in args.sizes:
        size, overlap = map(int, spec.split(":"))
        chunks = rechunk(docs, size, overlap)
        log.info(f"chunk size {size} (overlap {overlap}): {len(chunks)} chunks")
        res = evaluate(chunks, qa, model)
        res["chunk_words"], res["overlap"], res["n_chunks"] = size, overlap, len(chunks)
        frames.append(res)

    out = pd.concat(frames)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    g = out.groupby(["chunk_words", "k"])[["hit", "recall", "mrr", "precision"]].mean()
    meta = out.groupby("chunk_words")[["overlap", "n_chunks"]].first()
    lines = ["| chunk words (overlap) | chunks | k | hit@k | recall@k | MRR | context precision |", "|---|---|---|---|---|---|---|"]
    for (size, k), r in g.iterrows():
        lines.append(f"| {size} ({meta.loc[size, 'overlap']}) | {meta.loc[size, 'n_chunks']} | {k} | {r.hit:.3f} | {r.recall:.3f} | {r.mrr:.3f} | {r.precision:.3f} |")
    md = "\n".join(lines)
    OUT_MD.write_text(f"# Chunk-size sensitivity (vector retrieval, {len(qa)} answerable questions)\n\n"
                      f"Embedding model reads at most {model.max_seq_length} word pieces per chunk.\n\n{md}\n")
    print(md)


if __name__ == "__main__":
    main()
