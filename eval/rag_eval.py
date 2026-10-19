"""Track B evaluation: closed-book vs vector vs graph vs hybrid on
data/eval/qa_test_set.csv.

Three steps, so the cheap ones never need the language model:

    python eval/rag_eval.py retrieval            # Recall@k / MRR / precision. No Ollama. Seconds.
    python eval/rag_eval.py generate --runs 3    # answers + scores. Needs Ollama. Slow; resumable.
    python eval/rag_eval.py report               # tables with mean+-SD, 95% CIs, McNemar. No models.

`generate` appends one line per (question, mode, run) to
data/eval/runs/generation.jsonl and skips lines already there, so it can be
stopped and restarted. Retrieval is deterministic; the runs differ by the
sampling seed (temperature 0.3).

How an answer is scored is defined in eval/rag_metrics.py. In short:
unanswerable questions are correct only if the model refuses; answerable ones
need the expected entities and no refusal. "Claim-level hallucination rate" is
the share of the answer's sentences that the gold pages do not entail (NLI),
so it is independent of what was retrieved.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for sub in ("scripts", "generation", "retrieval", "eval"):
    sys.path.insert(0, str(ROOT / sub))
from common import CHUNKS_JSONL, ENTITIES_CSV, REPO_ROOT, get_logger  # noqa: E402

import rag_metrics as M  # noqa: E402
import rag_stats as S  # noqa: E402

log = get_logger("rag_eval")

QA_CSV = REPO_ROOT / "data" / "eval" / "qa_test_set.csv"
RUNS_JSONL = REPO_ROOT / "data" / "eval" / "runs" / "generation.jsonl"
RETRIEVAL_CSV = REPO_ROOT / "docs" / "eval_retrieval.csv"
REPORT_MD = REPO_ROOT / "docs" / "eval_report.md"
GEN_MODES = ("closed_book", "vector", "graph", "hybrid")
TEMPERATURE = 0.3


def load_questions(ids=None) -> pd.DataFrame:
    df = pd.read_csv(QA_CSV, keep_default_na=False)
    return df[df["id"].isin(ids)] if ids else df


# ---------------------------------------------------------------- retrieval

def cmd_retrieval(args):
    from hybrid_retrieve import retrieve

    df = load_questions(args.ids)
    df = df[df["answerable"] == "yes"]
    entities_df = pd.read_csv(ENTITIES_CSV, keep_default_na=False)
    ks = sorted(args.ks)
    rows = []
    for _, q in df.iterrows():
        gold = M.parse_gold_pages(q["gold_pages"])
        for mode in ("vector", "graph", "hybrid"):
            bundle = retrieve(q["question"], entities_df, k=max(ks), mode=mode)
            for k in ks if mode != "graph" else [0]:  # graph results do not depend on k
                groups = M.ranked_groups(bundle, k=k)
                rows.append({
                    "id": q["id"], "hop_type": q["hop_type"], "mode": mode, "k": k,
                    "n_items": len(groups),
                    "hit": M.hit_at_k(groups, gold), "recall": M.recall_at_k(groups, gold),
                    "mrr": M.reciprocal_rank(groups, gold), "precision": M.context_precision(groups, gold),
                })
        log.info(f"{q['id']} done")
    out = pd.DataFrame(rows)
    RETRIEVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(RETRIEVAL_CSV, index=False)
    log.info(f"wrote {len(out)} rows to {RETRIEVAL_CSV}")
    print(retrieval_markdown(out))


def retrieval_markdown(out: pd.DataFrame) -> str:
    g = out.groupby(["mode", "k"])[["hit", "recall", "mrr", "precision", "n_items"]].mean().round(3)
    lines = ["| mode | k | hit@k | recall@k | MRR | context precision | items |", "|---|---|---|---|---|---|---|"]
    for (mode, k), r in g.iterrows():
        lines.append(f"| {mode} | {k if k else '-'} | {r.hit:.3f} | {r.recall:.3f} | {r.mrr:.3f} | {r.precision:.3f} | {r.n_items:.1f} |")
    return "\n".join(lines)


# --------------------------------------------------------------- generation

def gold_sentences(gold_pages: str, chunks_by_group: dict) -> list:
    from claim_splitting import split_into_claims

    sents = []
    for g in M.parse_gold_pages(gold_pages):
        for text in chunks_by_group.get(g, []):
            sents += split_into_claims(text)
    return sents


def load_chunks_by_group() -> dict:
    by_group = {}
    with open(CHUNKS_JSONL) as f:
        for line in f:
            c = json.loads(line)
            g = M.chunk_group(c)
            if g is not None:
                by_group.setdefault(g, []).append(c["text"])
    return by_group


def done_keys() -> set:
    if not RUNS_JSONL.exists():
        return set()
    return {(r["id"], r["mode"], r["run"]) for r in map(json.loads, RUNS_JSONL.read_text().splitlines() if RUNS_JSONL.stat().st_size else [])}


def score_answer(q: dict, answer: str, bundle: dict, chunks_by_group: dict) -> dict:
    from claim_splitting import split_into_claims
    from entailment import classify_claims
    from trust_score import compute_trust_score

    gold = M.parse_gold_pages(q["gold_pages"])
    out = M.judge(answer, q)
    out["f1"] = M.token_f1(answer, q["gold_answer"]) if q["answerable"] == "yes" else math.nan
    out.update(M.citation_stats(answer, bundle, gold))

    trust = compute_trust_score(answer, bundle) if bundle is not None else {"trust_score": None, "n_claims": 0}
    out["trust"] = trust["trust_score"] if trust["trust_score"] is not None else math.nan
    out["n_claims"] = trust["n_claims"]

    out["claim_halluc"] = math.nan
    if q["answerable"] == "yes" and not out["refused"]:
        claims = split_into_claims(answer)
        if claims:
            res = classify_claims(claims, gold_sentences(q["gold_pages"], chunks_by_group))
            out["claim_halluc"] = sum(r["verdict"] != "entailed" for r in res) / len(res)
    return out


def cmd_generate(args):
    from generate_answer import answer_question
    from ollama_client import chat, is_reachable
    from prompt_template import build_closed_book_messages

    if not is_reachable():
        log.info("Ollama not reachable -- start it first (this is the heavy step)")
        sys.exit(1)
    df = load_questions(args.ids)
    entities_df = pd.read_csv(ENTITIES_CSV, keep_default_na=False)
    chunks_by_group = load_chunks_by_group()
    done = done_keys()
    RUNS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    jobs = [(q, mode, run) for run in range(1, args.runs + 1) for mode in args.modes for _, q in df.iterrows()]
    jobs = [j for j in jobs if (j[0]["id"], j[1], j[2]) not in done]
    log.info(f"{len(jobs)} answers to generate")
    with open(RUNS_JSONL, "a") as f:
        for i, (q, mode, run) in enumerate(jobs, 1):
            q = q.to_dict() if hasattr(q, "to_dict") else q
            t0 = time.time()
            if mode == "closed_book":
                answer = chat(build_closed_book_messages(q["question"]), temperature=TEMPERATURE, seed=run)
                bundle = None
            else:
                res = answer_question(q["question"], entities_df, k=args.k, mode=mode)
                answer, bundle = res["answer"], res["bundle"]
            row = {"id": q["id"], "mode": mode, "run": run, "hop_type": q["hop_type"],
                   "answerable": q["answerable"], "k": args.k, "answer": answer,
                   "seconds": round(time.time() - t0, 1)}
            row.update(score_answer(q, answer, bundle, chunks_by_group))
            f.write(json.dumps(row, default=float) + "\n")
            f.flush()
            log.info(f"[{i}/{len(jobs)}] {q['id']} {mode} run{run}: correct={row['correct']}")


# ------------------------------------------------------------------- report

def load_runs() -> pd.DataFrame:
    return pd.DataFrame([json.loads(l) for l in RUNS_JSONL.read_text().splitlines()])


def fmt(mean, sd=None, ci=None) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "n/a"
    s = f"{mean:.3f}"
    if sd is not None:
        s += f" ± {sd:.3f}"
    if ci is not None:
        s += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
    return s


def metric_cell(d: pd.DataFrame, col: str) -> str:
    """Mean ± SD across runs, plus a 95% bootstrap CI over questions."""
    if d[col].dropna().empty:
        return "n/a"
    per_run = d.groupby("run")[col].mean()
    per_q = d.groupby("id")[col].mean()
    m, sd = S.mean_sd(per_run.values)
    return fmt(m, sd, S.bootstrap_ci(per_q.values))


def cmd_report(args):
    runs = load_runs()
    for c in ("correct", "refused", "hallucinated_answer"):
        runs[c] = runs[c].astype(float)
    n_runs = runs["run"].nunique()
    modes = [m for m in GEN_MODES if m in set(runs["mode"])]
    out = [f"# Evaluation report\n\n{runs['id'].nunique()} questions, {n_runs} run(s) per system. "
           "Cells are mean ± SD across runs, then a 95% bootstrap CI over questions in brackets.\n"]

    out.append("## Overall (all questions)\n")
    out.append("| system | correct | hallucinated answer | refused |\n|---|---|---|---|")
    for m in modes:
        d = runs[runs["mode"] == m]
        out.append(f"| {m} | {metric_cell(d, 'correct')} | {metric_cell(d, 'hallucinated_answer')} | {metric_cell(d, 'refused')} |")

    out.append("\n## Answerable questions: quality and hallucination\n")
    out.append("| system | correct | token-F1 | claim-level hallucination rate | trust score | citation accuracy |\n|---|---|---|---|---|---|")
    for m in modes:
        d = runs[(runs["mode"] == m) & (runs["answerable"] == "yes")]
        out.append(f"| {m} | {metric_cell(d, 'correct')} | {metric_cell(d, 'f1')} | {metric_cell(d, 'claim_halluc')} | "
                   f"{metric_cell(d, 'trust')} | {metric_cell(d, 'citation_accuracy')} |")

    out.append("\n## By question type (correct)\n")
    hops = ["single_hop", "multi_hop", "temporal", "unanswerable"]
    out.append("| system | " + " | ".join(hops) + " |\n|---|" + "---|" * len(hops))
    for m in modes:
        cells = [metric_cell(runs[(runs["mode"] == m) & (runs["hop_type"] == h)], "correct") for h in hops]
        out.append(f"| {m} | " + " | ".join(cells) + " |")

    out.append("\n## Paired comparison (exact McNemar on per-question majority correctness)\n")
    out.append("| A vs B | A only right | B only right | p |\n|---|---|---|---|")
    maj = {m: (runs[runs["mode"] == m].groupby("id")["correct"].mean() >= 0.5) for m in modes}
    for a, b in [("hybrid", "vector"), ("vector", "closed_book"), ("hybrid", "closed_book"), ("graph", "vector"), ("hybrid", "graph")]:
        if a in maj and b in maj:
            ids = maj[a].index.intersection(maj[b].index)
            r = S.mcnemar_exact(maj[a][ids].values, maj[b][ids].values)
            out.append(f"| {a} vs {b} | {r['a_only']} | {r['b_only']} | {r['p_value']:.3f} |")

    if RETRIEVAL_CSV.exists():
        out.append("\n## Retrieval (answerable questions)\n")
        out.append(retrieval_markdown(pd.read_csv(RETRIEVAL_CSV)))
    REPORT_MD.write_text("\n".join(out) + "\n")
    log.info(f"wrote {REPORT_MD}")
    print("\n".join(out))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieval")
    r.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    r.add_argument("--ids", nargs="+")
    r.set_defaults(fn=cmd_retrieval)
    g = sub.add_parser("generate")
    g.add_argument("--runs", type=int, default=3)
    g.add_argument("--modes", nargs="+", choices=GEN_MODES, default=list(GEN_MODES))
    g.add_argument("--k", type=int, default=5)
    g.add_argument("--ids", nargs="+")
    g.set_defaults(fn=cmd_generate)
    rep = sub.add_parser("report")
    rep.set_defaults(fn=cmd_report)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
