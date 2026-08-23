"""Render the Week 7 "core proof slide": accuracy by retrieval mode
(vector-only / graph-only / hybrid), from eval/ablation.py's results CSV.

Usage:
    python eval/chart.py
    python eval/chart.py --results docs/week7_ablation_results.csv --out docs/week7_accuracy_by_mode.png
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import REPO_ROOT, get_logger  # noqa: E402

log = get_logger("chart")

DEFAULT_RESULTS = REPO_ROOT / "docs" / "week7_ablation_results.csv"
DEFAULT_OUT = REPO_ROOT / "docs" / "week7_accuracy_by_mode.png"

MODE_ORDER = ["vector", "graph", "hybrid"]
MODE_LABELS = {"vector": "Vector-only", "graph": "Graph-only", "hybrid": "Hybrid"}
VERDICT_COLORS = {
    "correct": "#2ca02c",
    "partial": "#ffbf00",
    "wrong": "#ff7f0e",
    "hallucinated": "#d62728",
    "refused": "#7f7f7f",
}
VERDICT_ORDER = ["correct", "partial", "wrong", "hallucinated", "refused"]


def build_chart(results_df: pd.DataFrame, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")  # headless -- just save a file, no display needed
    import matplotlib.pyplot as plt

    counts = results_df.groupby(["mode", "verdict"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=MODE_ORDER, columns=VERDICT_ORDER, fill_value=0)
    proportions = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = pd.Series([0.0] * len(proportions), index=proportions.index)
    for verdict in VERDICT_ORDER:
        ax.bar(
            [MODE_LABELS[m] for m in proportions.index],
            proportions[verdict],
            bottom=bottom,
            label=verdict,
            color=VERDICT_COLORS[verdict],
        )
        bottom += proportions[verdict]

    ax.set_ylabel("% of questions")
    ax.set_title("HalluciNet: answer quality by retrieval mode")
    ax.set_ylim(0, 100)
    ax.legend(title="Verdict", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.results.exists():
        log.info(f"{args.results} not found -- run eval/ablation.py first")
        sys.exit(1)

    results_df = pd.read_csv(args.results)
    build_chart(results_df, args.out)
    log.info(f"wrote chart to {args.out}")


if __name__ == "__main__":
    main()
