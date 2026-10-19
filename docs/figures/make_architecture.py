"""Draws docs/figures/architecture.{png,svg}. Run: python docs/figures/make_architecture.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent
fig, ax = plt.subplots(figsize=(12, 7.2))
ax.set_xlim(0, 120); ax.set_ylim(0, 72); ax.axis("off")

def box(x, y, w, h, text, fc="#eef2f7", ec="#33415c", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#33415c", lw=1.1))

def label(x, y, t):
    ax.text(x, y, t, fontsize=10, fontweight="bold", color="#33415c")

# Offline build
label(1, 68, "A. Offline: build the knowledge base (once)")
box(1, 55, 17, 9, "Source book (PDF)\n4-page groups,\ntext cleaning")
box(24, 55, 17, 9, "Chunking\n~180 words,\n40-word overlap")
box(47, 55, 19, 9, "Embeddings\nall-MiniLM-L6-v2")
box(72, 55, 17, 9, "FAISS vector\nindex", fc="#dff0e3")
arrow(18, 59.5, 24, 59.5); arrow(41, 59.5, 47, 59.5); arrow(66, 59.5, 72, 59.5)
box(1, 41, 24, 9, "Find names (spaCy NER)\n+ LLM typing (Gemma 3 4B)", fs=8)
box(30, 41, 28, 9, "Facts: grammar rules + NLI check,\nco-mention counts", fs=8)
box(63, 41, 20, 9, "Consolidate\n(merge names, type rules)", fs=8)
box(88, 41, 17, 9, "Neo4j knowledge\ngraph", fc="#dff0e3")
arrow(25, 45.5, 30, 45.5); arrow(58, 45.5, 63, 45.5); arrow(83, 45.5, 88, 45.5)
arrow(32, 55, 12, 50)  # chunks -> names

# Online
label(1, 35, "B. Online: answer a question")
box(1, 22, 14, 9, "Question\n(Streamlit UI /\nFastAPI)", fc="#fff3d6")
box(21, 22, 16, 9, "Entity\nextraction\n(alias match)")
box(43, 26, 19, 7, "Vector search\n(top-k, reads FAISS)", fc="#dff0e3")
box(43, 17, 19, 7, "Graph search\n(1-hop, reads Neo4j)", fc="#dff0e3")
box(68, 22, 14, 9, "Merged\ncontext")
box(88, 22, 17, 9, "Grounded prompt\n+ Gemma 3 4B\n(Ollama)")
arrow(15, 26.5, 21, 26.5); arrow(37, 26.5, 43, 29.5); arrow(37, 26.5, 43, 20.5)
arrow(62, 29.5, 68, 27); arrow(62, 20.5, 68, 26); arrow(82, 26.5, 88, 26.5)

box(86, 5, 20, 9, "Claim splitting +\nNLI check vs. evidence", fc="#fde2e2", fs=8)
box(63, 5, 20, 9, "Answer + trust score\n+ cited passages", fc="#fff3d6", bold=True)
arrow(96.5, 22, 96.5, 14); arrow(86, 9.5, 83, 9.5)

# Evaluation
box(1, 5, 31, 9, "Evaluation: 54 questions, 4 systems\n(closed-book / vector / graph / hybrid)", fc="#e6e0f5", fs=8)
box(37, 5, 22, 9, "Recall@k, MRR, claim-level\nhallucination, McNemar", fc="#e6e0f5", fs=8)
arrow(32, 9.5, 37, 9.5)
ax.text(60, 1, "Green = stores; red = hallucination check; purple = evaluation.", fontsize=8, color="#555")
fig.savefig(OUT / "architecture.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "architecture.svg", bbox_inches="tight")
