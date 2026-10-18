"""NLI-based entailment scoring: given a claim and a pool of evidence texts,
determine whether the claim is entailed, contradicted, or unsupported
(neutral) -- this is the anti-circularity check the master plan calls for
(don't have the same LLM grade its own answer; use an independent,
pretrained NLI model instead).

Model: cross-encoder/nli-MiniLM2-L6-H768 -- a small, fast cross-encoder
NLI model (the "lighter sentence-transformers cross-encoder" alternative
the master plan names to roberta-large-mnli), chosen for CPU speed since
this runs over every claim x every evidence text.

Usage:
    python eval/entailment.py "Rani Lakshmibai ruled Jhansi." "Rani Lakshmibai was the ruler of the princely state of Jhansi."
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import MODEL_DEVICE, configure_runtime, get_logger  # noqa: E402

log = get_logger("entailment")

NLI_MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
ENTAILMENT_THRESHOLD = 0.5  # softmax probability above which we call a claim "entailed"

_model = None  # lazy-loaded singleton
_label_map = None


def get_model():
    global _model, _label_map
    if _model is None:
        from sentence_transformers import CrossEncoder

        configure_runtime()
        _model = CrossEncoder(NLI_MODEL_NAME, device=MODEL_DEVICE)
        # Read the label order from the model config rather than hardcoding it --
        # different NLI checkpoints don't always agree on label ordering.
        _label_map = _model.config.id2label
    return _model


def get_label_map() -> dict:
    get_model()  # ensures _label_map is populated
    return _label_map


def _softmax(logits):
    import numpy as np

    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def classify_claims(claims: list, evidence_texts: list, threshold: float = ENTAILMENT_THRESHOLD) -> list:
    """Classify every claim against a pool of evidence texts in ONE batched
    model call (all claim x evidence pairs together) instead of one call per
    claim -- same results, far less per-call overhead.

    Scores each claim against every evidence text and keeps the
    highest-entailment-probability match, i.e. "is this claim entailed by
    ANY piece of retrieved evidence" -- the right question for a RAG system
    pulling from multiple sources.

    Returns one {"claim", "verdict", "score", "best_evidence"} per claim.
    verdict is "entailed", "contradicted", "neutral", or "no_evidence" (if
    evidence_texts is empty).
    """
    if not claims:
        return []
    if not evidence_texts:
        return [{"claim": c, "verdict": "no_evidence", "score": 0.0, "best_evidence": None} for c in claims]

    evidence = list(dict.fromkeys(evidence_texts))  # drop exact duplicates; order preserved
    model = get_model()
    label_map = get_label_map()
    entailment_idx = next(i for i, name in label_map.items() if name.lower() == "entailment")

    pairs = [(ev, claim) for claim in claims for ev in evidence]
    probs = _softmax(model.predict(pairs, batch_size=32, show_progress_bar=False))

    results = []
    n_ev = len(evidence)
    for ci, claim in enumerate(claims):
        block = probs[ci * n_ev : (ci + 1) * n_ev]
        best_i = int(block[:, entailment_idx].argmax())
        best_probs = {label_map[i].lower(): float(block[best_i][i]) for i in range(block.shape[1])}
        best_label = max(best_probs, key=best_probs.get)
        entailment_score = best_probs["entailment"]
        results.append(
            {
                "claim": claim,
                "verdict": "entailed" if entailment_score >= threshold else best_label,
                "score": entailment_score,
                "best_evidence": evidence[best_i],
            }
        )
    return results


def classify_claim(claim: str, evidence_texts: list, threshold: float = ENTAILMENT_THRESHOLD) -> dict:
    """Single-claim convenience wrapper around classify_claims()."""
    return classify_claims([claim], evidence_texts, threshold=threshold)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("claim")
    parser.add_argument("evidence")
    args = parser.parse_args()

    result = classify_claim(args.claim, [args.evidence])
    log.info(f"verdict={result['verdict']} score={result['score']:.3f}")


if __name__ == "__main__":
    main()
