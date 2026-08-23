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
from common import get_logger  # noqa: E402

log = get_logger("entailment")

NLI_MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
ENTAILMENT_THRESHOLD = 0.5  # softmax probability above which we call a claim "entailed"

_model = None  # lazy-loaded singleton
_label_map = None


def get_model():
    global _model, _label_map
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(NLI_MODEL_NAME)
        # Read the label order from the model config rather than hardcoding it --
        # different NLI checkpoints don't always agree on label ordering.
        _label_map = _model.config.id2label
    return _model


def get_label_map() -> dict:
    get_model()  # ensures _label_map is populated
    return _label_map


def classify_claim(claim: str, evidence_texts: list, threshold: float = ENTAILMENT_THRESHOLD) -> dict:
    """Classify a claim against a pool of evidence texts.

    Scores the claim against every evidence text and keeps the
    highest-entailment-probability match -- i.e. "is this claim entailed by
    ANY piece of retrieved evidence", which is the right question for a
    RAG system pulling from multiple sources.

    Returns {"claim", "verdict", "score", "best_evidence"}. verdict is one
    of "entailed", "contradicted", "neutral" (or "no_evidence" if
    evidence_texts is empty).
    """
    if not evidence_texts:
        return {"claim": claim, "verdict": "no_evidence", "score": 0.0, "best_evidence": None}

    import torch

    model = get_model()
    pairs = [(evidence, claim) for evidence in evidence_texts]
    logits = model.predict(pairs)  # numpy array, shape (len(pairs), num_labels)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1)
    label_map = get_label_map()

    entailment_idx = next(i for i, name in label_map.items() if name.lower() == "entailment")
    best_i = int(torch.argmax(probs[:, entailment_idx]))
    best_probs = {label_map[i].lower(): float(probs[best_i][i]) for i in range(probs.shape[1])}
    best_label = max(best_probs, key=best_probs.get)
    entailment_score = best_probs["entailment"]

    verdict = "entailed" if entailment_score >= threshold else best_label
    return {
        "claim": claim,
        "verdict": verdict,
        "score": entailment_score,
        "best_evidence": evidence_texts[best_i],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("claim")
    parser.add_argument("evidence")
    args = parser.parse_args()

    result = classify_claim(args.claim, [args.evidence])
    log.info(f"verdict={result['verdict']} score={result['score']:.3f}")


if __name__ == "__main__":
    main()
