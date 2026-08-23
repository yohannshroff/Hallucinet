"""Trust score: the fraction of an answer's claims that are entailed by the
retrieved evidence (graph facts + vector chunks) it was generated from.

This is the project's core anti-hallucination metric. It deliberately does
NOT ask the generating LLM to grade itself (see the master plan's risk
register on circularity) -- entailment is checked by an independent,
pretrained NLI model (eval/entailment.py).

Usage:
    python eval/trust_score.py "Who led the resistance at Jhansi?"
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generation"))
from common import ENTITIES_CSV, get_logger  # noqa: E402

from claim_splitting import split_into_claims  # noqa: E402
from entailment import classify_claim  # noqa: E402
from prompt_template import REFUSAL_PHRASE  # noqa: E402

log = get_logger("trust_score")

# Turns a graph fact's (subject, relation, object) into a natural-language
# sentence so the NLI model -- trained on prose, not "A --rel--> B" arrow
# notation -- can actually compare it against a claim. Kept in sync with
# RELATION_VOCAB in scripts/common.py; see tests/test_trust_score.py.
RELATION_TEMPLATES = {
    "led": "{subject} led {object}.",
    "ruled": "{subject} ruled {object}.",
    "fought_against": "{subject} fought against {object}.",
    "allied_with": "{subject} was allied with {object}.",
    "killed": "{subject} killed {object}.",
    "executed": "{subject} executed {object}.",
    "captured": "{subject} captured {object}.",
    "besieged": "{subject} besieged {object}.",
    "located_in": "{subject} is located in {object}.",
    "part_of": "{subject} was part of {object}.",
    "caused_by": "{subject} was caused by {object}.",
    "triggered": "{subject} triggered {object}.",
    "succeeded_by": "{subject} was succeeded by {object}.",
    "appointed_by": "{subject} was appointed by {object}.",
    "commanded": "{subject} commanded {object}.",
    "declared_emperor_by": "{subject} was declared emperor by {object}.",
    "died_during": "{subject} died during {object}.",
    "member_of": "{subject} was a member of {object}.",
    "disbanded_by": "{subject} was disbanded by {object}.",
}


def fact_to_sentence(fact: dict) -> str:
    template = RELATION_TEMPLATES.get(fact["relation"])
    if template is None:
        # Fallback for a relation not yet in RELATION_TEMPLATES -- ugly but
        # functional, and test_trust_score.py flags the gap so it gets a
        # proper template added.
        return f"{fact['subject']} {fact['relation'].replace('_', ' ')} {fact['object']}."
    return template.format(subject=fact["subject"], object=fact["object"])


def evidence_texts_from_bundle(bundle: dict) -> list:
    """Flatten a Part 4 retrieval bundle into a list of single-sentence
    evidence strings: one per graph fact, plus each vector chunk's text
    broken into individual sentences.

    Sentence-level evidence matters here, not just chunk-level: this
    cross-encoder NLI model (like most) is trained on single-sentence
    premise/hypothesis pairs, and empirically scores true, well-supported
    claims far too low when the premise is a whole ~180-word multi-topic
    chunk instead of the one sentence that actually supports the claim
    (see docs/part6_notes.md for the before/after numbers).
    """
    texts = [fact_to_sentence(fact) for fact in bundle.get("graph_facts", [])]
    for chunk in bundle.get("vector_chunks", []):
        texts += split_into_claims(chunk["text"])  # just sentence segmentation here, not "claims" per se
    return texts


def compute_trust_score(answer_text: str, bundle: dict) -> dict:
    """Split `answer_text` into claims and check each against the evidence
    in `bundle`. Returns {"trust_score", "claims", "n_claims", "n_entailed"}.

    trust_score is None (not 0) when the answer made no checkable factual
    claims at all -- e.g. a correct "I don't have enough information"
    refusal has nothing to verify, and scoring that as 0% would wrongly
    penalize exactly the behavior the prompt template is designed to
    produce.
    """
    evidence_texts = evidence_texts_from_bundle(bundle)
    claims = split_into_claims(answer_text)
    # The model's own "I don't know" refusal is meta-commentary about the
    # system, not a factual claim about history -- scoring it against
    # evidence would wrongly report 0% trust for exactly the behavior the
    # prompt template is designed to produce (see docs/part6_notes.md).
    claims = [c for c in claims if c.strip().rstrip(".") != REFUSAL_PHRASE.rstrip(".")]

    results = [classify_claim(claim, evidence_texts) for claim in claims]
    n_entailed = sum(1 for r in results if r["verdict"] == "entailed")

    trust_score = (n_entailed / len(results)) if results else None

    return {
        "trust_score": trust_score,
        "claims": results,
        "n_claims": len(results),
        "n_entailed": n_entailed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--entities", type=Path, default=ENTITIES_CSV)
    args = parser.parse_args()

    from generate_answer import answer_question

    entities_df = pd.read_csv(args.entities, keep_default_na=False)
    result = answer_question(args.question, entities_df)

    scoring = compute_trust_score(result["answer"], result["bundle"])

    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}\n")

    if scoring["trust_score"] is None:
        print("Trust score: N/A (no checkable claims -- answer made no factual assertions)")
    else:
        print(f"Trust score: {scoring['trust_score']:.0%} ({scoring['n_entailed']}/{scoring['n_claims']} claims entailed)")

    for r in scoring["claims"]:
        print(f"  [{r['verdict']}, score={r['score']:.2f}] {r['claim']}")
        if r["best_evidence"]:
            print(f"      best evidence: {r['best_evidence'][:150]}")


if __name__ == "__main__":
    main()
