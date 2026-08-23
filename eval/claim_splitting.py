"""Split an LLM-generated answer into individual claims for entailment
checking.

Design: a "claim" is a sentence, using spaCy's sentence segmentation. True
atomic-fact decomposition (breaking compound sentences into separate
subject-predicate-object claims) would catch more, but adds a lot of
complexity/failure surface for a 2-month project -- sentence-level
splitting is the simple, defensible choice, matching the master plan's
"keep it simple" guidance. See docs/part6_notes.md.

Citation markers the model tends to emit (e.g. "[1]", "[1, 2]",
"[https://en.wikipedia.org/wiki/X]") are stripped before scoring since
they're not part of the factual content and would just be noise to the NLI
model.

Usage:
    python eval/claim_splitting.py "Rani Lakshmibai ruled Jhansi [1]. She fought Hugh Rose [2]."
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import get_logger  # noqa: E402

log = get_logger("claim_splitting")

SPACY_MODEL_NAME = "en_core_web_sm"
MIN_CLAIM_WORDS = 4  # drop fragments like "References:" left over after stripping citations

CITATION_BRACKET_RE = re.compile(r"\[[^\]]*\]")

_nlp = None  # lazy-loaded singleton, same pattern as retrieval/entity_extraction.py


def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load(SPACY_MODEL_NAME)
    return _nlp


def strip_citations(text: str) -> str:
    text = CITATION_BRACKET_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def split_into_claims(answer_text: str, min_words: int = MIN_CLAIM_WORDS) -> list:
    """Split `answer_text` into cleaned, non-trivial claim sentences."""
    doc = get_nlp()(answer_text)
    claims = []
    for sent in doc.sents:
        cleaned = strip_citations(sent.text)
        if len(cleaned.split()) >= min_words:
            claims.append(cleaned)
    return claims


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answer", help="Answer text to split into claims")
    args = parser.parse_args()

    claims = split_into_claims(args.answer)
    log.info(f"{len(claims)} claim(s):")
    for i, claim in enumerate(claims, start=1):
        print(f"  [{i}] {claim}")


if __name__ == "__main__":
    main()
