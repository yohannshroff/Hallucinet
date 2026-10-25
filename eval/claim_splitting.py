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
# Other ways the model cites, all of which would be noise to the NLI model:
# "(data/raw/The Great Mutiny - India 1857 by Christopher Hibbert.pdf#page=393)",
# "(Source [2], [4])", "(pp. 197-200)", and a trailing "Source: [1], [2]".
PDF_PATH_RE = re.compile(r"[\[(]?\s*(?:\S*/)?[^()\[\]/\n]*\.pdf(?:#page=\d+)?\s*[\])]?", re.IGNORECASE)
SOURCE_PAREN_RE = re.compile(r"\(\s*sources?\b[^()]*\)?", re.IGNORECASE)
SOURCE_LABEL_RE = re.compile(r"\bsources?\s*:[\s,;]*(?=$|[.\n])", re.IGNORECASE)
PAGES_RE = re.compile(r"\(?\bpp?\.\s*\d+(?:\s*[-–]\s*\d+)?\)?")
EMPTY_PARENS_RE = re.compile(r"\(\s*[,;]*\s*\)")
MARKDOWN_RE = re.compile(r"\*+|^\s*[-•]\s+", re.MULTILINE)
LEAD_IN_RE = re.compile(r"^\s*here(?:'s|’s| is) [^:.\n]{0,60}:\s*", re.IGNORECASE)

_nlp = None  # lazy-loaded singleton, same pattern as retrieval/entity_extraction.py


def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load(SPACY_MODEL_NAME)
    return _nlp


def strip_citations(text: str) -> str:
    """Remove citations, page references and markdown, keeping the sentence."""
    for pattern in (PDF_PATH_RE, CITATION_BRACKET_RE, SOURCE_PAREN_RE, PAGES_RE, SOURCE_LABEL_RE, EMPTY_PARENS_RE, MARKDOWN_RE, LEAD_IN_RE):
        text = pattern.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([.,;:!?)])", r"\1", text)  # "Outram ." -> "Outram."
    text = re.sub(r"([.,;:])(?:\s*[.,;:])+", r"\1", text)  # "Delhi,." -> "Delhi,"
    return re.sub(r"\s+", " ", text).strip(" ,;")


def split_into_claims(answer_text: str, min_words: int = MIN_CLAIM_WORDS) -> list:
    """Split `answer_text` into cleaned, non-trivial claim sentences."""
    # Clean before segmenting too: "Hibbert.pdf#page=393)." and bullet lines confuse the sentence splitter.
    # A line (e.g. a former bullet point) is a sentence even without a full stop.
    lines = [strip_citations(line) for line in answer_text.splitlines()]
    answer_text = "\n".join(line if not line or line[-1] in ".!?:\"'’”" else line + "." for line in lines)
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
