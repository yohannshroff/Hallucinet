"""Stage 3 of building the knowledge graph: extract typed facts from the
book's sentence grammar and check each one (about 1-2 minutes, no LLM).

Why grammar rather than the LLM: two attempts to have the 4B local model
state relations directly (free-form from 250-word windows, then
sentence-by-sentence with a constrained reply) produced almost entirely
wrong or backwards facts (e.g. "Mangal Pande executed Barrackpore"). What
the small model does reliably is name typing (kg/type_entities.py), so it is
used for that, and relations come from grammar instead:

- spaCy's dependency parse finds "<subject> <verb> <object>" where the verb
  maps onto a controlled relation (commanded, besieged, captured, killed,
  fought_against, ...), handling passives ("Delhi was captured by
  Nicholson") and conjunctions ("Wilson and Nicholson besieged Delhi"). The
  subject and object must be entities from the typed entity list.
- appositions such as "Henry Lawrence, the Chief Commissioner of Oudh" and
  "Bahadur Shah, the King of Delhi" (title-of patterns).

Every extracted fact is then turned into a sentence ("Hugh Rose besieged
Jhansi.") and scored by the project's independent NLI model against its
source sentence -- the same check the trust score uses. The score is stored;
kg/consolidate_triples.py applies a threshold, so it can be re-tuned
without re-running this stage.

The same pass counts which pairs of entities are mentioned in the same
sentence (co-occurrence), which becomes the graph's `associated_with` edges.

Outputs (names, relations, scores and source pages only -- no book text, so
they are safe to commit):
    data/kg_build/extracted.jsonl     grammar-extracted facts + NLI score
    data/kg_build/cooccurrence.jsonl  entity pairs co-mentioned in sentences

Usage:
    python kg/extract_facts.py
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from book_text import reconstruct_documents  # noqa: E402
from common import CHUNKS_JSONL, COOCCURRENCE_JSONL, ENTITY_TYPES_JSON, EXTRACTED_JSONL, get_logger  # noqa: E402

log = get_logger("extract_facts")

LLM_TYPE_TO_SCHEMA_TYPE = {"PERSON": "Person", "PLACE": "Location", "ORG": "Organization", "EVENT": "Event"}
MAX_ENTITIES_FOR_COOCCURRENCE = 6  # longer lists (indexes, tables of contents) say nothing about a pair
MIN_SENTENCE_WORDS = 5
MAX_SENTENCE_WORDS = 90
BIBLIOGRAPHY_RE = re.compile(r"(\bibid\b|\bop\. ?cit|\bpp?\. ?\d|\bVol\.|\(\w[\w ,.]*,? ?\d{4}\)|\b\d{4}\)\.?$|^—|^See\b|^Quoted)")

# verb lemma -> (relation, direction, prepositions that can introduce the object)
#   "fwd": "A <verb> B"  => (A, relation, B)
#   "rev": "A <verb> B"  => (B, relation, A)   e.g. "A appointed B" => B appointed_by A
VERB_RULES = {
    "lead": ("led", "fwd", ()),
    "command": ("commanded", "fwd", ()),
    "rule": ("ruled", "fwd", ()),
    "govern": ("ruled", "fwd", ()),
    "reign": ("ruled", "fwd", ("over",)),
    "besiege": ("besieged", "fwd", ()),
    "capture": ("captured", "fwd", ()),
    "recapture": ("captured", "fwd", ()),
    "seize": ("captured", "fwd", ()),
    "kill": ("killed", "fwd", ()),
    "murder": ("killed", "fwd", ()),
    "hang": ("executed", "fwd", ()),
    "execute": ("executed", "fwd", ()),
    "fight": ("fought_against", "fwd", ("against",)),
    "attack": ("fought_against", "fwd", ()),
    "defeat": ("fought_against", "fwd", ()),
    "oppose": ("fought_against", "fwd", ()),
    "ally": ("allied_with", "fwd", ("with",)),
    "appoint": ("appointed_by", "rev", ()),
    "succeed": ("succeeded_by", "rev", ()),
    "disband": ("disbanded_by", "rev", ()),
}

# "<Person>, the <title> of <place/group>"
TITLE_RELATION = {
    "commander-in-chief": "commanded", "commander": "commanded", "leader": "led", "ruler": "ruled",
    "king": "ruled", "emperor": "ruled", "raja": "ruled", "rajah": "ruled", "nawab": "ruled", "begum": "ruled",
    "queen": "ruled", "maharaja": "ruled", "maharajah": "ruled", "governor": "ruled", "governor-general": "ruled",
    "chief commissioner": "ruled", "commissioner": "ruled",
}


def sentence_key(sentence: str) -> str:
    return hashlib.sha1(sentence.encode("utf-8")).hexdigest()[:12]


# Generic words and name fragments the typing step let through; they would create junk
# facts ("King ruled Delhi"). Group names that really act as a party (Sikhs, Gurkhas) stay.
EXCLUDED_NAMES = {
    "King", "Brigadier", "Major-General", "Highlanders", "Indians", "European", "Europeans", "Sahibs", "Eurasian",
    "Government", "Army", "Engineers", "Magistrate", "Colin", "Henry", "John", "James", "William", "George", "Charles",
    "Mutineers", "Rebels", "Cheer", "Mohurram", "Diwan", "Native Infantry", "Magazine", "Ridge", "Veranda",
    "London", "Edinburgh",  # almost always publication places in footnotes
    "Queen", "Rajah", "Nawab", "Brahmin", "Brahmins", "Maulvi", "Treasury", "Nana", "Sepoys", "Native",
}


def load_lexicon(path: Path) -> dict:
    types = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: LLM_TYPE_TO_SCHEMA_TYPE[t]
        for name, t in types.items()
        if t in LLM_TYPE_TO_SCHEMA_TYPE and name not in EXCLUDED_NAMES
    }


def lexicon_regex(names):
    names = sorted(names, key=len, reverse=True)  # longest first => "Henry Lawrence" beats "Lawrence"
    return re.compile(r"(?<![\w'’])(" + "|".join(re.escape(x) for x in names) + r")(?![\w])")


def conjuncts_of(token):
    return [token] + list(token.conjuncts)


def verb_facts(sent, ents_by_root):
    """Yield (subject, relation, object, rule) from verb frames in one sentence."""

    def mentions(tokens):
        return [ents_by_root[t.i] for t in tokens if t.i in ents_by_root]

    for tok in sent:
        if tok.pos_ != "VERB" or tok.lemma_.lower() not in VERB_RULES:
            continue
        relation, direction, preps = VERB_RULES[tok.lemma_.lower()]

        subjects, passive_patients, objects, agents = [], [], [], []
        for child in tok.children:
            if child.dep_ == "nsubj":
                subjects += mentions(conjuncts_of(child))
            elif child.dep_ == "nsubjpass":
                passive_patients += mentions(conjuncts_of(child))
            elif child.dep_ == "dobj":
                objects += mentions(conjuncts_of(child))
            elif child.dep_ == "prep" and child.lower_ in preps:
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        objects += mentions(conjuncts_of(pobj))
            elif child.dep_ == "agent":
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        agents += mentions(conjuncts_of(pobj))

        pairs = [(s, o) for s in subjects for o in objects]  # active voice
        pairs += [(a, p) for a in agents for p in passive_patients]  # passive: "Delhi was captured by Nicholson"
        for actor, patient in pairs:
            subj, obj = (actor, patient) if direction == "fwd" else (patient, actor)
            if subj != obj:
                yield subj, relation, obj, "verb:" + tok.lemma_.lower()


def appositive_facts(sentence: str, person_re, other_re):
    """"Henry Lawrence, the Chief Commissioner of Oudh" -> (Henry Lawrence, ruled, Oudh)."""
    titles = "|".join(re.escape(t) for t in sorted(TITLE_RELATION, key=len, reverse=True))
    pattern = re.compile(
        rf"(?P<p>{person_re.pattern}),? (?:the |a |an |then )?(?:[\w'’-]+ ){{0,2}}?(?P<t>(?i:{titles}))(?: of| in) (?:the )?(?P<o>{other_re.pattern})"
    )
    for m in pattern.finditer(sentence):
        yield m.group("p"), TITLE_RELATION[m.group("t").lower()], m.group("o"), "appositive:" + m.group("t").lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_JSONL)
    parser.add_argument("--types", type=Path, default=ENTITY_TYPES_JSON)
    parser.add_argument("--out", type=Path, default=EXTRACTED_JSONL)
    args = parser.parse_args()

    import spacy
    from entailment import _softmax, get_label_map, get_model
    from trust_score import fact_to_sentence

    lexicon = load_lexicon(args.types)
    all_re = lexicon_regex(lexicon)
    people_re = lexicon_regex([n for n, t in lexicon.items() if t == "Person"])
    other_re = lexicon_regex([n for n, t in lexicon.items() if t != "Person"])
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2_000_000

    docs = reconstruct_documents(args.chunks)
    log.info(f"{len(lexicon)} typed entities; parsing {len(docs)} documents ...")

    records = []  # [(meta, sentence, facts)]
    pair_sentences = Counter()  # (a, b) -> number of sentences mentioning both
    pair_docs = {}  # (a, b) -> Counter of doc ids
    for doc, meta in zip(nlp.pipe((d["text"] for d in docs), batch_size=4), docs):
        by_sentence = {}  # sentence start index -> (sent span, {root token index: surface})
        for m in all_re.finditer(doc.text):
            span = doc.char_span(m.start(1), m.end(1), alignment_mode="expand")
            if span is None:
                continue
            sent = span.sent
            by_sentence.setdefault(sent.start, (sent, {}))[1][span.root.i] = m.group(1)

        for sent, roots in by_sentence.values():
            sentence = re.sub(r"\s+", " ", sent.text).strip()
            n_words = len(sentence.split())
            if len(roots) < 2 or n_words < MIN_SENTENCE_WORDS or n_words > MAX_SENTENCE_WORDS or BIBLIOGRAPHY_RE.search(sentence):
                continue
            ents_by_root = {i: surface for i, surface in roots.items() if surface in lexicon}
            names = sorted(set(ents_by_root.values()))
            if 2 <= len(names) <= MAX_ENTITIES_FOR_COOCCURRENCE:
                for i, a in enumerate(names):
                    for b in names[i + 1 :]:
                        pair_sentences[(a, b)] += 1
                        pair_docs.setdefault((a, b), Counter())[meta["doc_id"]] += 1
            found = list(verb_facts(sent, ents_by_root)) + list(appositive_facts(sentence, people_re, other_re))
            facts, seen = [], set()
            for subj, rel, obj, rule in found:
                if subj in lexicon and obj in lexicon and (subj, rel, obj) not in seen:
                    seen.add((subj, rel, obj))
                    facts.append(
                        {
                            "subject": subj, "subject_type": lexicon[subj], "relation": rel,
                            "object": obj, "object_type": lexicon[obj], "rule": rule,
                        }
                    )
            if facts:
                records.append((meta, sentence, facts))

    n_facts = sum(len(f) for _, _, f in records)
    log.info(f"{n_facts} candidate facts from {len(records)} sentences; checking each against its sentence with the NLI model ...")

    model = get_model()
    label_map = get_label_map()
    entail_idx = next(i for i, name in label_map.items() if name.lower() == "entailment")
    pairs = [(sentence, fact_to_sentence(f)) for _, sentence, facts in records for f in facts]
    probs = _softmax(model.predict(pairs, batch_size=32, show_progress_bar=False)) if pairs else []
    scores = iter(float(p[entail_idx]) for p in probs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept_by_relation = Counter()
    with open(args.out, "w", encoding="utf-8") as f:
        for meta, sentence, facts in records:
            for fact in facts:
                fact["entail"] = round(next(scores), 4)
                kept_by_relation[fact["relation"]] += fact["entail"] >= 0.5
            f.write(
                json.dumps(
                    {
                        "key": sentence_key(sentence), "doc_id": meta["doc_id"], "title": meta["title"],
                        "source_url": meta["source_url"], "facts": facts,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    doc_meta = {d["doc_id"]: d for d in docs}
    with open(COOCCURRENCE_JSONL, "w", encoding="utf-8") as f:
        for (a, b), n in pair_sentences.most_common():
            doc_id = pair_docs[(a, b)].most_common(1)[0][0]
            f.write(
                json.dumps(
                    {"a": a, "b": b, "count": n, "doc_id": doc_id, "title": doc_meta[doc_id]["title"], "source_url": doc_meta[doc_id]["source_url"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    log.info(f"{len(pair_sentences)} entity pairs co-mentioned; {sum(1 for v in pair_sentences.values() if v >= 3)} of them in 3+ sentences")
    log.info(f"facts with NLI entailment >= 0.5, by relation: {dict(kept_by_relation.most_common())}")
    log.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
