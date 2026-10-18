"""Stage 1 of building the knowledge graph from the book (no LLM, ~1 minute).

Reads the book text and, using spaCy and simple rules only:

1. discovers the book's entities: every person, place, organisation and
   event name (NER plus a rule for event names such as "Siege of Delhi")
   that recurs often enough to matter, with its type;
2. writes the facts that rules can state reliably on their own
   (candidates.jsonl): an event's location from its own name ("Siege of
   Delhi" located_in Delhi) and "<Person> of the <Regiment>" (member_of);

kg/type_entities.py then has the local LLM type and clean the names, and
kg/extract_facts.py has it read the sentences that mention them.

Outputs (data/kg_build/, gitignored -- they contain book sentences):
    mentions.json     every entity name with its mention count and an example sentence
    candidates.jsonl  rule-based facts

Usage:
    python kg/find_candidates.py
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from book_text import reconstruct_documents  # noqa: E402
from common import CANDIDATES_JSONL, CHUNKS_JSONL, MENTIONS_JSON, get_logger  # noqa: E402

log = get_logger("find_candidates")

NER_LABEL_TO_TYPE = {
    "PERSON": "Person",
    "GPE": "Location",
    "LOC": "Location",
    "FAC": "Location",
    "ORG": "Organization",
    "NORP": "Organization",
    "EVENT": "Event",
}

EVENT_HEADS = ["siege", "battle", "relief", "fall", "capture", "massacre", "mutiny", "storming", "recapture", "defence"]
EVENT_NAME_RE = re.compile(r"^(%s) (of|at) (.+)$" % "|".join(EVENT_HEADS), re.IGNORECASE)

# Sentences are only sent to the LLM if they contain an action word -- this keeps the
# (slow) LLM stage focused on sentences that can actually state one of our relations.
ACTION_RE = re.compile(
    r"\b(led|lead|leads|leading|command\w*|rul(?:e|ed|er|ers|ing)|reign\w*|govern\w*|besieg\w*|siege|"
    r"captur\w*|seiz\w*|stormed|storm|kill\w*|murder\w*|slain|slew|hang\w*|execut\w*|blown|shot|"
    r"fought|fight\w*|attack\w*|defeat\w*|oppos\w*|allied|alli\w*|join\w*|appoint\w*|succeed\w*|"
    r"disband\w*|died|death|dead|proclaim\w*|crown\w*|declar\w*|garrison\w*|reliev\w*|relief|"
    r"massacre\w*|surrender\w*|marched|advanc\w*|recapture\w*|mutin\w*|rebel\w*|revolt\w*|"
    r"member|brigade|regiment|force|army|troops|column|gener\w*)\b",
    re.IGNORECASE,
)

PERSON_TITLES = {
    "general", "brigadier", "brigadier-general", "colonel", "lieutenant", "lieutenant-colonel", "captain", "major",
    "sir", "lord", "lady", "mr", "mrs", "miss", "rani", "raja", "maharaja", "maharajah", "nawab", "begum", "king",
    "emperor", "sergeant", "subadar", "havildar", "jemadar", "cornet", "ensign", "dr", "reverend", "bishop", "viceroy",
}
ORG_WORDS = {
    "regiment", "infantry", "cavalry", "lancers", "fusiliers", "artillery", "company", "army", "rifles", "highlanders",
    "horse", "guards", "brigade", "division", "column", "force", "battalion", "sepoys", "troopers", "mutineers", "rebels",
    "council", "government", "parliament", "dynasty", "house", "light",
}
PLACE_WORDS = {
    "gate", "fort", "palace", "river", "ridge", "cantonment", "bridge", "camp", "residency", "mosque", "temple",
    "church", "hospital", "barracks", "bazaar", "city", "province", "district", "ghat", "canal", "road",
}
MIN_MENTIONS = 3  # an entity must be mentioned at least this often in the book to enter the lexicon
MAX_SENTENCE_WORDS = 90
MIN_SENTENCE_WORDS = 6
STOP_SURFACES = {
    "God", "Christ", "Mr", "Mrs", "Sir", "Lord", "General", "Colonel", "Captain", "Major", "Lieutenant", "Notes",
    "Chapter", "Part", "Index", "Plate", "Plates", "East", "West", "North", "South", "Sunday", "Monday", "Tuesday",
    "Wednesday", "Thursday", "Friday", "Saturday", "Christian", "Christians", "Muslim", "Muslims", "Hindu", "Hindus",
}

MIN_SURFACE_CHARS = 3


def clean_surface(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip(" ,.;:'\"()[]-")
    t = re.sub(r"^(the|The)\s+", "", t)
    t = re.sub(r"(’s|'s)$", "", t).strip()
    if t and t[0].islower():
        t = t[0].upper() + t[1:]  # "siege of Delhi" -> "Siege of Delhi"
    return t


def load_nlp():
    import spacy

    nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
    ruler.add_patterns(
        [
            {
                "label": "EVENT",
                "pattern": [
                    {"LOWER": {"IN": EVENT_HEADS}},
                    {"LOWER": {"IN": ["of", "at"]}},
                    {"IS_TITLE": True},
                    {"IS_TITLE": True, "OP": "?"},
                ],
            }
        ]
    )
    return nlp


def mention_of(span):
    """(surface, type) for an NER span, or None if it isn't a usable entity."""
    etype = NER_LABEL_TO_TYPE.get(span.label_)
    surface = clean_surface(span.text)
    if etype is None or len(surface) < MIN_SURFACE_CHARS or not re.search(r"[A-Za-z]", surface):
        return None
    if not surface[0].isupper() or surface.lower() in {"british", "indian", "english"}:
        return None
    return surface, etype


def infer_type(surface: str, type_votes: Counter) -> str:
    """spaCy's small model mislabels many names here (e.g. "Nana Sahib" as an
    organisation), so combine its majority vote with the words in the name."""
    words = [w.lower().strip(".,") for w in surface.split()]
    if EVENT_NAME_RE.match(surface):
        return "Event"
    if words and words[0] in PERSON_TITLES:
        return "Person"
    if any(w in ORG_WORDS for w in words) or re.match(r"^\d+(st|nd|rd|th)\b", surface):
        return "Organization"
    if words and words[-1] in PLACE_WORDS:
        return "Location"
    return type_votes.most_common(1)[0][0]


def build_lexicon(mention_counts) -> dict:
    """surface -> entity type, for names mentioned often enough to matter."""
    lexicon = {}
    for surface, info in mention_counts.items():
        if info["count"] < MIN_MENTIONS or surface in STOP_SURFACES:
            continue
        if re.search(r"\d", surface) and not re.match(r"^\d+(st|nd|rd|th)\b", surface):
            continue  # footnote debris like "frightened';33"
        if not re.fullmatch(r"[A-Za-z0-9'’\- .]+", surface):
            continue
        lexicon[surface] = infer_type(surface, info["types"])
    return lexicon


def compile_lexicon_regex(lexicon: dict):
    surfaces = sorted(lexicon, key=len, reverse=True)  # longest first => "Henry Havelock" beats "Havelock"
    return re.compile(r"(?<![\w'’])(" + "|".join(re.escape(x) for x in surfaces) + r")(?![\w])")


def person_of_org_facts(doc, ents_by_root):
    """"<Person> of the <Regiment>" => member_of."""
    for root_i, (surface, etype) in ents_by_root.items():
        if etype != "Person":
            continue
        for child in doc[root_i].children:
            if child.dep_ == "prep" and child.lower_ == "of":
                for pobj in child.children:
                    hit = ents_by_root.get(pobj.i)
                    if pobj.dep_ == "pobj" and hit and hit[1] == "Organization" and hit[0] != surface:
                        yield surface, hit[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_JSONL)
    parser.add_argument("--candidates_out", type=Path, default=CANDIDATES_JSONL)
    parser.add_argument("--mentions_out", type=Path, default=MENTIONS_JSON)
    args = parser.parse_args()

    docs = reconstruct_documents(args.chunks)
    nlp = load_nlp()
    log.info(f"parsing {len(docs)} documents with spaCy ...")

    mention_counts = defaultdict(lambda: {"count": 0, "types": Counter(), "example": ""})
    rule_facts = []
    seen_rule = set()
    doc_sentences = []  # (doc meta, [sentence text, ...])

    def add_rule_fact(meta, sentence, subj, subj_type, relation, obj, obj_type, rule):
        key = (meta["doc_id"], subj, relation, obj)
        if key in seen_rule:
            return
        seen_rule.add(key)
        rule_facts.append(
            {
                "doc_id": meta["doc_id"], "title": meta["title"], "source_url": meta["source_url"],
                "sentence": sentence, "subject": subj, "subject_type": subj_type,
                "relation": relation, "object": obj, "object_type": obj_type, "rule": rule,
            }
        )

    for doc, meta in zip(nlp.pipe((d["text"] for d in docs), batch_size=4), docs):
        sentences, sent_of_token = [], {}
        for si, sent in enumerate(doc.sents):
            sentences.append(re.sub(r"\s+", " ", sent.text).strip())
            for t in sent:
                sent_of_token[t.i] = si

        ents_in_sentence = defaultdict(dict)  # sentence index -> {root token index: (surface, type)}
        for ent in doc.ents:
            m = mention_of(ent)
            if m is None:
                continue
            surface, etype = m
            mention_counts[surface]["count"] += 1
            mention_counts[surface]["types"][etype] += 1
            si = sent_of_token[ent.root.i]
            if not mention_counts[surface]["example"] and 8 <= len(sentences[si].split()) <= 45:
                mention_counts[surface]["example"] = sentences[si]
            ents_in_sentence[si][ent.root.i] = (surface, etype)

            name_match = EVENT_NAME_RE.match(surface)
            if etype == "Event" and name_match:  # "Siege of Delhi" is located_in Delhi
                loc = clean_surface(name_match.group(3))
                if loc:
                    add_rule_fact(meta, sentences[si], surface, "Event", "located_in", loc, "Location", "event_name")

        for si, ents_by_root in ents_in_sentence.items():
            for person, org in person_of_org_facts(doc, ents_by_root):
                add_rule_fact(meta, sentences[si], person, "Person", "member_of", org, "Organization", "person_of_org")

        doc_sentences.append((meta, sentences))

    lexicon = build_lexicon(mention_counts)
    log.info(f"{len(mention_counts)} distinct surface forms -> {len(lexicon)} names mentioned >= {MIN_MENTIONS} times")

    args.candidates_out.parent.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(rule_facts):
        c["cand_id"] = f"c{i:05d}"
    with open(args.candidates_out, "w", encoding="utf-8") as f:
        for c in rule_facts:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    args.mentions_out.write_text(
        json.dumps({k: {"count": v["count"], "types": dict(v["types"]), "example": v["example"]} for k, v in mention_counts.items()}, ensure_ascii=False),
        encoding="utf-8",
    )

    log.info(f"{len(rule_facts)} rule-based facts")
    log.info(f"wrote {args.candidates_out.name} and {args.mentions_out.name} in {args.candidates_out.parent}")


if __name__ == "__main__":
    main()
