"""Stage 3b of building the knowledge graph: draft typed facts for a person
to review, then export the reviewed ones (drafting ~1-2 hours, needs Ollama;
export takes seconds).

Why a review step: docs/kg_build.md records that fully automatic relation
extraction did not work on a laptop -- the 4B model alone was mostly wrong,
and grammar rules alone found only 12 facts. Here both only *propose*
facts; a person accepts or rejects each one, so every typed fact in the
graph has been checked against the book.

    python kg/draft_facts.py            # draft  -> data/kg_build/review_facts.csv
    (open the CSV, fill in the `decision` column: accept / reject / fix)
    python kg/draft_facts.py export     # review -> data/kg_seed/verified_facts.csv

What `draft` does:
1. Picks entity pairs that the book mentions together in 2+ sentences (the
   same pairs that become `associated_with` edges), and for each takes up to
   3 sentences naming both.
2. Asks the local LLM, one sentence at a time, which controlled relation (or
   "none") the sentence states between each pair. The reply is constrained to
   a JSON schema whose names and relations are the only allowed values.
3. Adds the grammar-rule facts of kg/extract_facts.py for the same sentences.
4. Drops facts whose entity types make the relation impossible (TYPE_RULES),
   scores the rest with the NLI model against their sentence (the score is a
   reading aid, not a filter), and writes one row per fact, best-supported
   first. Decisions already made in an earlier review file are kept.

Fairness: facts are drafted pair by pair from the whole book, never from the
evaluation questions. While reviewing, judge each fact by its sentence only;
do not add facts because a test question needs them.

Review columns: `decision` is accept / reject / fix. For `fix`, fill in
fixed_subject / fixed_relation / fixed_object (blank = keep the drafted value,
so swapping the direction means filling both names). A fixed fact is
exported as the corrected fact (accept) plus the drafted one (reject).
Rejected facts are exported too: kg/consolidate_triples.py removes them even
when the grammar rules find them again.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generation"))
from book_text import reconstruct_documents  # noqa: E402
from common import (  # noqa: E402
    CHUNKS_JSONL,
    COOCCURRENCE_JSONL,
    DRAFT_CACHE_JSON,
    ENTITY_TYPES_JSON,
    OLLAMA_MODEL,
    RELATION_VOCAB,
    REVIEW_FACTS_CSV,
    VERIFIED_FACTS_CSV,
    get_logger,
)
from consolidate_triples import TYPE_RULES, pages_of  # noqa: E402
from extract_facts import entity_sentences, grammar_facts, lexicon_regex, load_lexicon, sentence_key  # noqa: E402

log = get_logger("draft_facts")

TYPED_RELATIONS = sorted(RELATION_VOCAB - {"associated_with"})
NONE = "none"
SENTENCES_PER_PAIR = 3
SAVE_EVERY = 10  # sentences between cache saves

REVIEW_COLUMNS = [
    "review_id", "subject", "relation", "object", "pages", "sentence", "nli", "n_sentences", "drafted_by",
    "decision", "fixed_subject", "fixed_relation", "fixed_object", "note", "source", "title",
]
REVIEWER_COLUMNS = ["decision", "fixed_subject", "fixed_relation", "fixed_object", "note"]
VERIFIED_COLUMNS = ["subject", "relation", "object", "source", "title", "decision", "note"]

SYSTEM_PROMPT = """You read one sentence from a history book about the Indian Mutiny of 1857 and a numbered list of name pairs from it.
For each pair, say which relation the sentence itself states between the two names, choosing only from this list:
led, commanded (a person in command of a force or an operation), ruled (a sovereign over a place or people),
administered (an official in charge of a province or body), fought_against, allied_with, killed, executed, captured,
besieged, located_in, part_of, caused_by, triggered, succeeded_by, appointed_by, declared_emperor_by, died_during,
member_of, disbanded_by.
Use "none" when the sentence does not clearly state one of these relations between the two -- being mentioned
together is not enough, and do not use what you know from elsewhere.
Subject and object must be the two names of the pair; get the direction right ("Delhi was captured by Nicholson"
gives subject Nicholson, relation captured, object Delhi).
Evidence: copy the few words of the sentence that state the relation."""


def reply_schema(names: list) -> dict:
    """JSON schema for one sentence: subject/object limited to the sentence's names,
    relation limited to the controlled vocabulary (or none)."""
    return {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pair": {"type": "integer"},
                        "subject": {"type": "string", "enum": names},
                        "relation": {"type": "string", "enum": TYPED_RELATIONS + [NONE]},
                        "object": {"type": "string", "enum": names},
                        "evidence": {"type": "string"},
                    },
                    "required": ["pair", "subject", "relation", "object", "evidence"],
                },
            }
        },
        "required": ["facts"],
    }


def parse_reply(reply: dict, pairs: list) -> list:
    """Keep only well-formed facts: a known pair number, its two names (either direction),
    and a typed relation."""
    facts = []
    for item in reply.get("facts", []):
        n = item.get("pair")
        if not isinstance(n, int) or not 1 <= n <= len(pairs):
            continue
        if item.get("relation") not in TYPED_RELATIONS:
            continue
        if {item.get("subject"), item.get("object")} != set(pairs[n - 1]):
            continue
        facts.append({"subject": item["subject"], "relation": item["relation"], "object": item["object"], "evidence": item.get("evidence", "")})
    return facts


def draft_sentence(sentence: str, pairs: list, chat_fn) -> list:
    """Ask the LLM which relation the sentence states for each (a, b) pair."""
    names = sorted({x for pair in pairs for x in pair})
    lines = [f"{i}. {a} / {b}" for i, (a, b) in enumerate(pairs, start=1)]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Sentence: "{sentence}"\n\nPairs:\n' + "\n".join(lines)},
    ]
    reply = json.loads(chat_fn(messages, temperature=0.0, num_predict=400, format=reply_schema(names)))
    return parse_reply(reply, pairs)


def type_ok(fact: dict, lexicon: dict) -> bool:
    allowed_s, allowed_o = TYPE_RULES[fact["relation"]]
    return lexicon[fact["subject"]] in allowed_s and lexicon[fact["object"]] in allowed_o


def chosen_pairs(path: Path, lexicon: dict, min_cooc: int, max_pairs: int = None) -> set:
    pairs = []
    for line in open(path, encoding="utf-8"):
        c = json.loads(line)
        if c["count"] >= min_cooc and c["a"] in lexicon and c["b"] in lexicon:
            pairs.append((c["count"], tuple(sorted((c["a"], c["b"])))))
    pairs.sort(reverse=True)
    return {p for _, p in pairs[:max_pairs]}


def previous_decisions(path: Path) -> dict:
    """(subject, relation, object) -> reviewer columns from an earlier review file."""
    if not path.exists():
        return {}
    out = {}
    for row in csv.DictReader(open(path, encoding="utf-8")):
        if any(row.get(c) for c in REVIEWER_COLUMNS):
            out[(row["subject"], row["relation"], row["object"])] = {c: row.get(c, "") for c in REVIEWER_COLUMNS}
    return out


def cmd_draft(args):
    from ollama_client import chat, is_reachable

    if not args.no_llm and not is_reachable():
        log.info("Ollama not reachable -- start it first (`ollama serve`), or pass --no_llm for grammar facts only")
        sys.exit(1)

    lexicon = load_lexicon(args.types)
    people_re = lexicon_regex([n for n, t in lexicon.items() if t == "Person"])
    other_re = lexicon_regex([n for n, t in lexicon.items() if t != "Person"])
    wanted = chosen_pairs(COOCCURRENCE_JSONL, lexicon, args.min_cooc, args.max_pairs)
    log.info(f"{len(wanted)} entity pairs co-mentioned in >= {args.min_cooc} sentences")

    cache_key_prefix = OLLAMA_MODEL + ":"
    cache = json.loads(DRAFT_CACHE_JSON.read_text(encoding="utf-8")) if DRAFT_CACHE_JSON.exists() else {}

    drafts = []  # (fact, sentence, meta, drafted_by)
    seen_per_pair = defaultdict(int)
    n_new = 0
    docs = reconstruct_documents(args.chunks)
    for meta, sent, sentence, ents_by_root in entity_sentences(docs, lexicon):
        for f in grammar_facts(sent, sentence, ents_by_root, lexicon, people_re, other_re):
            drafts.append((f, sentence, meta, "grammar"))
        if args.no_llm:
            continue

        names = sorted(set(ents_by_root.values()))
        pairs = [
            (a, b) for i, a in enumerate(names) for b in names[i + 1 :]
            if (a, b) in wanted and seen_per_pair[(a, b)] < SENTENCES_PER_PAIR
        ]
        if not pairs:
            continue
        for p in pairs:
            seen_per_pair[p] += 1
        key = cache_key_prefix + sentence_key(sentence) + ":" + "|".join(f"{a}/{b}" for a, b in pairs)
        if key not in cache:
            if args.limit is not None and n_new >= args.limit:
                continue
            try:
                cache[key] = draft_sentence(sentence, pairs, chat)
            except Exception as exc:  # noqa: BLE001 - retried on the next run
                log.info(f"  sentence {sentence_key(sentence)}: FAILED ({exc})")
                continue
            n_new += 1
            if n_new % SAVE_EVERY == 0:
                DRAFT_CACHE_JSON.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
                log.info(f"  {n_new} sentences sent to the LLM")
        for f in cache[key]:
            drafts.append(({**f, "subject_type": lexicon[f["subject"]], "object_type": lexicon[f["object"]]}, sentence, meta, "llm"))
    DRAFT_CACHE_JSON.parent.mkdir(parents=True, exist_ok=True)
    DRAFT_CACHE_JSON.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info(f"{n_new} new LLM calls; {len(drafts)} drafted facts before checks")

    drafts = [d for d in drafts if type_ok(d[0], lexicon)]
    log.info(f"{len(drafts)} drafted facts with possible entity types; scoring each against its sentence ...")
    from entailment import _softmax, get_label_map, get_model
    from trust_score import fact_to_sentence

    model = get_model()
    entail_idx = next(i for i, name in get_label_map().items() if name.lower() == "entailment")
    pairs = [(sentence, fact_to_sentence(f)) for f, sentence, _, _ in drafts]
    probs = _softmax(model.predict(pairs, batch_size=32, show_progress_bar=False)) if pairs else []

    # one row per (subject, relation, object): the best-scoring sentence stands for it
    rows = {}
    for (f, sentence, meta, by), p in zip(drafts, probs):
        triple = (f["subject"], f["relation"], f["object"])
        nli = round(float(p[entail_idx]), 3)
        row = rows.get(triple)
        if row is None:
            row = rows[triple] = {"n_sentences": 0, "drafted_by": set(), "nli": -1.0}
        row["n_sentences"] += 1
        row["drafted_by"].add(by)
        if nli > row["nli"]:
            row.update(nli=nli, sentence=sentence, source=meta["source_url"], title=meta["title"])

    kept_decisions = previous_decisions(REVIEW_FACTS_CSV)
    ordered = sorted(rows.items(), key=lambda kv: (-kv[1]["nli"], kv[0]))
    REVIEW_FACTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_FACTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        for i, ((s, rel, o), r) in enumerate(ordered, start=1):
            w.writerow(
                {
                    "review_id": f"R{i:04d}", "subject": s, "relation": rel, "object": o, "pages": pages_of(r["title"]),
                    "sentence": r["sentence"], "nli": r["nli"], "n_sentences": r["n_sentences"],
                    "drafted_by": "+".join(sorted(r["drafted_by"])), "source": r["source"], "title": r["title"],
                    **kept_decisions.get((s, rel, o), {c: "" for c in REVIEWER_COLUMNS}),
                }
            )
    log.info(f"wrote {len(ordered)} facts to review ({len(kept_decisions)} earlier decisions kept) -> {REVIEW_FACTS_CSV}")


def export_rows(review_rows: list) -> tuple:
    """Turn reviewed rows into verified-fact rows. Returns (rows, counts, problems)."""
    out, counts, problems = {}, defaultdict(int), []

    def add(s, rel, o, decision, row):
        out[(s, rel, o)] = {
            "subject": s, "relation": rel, "object": o, "source": row["source"], "title": row["title"],
            "decision": decision, "note": row.get("note", ""),
        }

    for row in review_rows:
        decision = row.get("decision", "").strip().lower()
        counts[decision or "unreviewed"] += 1
        if decision in ("accept", "reject"):
            add(row["subject"], row["relation"], row["object"], decision, row)
        elif decision == "fix":
            s = row.get("fixed_subject", "").strip() or row["subject"]
            rel = row.get("fixed_relation", "").strip() or row["relation"]
            o = row.get("fixed_object", "").strip() or row["object"]
            if rel not in TYPED_RELATIONS:
                problems.append(f"{row['review_id']}: unknown relation {rel!r}")
                continue
            add(row["subject"], row["relation"], row["object"], "reject", row)
            add(s, rel, o, "accept", row)
        elif decision:
            problems.append(f"{row['review_id']}: unknown decision {decision!r} (use accept / reject / fix)")
    return list(out.values()), dict(counts), problems


def cmd_export(args):
    review_rows = list(csv.DictReader(open(REVIEW_FACTS_CSV, encoding="utf-8")))
    rows, counts, problems = export_rows(review_rows)
    for p in problems:
        log.info(f"  skipped {p}")
    VERIFIED_FACTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(VERIFIED_FACTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=VERIFIED_COLUMNS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["decision"], r["subject"], r["relation"], r["object"])))
    log.info(f"review decisions: {counts}")
    log.info(f"wrote {len(rows)} verified facts -> {VERIFIED_FACTS_CSV}; now run kg/consolidate_triples.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="draft", choices=["draft", "export"])
    parser.add_argument("--chunks", type=Path, default=CHUNKS_JSONL)
    parser.add_argument("--types", type=Path, default=ENTITY_TYPES_JSON)
    parser.add_argument("--min_cooc", type=int, default=2, help="Draft facts for pairs co-mentioned in at least this many sentences")
    parser.add_argument("--max_pairs", type=int, default=None, help="Only the N most co-mentioned pairs")
    parser.add_argument("--limit", type=int, default=None, help="Only send this many NEW sentences to the LLM (pilot)")
    parser.add_argument("--no_llm", action="store_true", help="Grammar-rule facts only (no Ollama)")
    args = parser.parse_args()
    cmd_draft(args) if args.command == "draft" else cmd_export(args)


if __name__ == "__main__":
    main()
