"""Stage 4 of building the knowledge graph: turn the cached LLM answers and
rule-based facts into data/kg_seed/entities.csv and relationships.csv
(seconds, no LLM -- tune and re-run freely).

What it does:
0. Reads the grammar-extracted facts (kg/extract_facts.py) that passed the
   NLI check, plus the rule-based facts (kg/find_candidates.py).
1. Merges name variants into one entity: "Havelock" -> "Henry Havelock"
   when exactly one longer name ends with it, and spelling variants such as
   "Tatya Tope" / "Tantia Tope" by fuzzy matching within the same type.
2. Drops facts whose entity types make the relation impossible (a place
   cannot "lead" a person) -- the type table below.
3. Counts how many different book sentences support each fact, and keeps
   facts with enough support (rule-based facts count as certain).
4. Writes the CSVs in the existing schema (docs/schema.md); every
   relationship cites the book pages it came from.

Usage:
    python kg/consolidate_triples.py
    python kg/consolidate_triples.py --min_support 2
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import (  # noqa: E402
    CANDIDATES_JSONL,
    COOCCURRENCE_JSONL,
    ENTITIES_CSV,
    ENTITY_TYPES_JSON,
    EXTRACTED_JSONL,
    MENTIONS_JSON,
    RELATIONSHIPS_CSV,
    get_logger,
)
from find_candidates import PERSON_TITLES  # noqa: E402
from extract_facts import EXCLUDED_NAMES, LLM_TYPE_TO_SCHEMA_TYPE  # noqa: E402

log = get_logger("consolidate_triples")

PERIOD_TAG = "1857_revolt"
FUZZY_MERGE_THRESHOLD = 88

P, L, O, E = "Person", "Location", "Organization", "Event"
ACTORS = {P, O}
# relation -> (allowed subject types, allowed object types)
TYPE_RULES = {
    "led": (ACTORS, {O, E, L}),
    "commanded": ({P}, {O, E}),
    "ruled": (ACTORS, {L, O}),
    "fought_against": (ACTORS, {P, O, L}),
    "allied_with": (ACTORS, ACTORS),
    "killed": (ACTORS, {P, O}),
    "executed": (ACTORS, {P, O}),
    "captured": (ACTORS, {L, P, O}),
    "besieged": (ACTORS, {L, O}),
    "located_in": ({L, E, O}, {L}),
    "part_of": ({L, E, O}, {L, E, O}),
    "caused_by": ({E, O}, {E, O, P}),
    "triggered": ({E, O, P}, {E, O}),
    "succeeded_by": ({P}, {P}),
    "appointed_by": ({P}, ACTORS),
    "declared_emperor_by": ({P}, ACTORS),
    "died_during": ({P}, {E}),
    "member_of": (ACTORS, {O}),
    "disbanded_by": ({O}, ACTORS),
    "associated_with": ({P, L, O, E}, {P, L, O, E}),
}

# The book refers to the same thing in several ways; these are merged into one entity.
ALIAS_OVERRIDES = {
    "Wilson": "Archdale Wilson",
    "Brigadier Wilson": "Archdale Wilson",
    "Company": "East India Company",
    "Mutiny": "Indian Mutiny",
    "Indian Rebellion": "Indian Mutiny",
    "Sepoy Revolt": "Indian Mutiny",
}


# Lord / Lady / Mrs / Miss tell two people with the same surname apart ("Lady Canning" is not
# "Canning" the Governor-General), so they are not stripped when comparing names.
NON_STRIPPABLE_TITLES = {"lord", "lady", "mrs", "miss", "rani", "begum", "king", "emperor"}
# A bare surname is only absorbed into a full name if the full name is mentioned at least this
# fraction as often -- otherwise the bare name probably refers to someone not in the list.
BARE_NAME_MIN_RATIO = 0.25


def strip_titles(name: str) -> list:
    tokens = re.sub(r"[.,]", "", name).split()
    while len(tokens) > 1 and tokens[0].lower() in PERSON_TITLES - NON_STRIPPABLE_TITLES:
        tokens = tokens[1:]
    return tokens


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def merge_name_variants(names_by_type: dict, mention_count: dict) -> dict:
    """Return {surface: canonical surface}."""
    from fuzzywuzzy import fuzz

    uf = UnionFind()
    for etype, names in names_by_type.items():
        names = sorted(set(names))
        for n in names:
            uf.find(n)

        if etype == P:  # surname / short form -> the one longer name that ends with it
            tails = defaultdict(list)
            for n in names:
                toks = strip_titles(n)
                # "Lady Canning" / "Mrs Inglis" are distinct people, never the target a bare surname merges into
                if len(toks) >= 2 and toks[0].lower() not in NON_STRIPPABLE_TITLES:
                    tails[toks[-1].lower()].append(n)
            for n in names:
                toks = strip_titles(n)
                if len(toks) == 1:
                    candidates = tails.get(toks[0].lower(), [])
                    if len(candidates) == 1 and mention_count.get(candidates[0], 0) >= BARE_NAME_MIN_RATIO * mention_count.get(n, 0):
                        uf.union(candidates[0], n)
                elif len(toks) >= 2:  # "General Wilson" / "Archdale Wilson" style: same last name AND first name matches
                    for other in names:
                        otoks = strip_titles(other)
                        if other != n and len(otoks) >= 2 and otoks[-1].lower() == toks[-1].lower() and toks[0].lower() == otoks[0].lower():
                            uf.union(n, other)

        if etype in (E, O):  # event/organization names differ systematically ("Mutiny at Indore" vs "Mutiny at Dinapore")
            continue
        for i, a in enumerate(names):  # spelling variants of a person or place name
            for b in names[i + 1 :]:
                if abs(len(a) - len(b)) <= 3 and fuzz.ratio(a.lower(), b.lower()) >= FUZZY_MERGE_THRESHOLD:
                    uf.union(a, b)

    groups = defaultdict(list)
    for names in names_by_type.values():
        for n in names:
            groups[uf.find(n)].append(n)
    canonical = {}
    for members in groups.values():
        best = max(members, key=lambda n: (len(strip_titles(n)) >= 2, mention_count.get(n, 0), len(n)))
        for n in members:
            canonical[n] = best
    return canonical


def pages_of(title: str) -> str:
    m = re.search(r"\(pp\. ([\d-]+)\)", title)
    return f"The Great Mutiny, pp. {m.group(1)}" if m else title


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min_support", type=int, default=1, help="Minimum distinct supporting sentences for an extracted fact")
    parser.add_argument("--min_entail", type=float, default=0.5, help="Minimum NLI entailment (does the source sentence entail the fact?) for an extracted fact")
    parser.add_argument("--min_cooc", type=int, default=3, help="Minimum sentences in which two entities are co-mentioned for an associated_with edge")
    parser.add_argument("--max_per_entity", type=int, default=6, help="Keep an associated_with edge only if it is among the strongest N of at least one endpoint")
    parser.add_argument("--entities_out", type=Path, default=ENTITIES_CSV)
    parser.add_argument("--relationships_out", type=Path, default=RELATIONSHIPS_CSV)
    args = parser.parse_args()

    mentions = json.loads(MENTIONS_JSON.read_text(encoding="utf-8"))
    mention_count = {k: v["count"] for k, v in mentions.items()}
    llm_types = json.loads(ENTITY_TYPES_JSON.read_text(encoding="utf-8"))
    lexicon = {n: LLM_TYPE_TO_SCHEMA_TYPE[t] for n, t in llm_types.items() if t in LLM_TYPE_TO_SCHEMA_TYPE and n not in EXCLUDED_NAMES}

    # --- gather raw facts: (subject, stype, relation, object, otype, evidence_id, source_url, title) ---
    raw = []
    n_unverified = 0
    if EXTRACTED_JSONL.exists():
        for line in open(EXTRACTED_JSONL, encoding="utf-8"):
            rec = json.loads(line)
            for f in rec["facts"]:
                if f["subject"] in EXCLUDED_NAMES or f["object"] in EXCLUDED_NAMES:
                    continue
                if f["entail"] < args.min_entail:
                    n_unverified += 1
                    continue
                raw.append((f["subject"], f["subject_type"], f["relation"], f["object"], f["object_type"], ("text", rec["key"]), rec["source_url"], rec["title"]))
    n_llm = len(raw)
    log.info(f"NLI check: dropped {n_unverified} extracted facts whose sentence does not entail them (< {args.min_entail})")
    if CANDIDATES_JSONL.exists():
        for line in open(CANDIDATES_JSONL, encoding="utf-8"):
            c = json.loads(line)
            stype = lexicon.get(c["subject"], c["subject_type"] if c["rule"] == "event_name" else None)
            otype = lexicon.get(c["object"], c["object_type"] if c["rule"] == "event_name" else None)
            if stype and otype:
                raw.append((c["subject"], stype, c["relation"], c["object"], otype, ("rule", c["doc_id"]), c["source_url"], c["title"]))
    log.info(f"{n_llm} extracted facts + {len(raw) - n_llm} rule-based facts to consolidate")

    cooc = []  # (a, b, sentences, source_url, title)
    if COOCCURRENCE_JSONL.exists():
        for line in open(COOCCURRENCE_JSONL, encoding="utf-8"):
            c = json.loads(line)
            if c["a"] in lexicon and c["b"] in lexicon:
                cooc.append((c["a"], c["b"], c["count"], c["source_url"], c["title"]))

    # --- merge name variants ---
    names_by_type = defaultdict(set)
    for s, st, _, o, ot, *_ in raw:
        names_by_type[st].add(s)
        names_by_type[ot].add(o)
    for a, b, *_ in cooc:
        names_by_type[lexicon[a]].add(a)
        names_by_type[lexicon[b]].add(b)
    canonical = merge_name_variants({t: list(v) for t, v in names_by_type.items()}, mention_count)
    for name, canon in list(canonical.items()):
        target = ALIAS_OVERRIDES.get(canon)
        if target and target in canonical:
            canonical[name] = canonical[target]
    entity_type = {}
    for s, st, _, o, ot, *_ in raw:
        entity_type[canonical[s]] = st
        entity_type[canonical[o]] = ot
    for name in canonical:
        if name in lexicon:
            entity_type[canonical[name]] = lexicon[canonical[name]] if canonical[name] in lexicon else entity_type.get(canonical[name], lexicon[name])

    # --- type-check, aggregate by (subject, relation, object) ---
    facts = defaultdict(lambda: {"evidence": set(), "rule": False, "sources": Counter()})
    dropped_types = dropped_self = 0
    for s, st, rel, o, ot, evidence, source_url, title in raw:
        cs, co = canonical[s], canonical[o]
        if cs == co:
            dropped_self += 1
            continue
        allowed_s, allowed_o = TYPE_RULES[rel]
        if entity_type[cs] not in allowed_s or entity_type[co] not in allowed_o:
            dropped_types += 1
            continue
        fact = facts[(cs, rel, co)]
        fact["evidence"].add(evidence)
        fact["rule"] |= evidence[0] == "rule"
        fact["sources"][(source_url, title)] += 1
    log.info(f"dropped {dropped_types} facts with impossible types and {dropped_self} self-relations")

    kept = {k: v for k, v in facts.items() if v["rule"] or len(v["evidence"]) >= args.min_support}
    log.info(f"{len(facts)} distinct facts -> {len(kept)} kept (rule-based, or supported by >= {args.min_support} sentences)")

    # --- associated_with edges from co-occurrence ---
    pair_count, pair_sources = Counter(), defaultdict(Counter)
    for a, b, n, source_url, title in cooc:
        ca, cb = canonical[a], canonical[b]
        if ca == cb:
            continue
        key = tuple(sorted((ca, cb)))
        pair_count[key] += n
        pair_sources[key][(source_url, title)] += n
    strong = {k: n for k, n in pair_count.items() if n >= args.min_cooc}
    by_entity = defaultdict(list)
    for key, n in strong.items():
        for e in key:
            by_entity[e].append((n, key))
    keep_pairs = set()
    for e, lst in by_entity.items():
        for n, key in sorted(lst, reverse=True)[: args.max_per_entity]:
            keep_pairs.add(key)
    log.info(f"{len(pair_count)} co-mentioned pairs -> {len(strong)} with >= {args.min_cooc} sentences -> {len(keep_pairs)} associated_with edges kept (strongest {args.max_per_entity} per entity)")

    # --- entities: only those that take part in a kept fact ---
    assoc_edges = []
    for key in keep_pairs:
        a, b = key
        assoc_edges.append((a, b, pair_count[key], pair_sources[key]))
    used = {x for (s, _, o) in kept for x in (s, o)} | {x for a, b, *_ in assoc_edges for x in (a, b)}
    members = defaultdict(set)
    for surface, canon in canonical.items():
        if canon in used:
            members[canon].add(surface)

    def total_mentions(canon):
        return sum(mention_count.get(m, 0) for m in members[canon])

    ordered = sorted(used, key=lambda c: -total_mentions(c))
    ids = {c: f"E{i:03d}" for i, c in enumerate(ordered, start=1)}

    degree = Counter()
    for (s, _, o) in kept:
        degree[s] += 1
        degree[o] += 1
    for a, b, *_ in assoc_edges:
        degree[a] += 1
        degree[b] += 1

    claimed = {c.lower() for c in ordered}
    entity_rows = []
    for c in ordered:
        aliases = []
        for m in sorted(members[c] - {c}):
            if m.lower() not in claimed:
                aliases.append(m)
                claimed.add(m.lower())
        entity_rows.append(
            {
                "entity_id": ids[c], "name": c, "entity_type": entity_type[c], "aliases": ";".join(aliases),
                "period_tag": PERIOD_TAG,
                "description": f"{entity_type[c]} named {total_mentions(c)} times in the source book; appears in {degree[c]} graph fact(s).",
            }
        )

    rel_rows = []
    for (s, rel, o), v in sorted(kept.items(), key=lambda kv: (-len(kv[1]["evidence"]), kv[0])):
        (source_url, title), _ = v["sources"].most_common(1)[0]
        support = len(v["evidence"])
        rel_rows.append(
            {
                "subject": s, "relation": rel, "object": o, "source": source_url, "period_tag": PERIOD_TAG, "weight": support,
                "notes": f"{pages_of(title)}; " + (f"pattern in the text, seen in {support} place(s)" if v["rule"] else f"stated in {support} sentence(s)"),
            }
        )

    for a, b, n, sources in sorted(assoc_edges, key=lambda e: (-e[2], e[0], e[1])):
        subj, obj = (a, b) if total_mentions(a) >= total_mentions(b) else (b, a)
        (source_url, title), _ = sources.most_common(1)[0]
        rel_rows.append(
            {
                "subject": subj, "relation": "associated_with", "object": obj, "source": source_url, "period_tag": PERIOD_TAG, "weight": n,
                "notes": f"{pages_of(title)}; mentioned together in {n} sentences",
            }
        )

    args.entities_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.entities_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["entity_id", "name", "entity_type", "aliases", "period_tag", "description"])
        w.writeheader()
        w.writerows(entity_rows)
    with open(args.relationships_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "relation", "object", "source", "period_tag", "weight", "notes"])
        w.writeheader()
        w.writerows(rel_rows)

    log.info(f"wrote {len(entity_rows)} entities {dict(Counter(r['entity_type'] for r in entity_rows))}")
    log.info(f"wrote {len(rel_rows)} relationships {dict(Counter(r['relation'] for r in rel_rows).most_common())}")


if __name__ == "__main__":
    main()
