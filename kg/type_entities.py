"""Stage 2 of building the knowledge graph: have the local LLM type and clean
the entity names found by kg/find_candidates.py (about 5-10 minutes).

spaCy's small model mislabels many names in this book (it calls Meerut and
Jhansi people, Havelock and Hodson places), and the name list also contains
footnote debris ("Vol", "N.d", "London, 1908"). Typing a name from a short
example sentence is something the 4B model does well, so each name gets one
of PERSON / PLACE / ORG / EVENT / OTHER; OTHER (junk, titles, book names,
common words) is dropped from the graph.

Names are sent in small batches and the reply is constrained to a JSON
schema, so it can only contain valid types. Results are saved after every
batch to data/kg_build/entity_types.json (names + types only, no book text),
and re-running resumes where it stopped.

Usage:
    python kg/type_entities.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generation"))
from common import ENTITY_TYPES_JSON, MENTIONS_JSON, get_logger  # noqa: E402
from find_candidates import build_lexicon  # noqa: E402

from ollama_client import chat, is_reachable  # noqa: E402

log = get_logger("type_entities")

BATCH_SIZE = 12
TYPE_CHOICES = ["PERSON", "PLACE", "ORG", "EVENT", "OTHER"]
SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"n": {"type": "integer"}, "type": {"type": "string", "enum": TYPE_CHOICES}},
                "required": ["n", "type"],
            },
        }
    },
    "required": ["items"],
}

SYSTEM_PROMPT = """You label names that appear in a history book about the Indian Mutiny of 1857.
For each numbered name you get an example sentence from the book. Choose one type:
PERSON - an individual person (a surname or first name alone still counts if it clearly refers to a person)
PLACE - a city, town, region, country, river, fort, gate, building or other geographic place
ORG - an army unit or regiment, government body, company, ruling house, or a people/community acting as a party (e.g. Sikhs, Europeans)
EVENT - a battle, siege, mutiny, massacre, war or similar named event
OTHER - anything else: a book or author, a title or common word, a date, a religion, a language, an abbreviation, or junk
Answer with JSON: one entry per name, with its number and type."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mentions", type=Path, default=MENTIONS_JSON)
    parser.add_argument("--out", type=Path, default=ENTITY_TYPES_JSON)
    parser.add_argument("--limit", type=int, default=None, help="Only type this many NEW names (pilot)")
    args = parser.parse_args()

    if not is_reachable():
        log.info("Ollama not reachable -- start it first (brew services start ollama)")
        sys.exit(1)

    mentions = json.loads(args.mentions.read_text(encoding="utf-8"))
    from collections import Counter

    for v in mentions.values():
        v["types"] = Counter(v["types"])
    names = sorted(build_lexicon(mentions), key=lambda n: -mentions[n]["count"])

    done = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    todo = [n for n in names if n not in done]
    if args.limit:
        todo = todo[: args.limit]
    log.info(f"{len(names)} names, {len(done)} already typed, {len(todo)} to do")

    started = time.perf_counter()
    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i : i + BATCH_SIZE]
        lines = []
        for n, name in enumerate(batch, start=1):
            example = mentions[name].get("example", "")
            lines.append(f'{n}. {name} -- "{example[:220]}"')
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Names:\n" + "\n".join(lines)},
        ]
        try:
            reply = json.loads(chat(messages, temperature=0.0, num_predict=560, format=SCHEMA))
        except Exception as exc:  # noqa: BLE001 - retried on the next run
            log.info(f"  batch at {i}: FAILED ({exc})")
            continue
        for item in reply.get("items", []):
            if 1 <= item["n"] <= len(batch):
                done[batch[item["n"] - 1]] = item["type"]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(done, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        n_done = min(i + BATCH_SIZE, len(todo))
        elapsed = time.perf_counter() - started
        log.info(f"  {n_done}/{len(todo)} names  (~{elapsed / n_done * (len(todo) - n_done) / 60:.0f} min left)")

    counts = Counter(done.values())
    log.info(f"typed {len(done)} names: {dict(counts)}")


if __name__ == "__main__":
    main()
