# Part 5 notes

Goals: get a local LLM running, write a grounding-only prompt template, and
test answer generation on 10 sample questions.

- Ollama: installed via Homebrew, running as a `brew services` background
  daemon (`http://localhost:11434`). Model: `gemma3:4b` — small enough to
  run acceptably on CPU, per the master plan's demo-risk mitigation (a
  larger model would answer better but risks stalling a live viva demo).
- Prompt template (`generation/prompt_template.py`): a system prompt that
  forbids outside knowledge and requires an explicit "does not contain
  enough information" response when the context doesn't support an answer,
  rather than letting the model guess.
- Pipeline: `generation/generate_answer.py` chains Part 4's
  `retrieve()` → `format_context_only()` → the prompt template → Ollama.
- Ran all 10 questions in `data/eval/sample_questions.csv` through the full
  pipeline; transcript saved to
  [docs/part5_sample_answers.md](part5_sample_answers.md).

## Results (manual read of the 10-question transcript)

**8/10 solid** — correctly grounded, cited, and on-topic (Q01–Q04, Q06,
Q07, Q09, and the deliberately-unanswerable Q10, which the model correctly
declined rather than inventing a Napoleon connection — exactly the
behavior the prompt template is meant to produce).

**2 genuine problems found, both informative:**

- **Q05 ("Who commanded the British forces that recaptured Delhi?") —
  retrieval gap, not a generation hallucination.** Entity extraction only
  found `Delhi` and `British East India Company` as seeds — it can't find
  `John Nicholson` because his name isn't in the query. In the graph,
  Nicholson connects to `Fall of Delhi` (`led`), and `Fall of Delhi`
  connects to `Delhi` (`located_in`) — so Nicholson is **2 hops** from the
  `Delhi` seed, but `graph_search.py` only does 1-hop expansion (a
  limitation already flagged in `docs/part4_notes.md`'s open items). With
  no useful graph facts, the model reached for the closest thing in
  context (`Lord Canning appointed_by East India Company`) and gave an
  irrelevant, off-target answer. It didn't fabricate anything, but it also
  didn't say "I don't know" when it should have — worth watching in Part
  6's trust scoring and Part 7's evaluation set.
- **Q08 ("What happened to Bahadur Shah II after the fall of Delhi?") —
  entity conflation within an otherwise-grounded answer.** The answer
  correctly states he was exiled to Rangoon, then later in the *same
  answer* says he "died in the Terai plains of Nepal in 1859" — which is
  actually Begum Hazrat Mahal's story (she fled to Nepal), pulled in from
  the same batch of retrieved chunks and conflated with Bahadur Shah II.
  Every individual fact traces to a real cited source, but they got
  attributed to the wrong person when synthesized. This is exactly the
  subtle, source-grounded-but-still-wrong error type that Part 6's
  claim-level NLI entailment check is meant to catch — a per-claim check
  against its cited passage should flag "Bahadur Shah II ... died in ...
  Nepal" as NOT entailed by the Bahadur_Shah_Zafar article.

## Why this is worth keeping, not fixing away

Both failures are real, useful signal for the project's actual thesis
(hybrid retrieval + verification reduces hallucination) — they're the kind
of case Part 6 (trust scoring) and Part 7 (ablation) are specifically
built to catch and quantify, not bugs to silently patch over. Recommend
adding both Q05 and Q08's underlying question types to the Part 7 30-40
question test set as known-hard cases.

## Open items

- Consider 2-hop graph expansion (or expanding seed entities via one
  extra graph hop before vector search) to fix the Q05-style gap — decide
  after seeing how often this pattern recurs in Part 7's larger test set.
- Part 6's trust score should be run against Q08's transcript as a sanity
  check that NLI entailment actually catches the Nepal/Rangoon conflation.
