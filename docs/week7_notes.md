# Week 7 notes

Goals: finalize a 30-40 question Q&A test set, run the vector-only vs
graph-only vs hybrid ablation, score it, and produce the comparison chart.

- Test set: [data/eval/qa_test_set.csv](../data/eval/qa_test_set.csv), 30
  questions across person/location/cause/event/aftermath categories, each
  with `expected_entities` (the answer's key entity/entities) drawn
  directly from the Week 1 KG seed so ground truth is known.
- Ablation: `eval/ablation.py` ran all 30 questions x 3 modes (90 answers
  total) through `generate_answer` with `mode` threaded down to
  `retrieve()` (added a `mode` param -- "vector"/"graph"/"hybrid" -- to
  `retrieval/hybrid_retrieve.py` for this).
- Scoring: **automated, not hand-scored** -- see "Scoring methodology"
  below for why and how. Results:
  [docs/week7_ablation_results.csv](week7_ablation_results.csv), chart:
  [docs/week7_accuracy_by_mode.png](week7_accuracy_by_mode.png).

## Headline numbers (90 answers, 30 questions x 3 modes)

| mode | correct | partial | wrong | hallucinated | refused |
|---|---|---|---|---|---|
| graph-only | 20 | 5 | 1 | 1 | 3 |
| hybrid | 15 | 9 | 6 | 0 | 0 |
| vector-only | 6 | 7 | 9 | 7 | 1 |

**Read this with the caveats below before quoting it anywhere** -- the
raw numbers make graph-only look strictly best, which is not the honest
takeaway.

## Scoring methodology (and why it's not hand-scored)

The master plan suggests scoring by hand: correct/partial/wrong/
hallucinated, by a human reading each answer. That's the right gold
standard, but not something this pipeline can do unattended over 90
answers. Instead, `eval/ablation.py` uses two independent, non-LLM
signals already built earlier in this project:

1. **Entity match**: does the answer contain a case-insensitive substring
   match for any of the question's `expected_entities`?
2. **Trust score** (Week 6's NLI entailment check): is the answer actually
   grounded in the evidence it was given?

`correct` = both; `partial` = right entity, weak grounding; `wrong` =
grounded but the wrong entity; `hallucinated` = neither; `refused` = the
model declined to answer at all (kept separate from `wrong` since it's a
much safer failure mode than confidently inventing an answer).

**This is a proxy, not a substitute for human judgment**, and the master
plan's risk register explicitly recommends a 2-rater manual agreement
check on a sample for credibility -- do that against this script's output
before these numbers go in the final report/slides.

## Why graph-only's "win" needs a big asterisk

Two structural biases in this methodology favor graph-only, independent
of actual answer quality:

1. **The test set was authored from the same KG graph-only retrieves
   from.** Nearly every question is phrased to match an exact
   `relationships.csv` triple almost verbatim (by design, so ground truth
   would be known) -- of course the retrieval method built from that exact
   triple store does well on it. This isn't a fair neutral benchmark; it's
   closer to "does graph search find graph facts."
2. **The entity-match scorer favors canonical KG naming, which graph
   answers naturally echo and vector answers often don't.** Two concrete
   examples found by actually reading the output:
   - **T25** ("What was the EIC's stated justification for annexing
     Awadh?") — vector mode's answer correctly said *"alleged
     maladministration"* (quoting the source), but the test set originally
     only listed `misgovernance` as the expected entity — a synonym, not
     the source's actual word. Scored `wrong` until caught and fixed
     (`data/eval/qa_test_set.csv` now lists both terms) — see the git diff
     for the fix and the T25 rows in the results CSV for the corrected
     before/after (vector: wrong → correct).
   - **T21** ("What caused the Greased Cartridge Incident?") — vector
     mode's answer accurately described the cow-tallow/pig-lard grease
     controversy in the source's own words, but never says the KG's
     literal entity name "Enfield Cartridge Issue" — scored `hallucinated`
     despite being factually accurate prose.
   
   Net effect: vector-only's hallucination/wrong counts are inflated by
   scorer brittleness to paraphrasing, not just by actual invented facts.
   Take the vector-only column as an upper bound on its real error rate,
   not a precise one.

Given both biases point the same direction (favoring graph-only, against
vector-only), **the true relative ordering is less clear-cut than the raw
table suggests** — likely narrower than shown, possibly closer for
hybrid vs. graph-only specifically. This is exactly why the master plan
calls for a human-scored check before reporting final numbers.

## Case studies (reading actual answers, not just verdicts)

**T13 — "Who commanded the British forces that recaptured Delhi?"**
(expected: John Nicholson, a 2-hop graph fact — Delhi → Fall of Delhi →
John Nicholson, flagged as a hard case back in Week 4/5.)
- graph-only: **correctly refused** — "does not contain enough
  information." 1-hop search genuinely can't reach Nicholson; the model
  didn't invent an answer either. This is the safe failure mode.
- hybrid: **confidently wrong** — answered "Lord Canning appointed by the
  British East India Company," a real, cited, but completely irrelevant
  graph fact that happened to be the closest thing in context. Adding
  vector context didn't help find the right answer here, and gave the
  model material to produce a *confident* wrong answer instead of
  refusing like graph-only did. This is a genuinely useful, reproducible
  finding: more context isn't strictly safer.

**T30 — "What war cry emerged in the aftermath of the Cawnpore
massacre?"** (expected: "Remember Cawnpore!", a fact that exists only in
the fetched Wikipedia prose, not in the KG at all.)
- graph-only: **correctly refused** — this fact genuinely isn't in the
  graph, and it said so instead of guessing. Exactly the intended
  behavior, and a clean positive case for why vector retrieval still
  matters even though graph-only "wins" on the headline numbers.
- vector-only and hybrid: both correctly surfaced "Remember Cawnpore!"

**T10 — "Who commanded rebel forces in Delhi?"** (expected: Bakht Khan)
- graph-only: **hallucinated** — answered "Bahadur Shah II commanded
  rebel forces in Delhi," which is wrong (Bakht Khan did). Root cause: in
  the graph, `Bakht Khan --commanded--> rebel forces in Delhi` points to a
  free-text `:Concept` node (see `docs/schema.md`), which is **not**
  graph-connected to the `Delhi` `:Location` entity itself — only its
  *text* happens to mention Delhi. So when the query resolves to the
  `Delhi` entity as its seed, 1-hop search only finds
  `Bahadur Shah II --located_in--> Delhi`, never reaches Bakht Khan at
  all, and the model plausibly-but-wrongly stitched "commanded... in
  Delhi" onto the entity it did see. This is a real, specific
  architectural gap (Concept nodes aren't linked to the Location entities
  named in their own text) worth fixing before this KG is extended
  further — flagged as an open item below.

## Bugs/gaps found and fixed this week

- **T25's expected answer** (documented above) — test-set authoring bug,
  fixed.
- **Missing alias**: `Tantia Tope`'s entities.csv row only listed "Tantia
  Topi" as an alias; "Tatya Tope" — a very common alternate spelling
  used in some of the fetched source text (see T02/T03's vector-mode
  answers) — was missing entirely, meaning our own entity resolution
  would fail to recognize it in a user query. Added it, re-validated,
  re-loaded the graph.

## Open items

- Fix the Concept-node disconnection from T10 before Week 8 locks in the
  demo -- either link Concept nodes to any Location entities mentioned in
  their text, or accept the gap and document it as a known limitation in
  the final report.
- Run the recommended 2-rater manual agreement check on a sample (e.g. all
  90 hybrid+graph rows, or a random 20) before quoting these numbers in
  the final report/slides.
- Consider whether the test set needs a second, independently-authored
  batch of questions (not derived from the same KG) for a fairer
  vector-vs-graph comparison in any future iteration.
