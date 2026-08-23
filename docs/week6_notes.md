# Week 6 notes

Goals: claim splitting, NLI-based entailment scoring against retrieved
evidence, and a trust score formula.

- Claim splitting (`eval/claim_splitting.py`): a claim = one sentence, via
  spaCy sentence segmentation, with citation brackets (`[1]`,
  `[https://...]`) stripped and trivial fragments (<4 words) dropped. Full
  atomic-fact decomposition would catch more but adds real complexity for
  a 2-month project -- sentence-level is the simple, defensible choice.
- Entailment (`eval/entailment.py`): `cross-encoder/nli-MiniLM2-L6-H768`, a
  small fast NLI cross-encoder -- the "lighter sentence-transformers
  cross-encoder" the master plan names as the alternative to
  `roberta-large-mnli`. Given a claim and a pool of evidence texts, it
  scores the claim against every evidence text and keeps the
  highest-entailment match. Label order is read from the model config at
  runtime rather than hardcoded (checked empirically: `{0: contradiction,
  1: entailment, 2: neutral}` for this checkpoint).
- Trust score (`eval/trust_score.py`): **% of an answer's claims that are
  entailed by the retrieved evidence.** `None` (not 0%) when the answer
  made no checkable claims at all, so a correct "I don't know" refusal
  isn't penalized as if it were an unsupported claim.
- This deliberately avoids the circularity the master plan's risk register
  warns about (LLM grading itself) -- entailment is checked by a separate,
  independent, pretrained model, not the generating LLM.

## Two real bugs found and fixed while testing on real output

Ran `eval/run_trust_scores.py` over the Week 5 sample questions twice —
before and after fixing these — see
[docs/week6_trust_scores.md](week6_trust_scores.md) for the final
transcript.

1. **The refusal phrase was being scored as an unsupported claim.** Q10's
   correct "The provided context does not contain enough information..."
   response was getting split, fed to the NLI model, and scored 0% trust
   — exactly backwards, since refusing to answer is the *desired*
   behavior. Fixed by excluding the exact refusal phrase (now a shared
   `REFUSAL_PHRASE` constant in `generation/prompt_template.py`) from
   claim scoring — its trust score is now correctly `N/A`, not `0%`.

2. **Evidence chunks were fed to the NLI model whole, not split into
   sentences — this was quietly destroying real, well-supported claims'
   scores.** Cross-encoder NLI models are trained on single-sentence
   premise/hypothesis pairs; comparing a claim against an entire ~180-word
   multi-topic chunk dilutes the signal badly. Concrete before/after on
   Q09 ("Which act transferred rule of India from the East India Company
   to the Crown?"): a claim that's a near-verbatim quote of the retrieved
   passage scored **0.05 (neutral)** against the whole chunk, but **0.96
   (entailed)** once the chunk was split into sentences and compared
   sentence-to-sentence. Fixed in
   `evidence_texts_from_bundle()`, which now runs each vector chunk
   through the same sentence splitter used for claims.

**Combined effect across all 10 sample questions: average trust score went
from 21% to 61%** (measured over 9 scored answers both times — Q10's
refusal is correctly excluded from the average once fix #1 landed). The
21% number wasn't really "the system hallucinates most of the time" — it
was substantially a measurement artifact from feeding the NLI model the
wrong granularity of evidence. Worth stating plainly in the eventual
report: **this is exactly why an independent verification step matters**
— it caught real bugs in our own evaluation code, not just problems in the
generation side.

## Remaining (real, not a bug) limitation

Even after both fixes, claims that *paraphrase* rather than closely quote
the evidence still tend to score lower than claims that echo it closely
(e.g. Q09's first claim, a paraphrase, still scored only 0.05 while the
two more quote-like claims scored 0.96 each). This is a genuine limitation
of a small cross-encoder NLI model, not a bug — worth flagging in the
final report as a specific direction for improvement (a larger NLI model,
or claim/evidence normalization) rather than something to silently work
around.

## Open items

- Consider running `roberta-large-mnli` (the master plan's other named
  option) on a sample to see whether the paraphrase-scoring gap narrows
  with a larger model, time permitting.
- The 2-rater manual agreement check the master plan's risk register
  recommends for scoring credibility should be run against this script's
  verdicts before they go in the final report (see also
  `docs/week7_notes.md`).
