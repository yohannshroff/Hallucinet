# generation/ (Part 5)

LLM integration on top of Part 4's hybrid retrieval.

- `prompt_template.py` — the grounding-only system prompt: answer strictly
  from the retrieved context, explicitly say so when the context doesn't
  support an answer, never fall back on the model's own training-time
  knowledge. This is the core anti-hallucination lever for this project.
  The context is a numbered source list (graph facts, then book passages)
  and the model cites by number (`[2]`, `[1][4]`), never by file name or
  page. It is also told that "mentioned together" is not a relationship,
  to say who did what only when a source states it, and to answer in plain
  sentences.
- `ollama_client.py` — thin wrapper around the local Ollama `/api/chat`
  HTTP endpoint.
- `generate_answer.py` — ties retrieval (`retrieval/hybrid_retrieve.py`) +
  the prompt template + Ollama into one grounded-answer call.
- `run_sample_questions.py` — runs the questions in
  `data/eval/sample_questions.csv` through the full pipeline and saves a
  transcript to `docs/part5_sample_answers.md` for manual review (not
  scored — Part 7 turns this into a real scored ablation).

See [docs/part5_notes.md](../docs/part5_notes.md) for what was run and
verified, and [docs/manual_setup_ollama.md](../docs/manual_setup_ollama.md)
for getting Ollama running.
