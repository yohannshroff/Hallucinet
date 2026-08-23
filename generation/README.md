# generation/ (Week 5)

LLM integration on top of Week 4's hybrid retrieval.

- `prompt_template.py` — the grounding-only system prompt: answer strictly
  from the retrieved context, explicitly say so when the context doesn't
  support an answer, never fall back on the model's own training-time
  knowledge. This is the core anti-hallucination lever for this project.
- `ollama_client.py` — thin wrapper around the local Ollama `/api/chat`
  HTTP endpoint.
- `generate_answer.py` — ties retrieval (`retrieval/hybrid_retrieve.py`) +
  the prompt template + Ollama into one grounded-answer call.
- `run_sample_questions.py` — runs the 10 questions in
  `data/eval/sample_questions.csv` through the full pipeline and saves a
  transcript to `docs/week5_sample_answers.md` for manual review (not
  scored — Week 7 turns this into a real scored ablation).

See [docs/week5_notes.md](../docs/week5_notes.md) for what was run and
verified, and [docs/manual_setup_ollama.md](../docs/manual_setup_ollama.md)
for getting Ollama running.
