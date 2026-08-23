# Manual setup: Ollama (needed starting Week 5)

Not installed by any Week 1-4 script — set this up when you reach Week 5
(LLM integration). This project's dev instance was set up as below.

## Install (Homebrew, macOS)

```bash
brew install ollama
brew services start ollama
```

- API: `http://localhost:11434`
- Stop with `brew services stop ollama`; check status with
  `brew services info ollama`.

(Other platforms: download from https://ollama.com/download and run
`ollama serve`.)

## Pull a model

```bash
ollama pull gemma3:4b
# or, the other option the master plan named:
ollama pull llama3.2
```

`gemma3:4b` is what this repo's `.env.example` defaults to — small enough
to run acceptably on CPU, which matters for live-demo risk (see the master
plan's risk register). A larger model answers better but is slower.

## Sanity check

```bash
ollama run gemma3:4b "Say hello in one sentence."
```

## After it's running

Copy `.env.example` to `.env` (if you haven't already) and set
`OLLAMA_HOST`/`OLLAMA_MODEL` to match. `scripts/common.py` loads these via
`python-dotenv`, and `generation/` reads the connection config from there.

Then:

```bash
python generation/generate_answer.py "Who led the resistance at Jhansi?"
python generation/run_sample_questions.py
```
