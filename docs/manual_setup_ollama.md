# Manual setup: Ollama (needed starting Week 5)

Not installed by any script in this repo — run these yourself when you
reach Week 5 (LLM integration).

## Install

```bash
brew install ollama
```

(Or download from https://ollama.com/download for other platforms.)

## Pull a model

```bash
ollama pull llama3
# or, for a lighter/faster option:
ollama pull gemma3
```

## Run

```bash
ollama serve
```

This starts the local API at `http://localhost:11434` by default, matching
`OLLAMA_HOST` in `.env.example`. Set `OLLAMA_MODEL` to whichever model you
pulled.

## Sanity check

```bash
ollama run llama3 "Say hello in one sentence."
```
