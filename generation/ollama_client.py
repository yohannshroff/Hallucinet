"""Thin wrapper around the local Ollama HTTP API (/api/chat).

Usage:
    python generation/ollama_client.py "Say hello in one sentence."
"""

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import OLLAMA_HOST, OLLAMA_KEEP_ALIVE, OLLAMA_MODEL, OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT, get_logger  # noqa: E402

log = get_logger("ollama_client")

DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_TEMPERATURE = 0.1  # low temperature: we want faithful grounding, not creative variety


_session = requests.Session()  # keep-alive: reuse one TCP connection instead of reconnecting per call


def is_reachable(host: str = OLLAMA_HOST) -> bool:
    try:
        resp = _session.get(host, timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def chat(
    messages: list,
    model: str = OLLAMA_MODEL,
    host: str = OLLAMA_HOST,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    num_predict: int = None,
    format: dict = None,
    seed: int = None,
) -> str:
    """Send a /api/chat-style messages list to Ollama, return the reply text.
    `format` (optional) is a JSON schema; Ollama then constrains the reply to match it."""
    payload_format = {"format": format} if format else {}
    resp = _session.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            **payload_format,
            # Free the model's RAM after idle time; pin the context size to the
            # server's so Ollama doesn't reload the model; cap reply length.
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": temperature,
                "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": num_predict or OLLAMA_NUM_PREDICT,
                **({"seed": seed} if seed is not None else {}),
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="A plain user prompt (no system message)")
    parser.add_argument("--model", default=OLLAMA_MODEL)
    args = parser.parse_args()

    if not is_reachable():
        log.info(f"Ollama not reachable at {OLLAMA_HOST} -- see docs/manual_setup_ollama.md")
        sys.exit(1)

    reply = chat([{"role": "user", "content": args.prompt}], model=args.model)
    print(reply)


if __name__ == "__main__":
    main()
