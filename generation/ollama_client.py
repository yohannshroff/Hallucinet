"""Thin wrapper around the local Ollama HTTP API (/api/chat).

Usage:
    python generation/ollama_client.py "Say hello in one sentence."
"""

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import OLLAMA_HOST, OLLAMA_MODEL, get_logger  # noqa: E402

log = get_logger("ollama_client")

DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_TEMPERATURE = 0.1  # low temperature: we want faithful grounding, not creative variety


def is_reachable(host: str = OLLAMA_HOST) -> bool:
    try:
        resp = requests.get(host, timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def chat(
    messages: list,
    model: str = OLLAMA_MODEL,
    host: str = OLLAMA_HOST,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Send a /api/chat-style messages list to Ollama, return the reply text."""
    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
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
