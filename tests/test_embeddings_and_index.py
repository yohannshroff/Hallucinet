"""End-to-end correctness check for embeddings + FAISS search, using 3 tiny
fixture sentences instead of the full corpus. Requires sentence-transformers
and faiss-cpu to be installed (run scripts/setup_env.sh first) -- these
tests are skipped otherwise so `pytest tests/` still runs cleanly on a bare
Python install.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

pytest.importorskip("sentence_transformers")
pytest.importorskip("faiss")

from build_embeddings import embed_chunks  # noqa: E402
from build_faiss_index import build_index  # noqa: E402


FIXTURE_SENTENCES = [
    "Rani Lakshmibai ruled the princely state of Jhansi.",
    "Mangal Pandey was a sepoy in the Bengal Army.",
    "The Doctrine of Lapse was an East India Company annexation policy.",
]


def test_embed_and_search_returns_self_as_top_hit():
    embeddings = embed_chunks(FIXTURE_SENTENCES)
    assert embeddings.shape == (3, 384)

    index = build_index(embeddings)
    assert index.ntotal == 3

    # querying with a sentence verbatim should return itself as the top hit
    query_vec = embeddings[1:2]  # the Mangal Pandey sentence
    scores, indices = index.search(query_vec, 1)
    assert indices[0][0] == 1
