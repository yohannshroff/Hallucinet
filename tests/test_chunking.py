"""Tests for the chunking logic in scripts/clean_and_chunk.py, using a
synthetic word list rather than real fetched articles."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from clean_and_chunk import chunk_words, clean_text  # noqa: E402


def test_chunk_words_respects_size_and_overlap():
    words = [str(i) for i in range(500)]
    chunks = chunk_words(words, chunk_size=180, overlap=40)

    assert all(len(c) <= 180 for c in chunks)
    # stride should be 140: second chunk starts where first chunk's word 140 is
    assert chunks[0][140] == chunks[1][0]


def test_chunk_words_covers_all_words():
    words = [str(i) for i in range(50)]
    chunks = chunk_words(words, chunk_size=180, overlap=40)
    # short input fits in a single chunk
    assert len(chunks) == 1
    assert chunks[0] == words


def test_chunk_words_empty_input():
    assert chunk_words([], chunk_size=180, overlap=40) == []


def test_clean_text_strips_citation_markers_and_boilerplate():
    raw = "Rani Lakshmibai ruled Jhansi.[1] She fought the British.[23]\n\n== References ==\nSome citation text."
    cleaned = clean_text(raw)
    assert "[1]" not in cleaned
    assert "[23]" not in cleaned
    assert "Some citation text" not in cleaned
    assert "Rani Lakshmibai ruled Jhansi." in cleaned


def test_clean_text_strips_section_headers():
    raw = "Intro text.\n\n== Early life ==\nMore text here."
    cleaned = clean_text(raw)
    assert "==" not in cleaned
    assert "Intro text." in cleaned
    assert "More text here." in cleaned
