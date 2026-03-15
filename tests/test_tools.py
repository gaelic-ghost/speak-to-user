from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.text_chunking import chunk_text_for_tts


def test_chunk_text_for_tts_always_chunks_by_sentence_even_when_short() -> None:
    text = "A short paragraph. A second sentence."

    result = chunk_text_for_tts(text, max_chars=200)

    assert result == ["A short paragraph.", "A second sentence."]


def test_chunk_text_for_tts_keeps_one_sentence_per_chunk() -> None:
    text = "First sentence. Second sentence. Third sentence."

    result = chunk_text_for_tts(text, max_chars=36)

    assert result == [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]


def test_chunk_text_for_tts_splits_only_overlong_sentences_by_words() -> None:
    text = (
        "This is the first sentence. "
        "This second sentence is longer than the limit. "
        "Supercalifragilisticexpialidocious"
    )

    result = chunk_text_for_tts(text, max_chars=20)

    assert result == [
        "This is the first",
        "sentence.",
        "This second sentence",
        "is longer than the",
        "limit.",
        "Supercalifragilistic",
        "expialidocious",
    ]
