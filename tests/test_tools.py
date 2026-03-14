from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tools import chunk_text_for_tts


def test_chunk_text_for_tts_keeps_short_text_as_one_chunk() -> None:
    text = "A short paragraph that should stay together."

    result = chunk_text_for_tts(text, max_chars=200)

    assert result == [text]


def test_chunk_text_for_tts_prefers_paragraph_boundaries() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    result = chunk_text_for_tts(text, max_chars=36)

    assert result == [
        "First paragraph.\n\nSecond paragraph.",
        "Third paragraph.",
    ]


def test_chunk_text_for_tts_splits_long_paragraph_by_sentence_then_words() -> None:
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
