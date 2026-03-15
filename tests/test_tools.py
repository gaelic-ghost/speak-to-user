from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.text_chunking import chunk_text_for_tts


def test_chunk_text_for_tts_keeps_short_text_together_when_it_fits() -> None:
    text = "A short paragraph. A second sentence."

    result = chunk_text_for_tts(text, max_chars=200)

    assert result == [text]


def test_chunk_text_for_tts_packs_words_without_waiting_for_sentence_boundaries() -> None:
    text = "First sentence. Second sentence. Third sentence."

    result = chunk_text_for_tts(text, max_chars=18)

    assert result == [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]


def test_chunk_text_for_tts_splits_overlong_text_by_words() -> None:
    text = (
        "This is the first sentence. "
        "This second sentence is longer than the limit. "
        "Supercalifragilisticexpialidocious"
    )

    result = chunk_text_for_tts(text, max_chars=20)

    assert result == [
        "This is the first",
        "sentence. This",
        "second sentence is",
        "longer than the",
        "limit.",
        "Supercalifragilistic",
        "expialidocious",
    ]
