from __future__ import annotations

import re

# MARK: Constants

DEFAULT_TTS_CHUNK_MAX_CHARS = 1200
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+")


# MARK: Chunking Helpers

def _append_chunk(chunks: list[str], value: str) -> None:
    # Step 1: normalize candidate chunk text and keep only non-empty values.
    normalized = value.strip()
    if normalized:
        chunks.append(normalized)


def _split_long_word(word: str, max_chars: int) -> list[str]:
    # Step 2: hard-split a single overlong word when no softer boundary exists.
    return [word[index : index + max_chars] for index in range(0, len(word), max_chars)]


def _chunk_words(text: str, max_chars: int) -> list[str]:
    # Step 3: pack words into bounded chunks when sentence boundaries are unavailable.
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current = ""

    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        _append_chunk(chunks, current)

        if len(word) > max_chars:
            chunks.extend(_split_long_word(word, max_chars))
            current = ""
        else:
            current = word

    _append_chunk(chunks, current)
    return chunks


def _chunk_sentences(text: str, max_chars: int) -> list[str]:
    # Step 4: emit one sentence per chunk whenever possible, then fall back to word chunking
    # only for individual overlong sentences.
    sentences = [part.strip() for part in _SENTENCE_BREAK_RE.split(text.strip()) if part.strip()]
    if not sentences:
        return _chunk_words(text, max_chars)

    chunks: list[str] = []

    for sentence in sentences:
        if len(sentence) > max_chars:
            chunks.extend(_chunk_words(sentence, max_chars))
        else:
            _append_chunk(chunks, sentence)
    return chunks


def chunk_text_for_tts(text: str, max_chars: int = DEFAULT_TTS_CHUNK_MAX_CHARS) -> list[str]:
    # Step 5: always split through the sentence-first chain so every request is chunked,
    # while still allowing overlong single sentences to fall back to word chunking.
    normalized = text.strip()
    if not normalized:
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    return _chunk_sentences(normalized, max_chars)
