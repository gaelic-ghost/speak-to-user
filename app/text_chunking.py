from __future__ import annotations

import re

# MARK: Constants

DEFAULT_TTS_CHUNK_MAX_CHARS = 160
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


def _find_sentence_break_before_limit(text: str, max_chars: int) -> tuple[int, int] | None:
    # Step 4: prefer the nearest sentence boundary at or before the character limit.
    window = text[: max_chars + 1]
    matches = list(_SENTENCE_BREAK_RE.finditer(window))
    if not matches:
        return None

    match = matches[-1]
    return (match.start(), match.end())


def _find_word_break_before_limit(text: str, max_chars: int) -> int | None:
    # Step 5: if no sentence boundary fits, fall back to the last whitespace before the limit.
    window = text[: max_chars + 1]
    for index in range(len(window) - 1, -1, -1):
        if window[index].isspace():
            return index
    return None


def _chunk_text_by_preferred_boundaries(text: str, max_chars: int) -> list[str]:
    # Step 6: fill chunks toward the character limit and break on the best available boundary.
    remaining = text.strip()
    if not remaining:
        return []

    chunks: list[str] = []

    while remaining:
        if len(remaining) <= max_chars:
            _append_chunk(chunks, remaining)
            break

        sentence_break = _find_sentence_break_before_limit(remaining, max_chars)
        if sentence_break is not None:
            chunk_end, next_start = sentence_break
            _append_chunk(chunks, remaining[:chunk_end])
            remaining = remaining[next_start:].lstrip()
            continue

        word_break = _find_word_break_before_limit(remaining, max_chars)
        if word_break is not None:
            _append_chunk(chunks, remaining[:word_break])
            remaining = remaining[word_break + 1 :].lstrip()
            continue

        _append_chunk(chunks, remaining[:max_chars])
        remaining = remaining[max_chars:].lstrip()

    return chunks


def chunk_text_for_tts(text: str, max_chars: int = DEFAULT_TTS_CHUNK_MAX_CHARS) -> list[str]:
    # Step 7: chunk near the character limit and prefer sentence boundaries before falling
    # back to word boundaries or hard splits.
    normalized = text.strip()
    if not normalized:
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    return _chunk_text_by_preferred_boundaries(normalized, max_chars)
