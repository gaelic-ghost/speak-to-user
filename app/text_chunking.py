from __future__ import annotations

# MARK: Constants

DEFAULT_TTS_CHUNK_MAX_CHARS = 1200


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
    # Step 3: pack words into bounded chunks for the active word-level speech path.
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
def chunk_text_for_tts(text: str, max_chars: int = DEFAULT_TTS_CHUNK_MAX_CHARS) -> list[str]:
    # Step 4: always split through the word-level chain so every request is chunked
    # without waiting on sentence boundaries.
    normalized = text.strip()
    if not normalized:
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    return _chunk_words(normalized, max_chars)
