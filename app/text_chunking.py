from __future__ import annotations

# MARK: Imports

import re


_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")


# MARK: Public API

def chunk_text_for_tts(text: str, *, max_chars: int) -> list[str]:
    normalized_text = " ".join(text.split())
    if not normalized_text:
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    sentences = _split_sentences(normalized_text)
    chunks: list[str] = []

    for sentence in sentences:
        if len(sentence) <= max_chars:
            if chunks:
                candidate = f"{chunks[-1]} {sentence}"
                if len(candidate) <= max_chars:
                    chunks[-1] = candidate
                    continue
            chunks.append(sentence)
            continue

        chunks.extend(_split_overlong_sentence(sentence, max_chars=max_chars))

    return chunks


# MARK: Helpers

def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_BOUNDARY_PATTERN.split(text) if part.strip()]


def _split_overlong_sentence(sentence: str, *, max_chars: int) -> list[str]:
    words = sentence.split()
    chunks: list[str] = []
    current_words: list[str] = []

    for word in words:
        if len(word) > max_chars:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
            chunks.extend(_split_long_word(word, max_chars=max_chars))
            continue

        candidate = " ".join([*current_words, word])
        if current_words and len(candidate) > max_chars:
            chunks.append(" ".join(current_words))
            current_words = [word]
            continue

        current_words.append(word)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def _split_long_word(word: str, *, max_chars: int) -> list[str]:
    return [word[index : index + max_chars] for index in range(0, len(word), max_chars)]
