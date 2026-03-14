from __future__ import annotations

import asyncio
from collections import deque
import datetime
import re
from typing import cast

from fastmcp import Context
from fastmcp.dependencies import Progress

from app.runtime import TTSRuntime

# MARK: Constants

DEFAULT_TTS_CHUNK_MAX_CHARS = 1200
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n+")
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+")


# MARK: General Helpers

def health_payload() -> dict[str, str]:
    return {
        "status": "ok",
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }


def _runtime_from_context(ctx: Context) -> TTSRuntime:
    runtime = ctx.lifespan_context.get("runtime")
    if runtime is None:
        raise RuntimeError("TTS runtime is unavailable in lifespan context")
    return cast(TTSRuntime, runtime)


def _append_chunk(chunks: list[str], value: str) -> None:
    normalized = value.strip()
    if normalized:
        chunks.append(normalized)


def _split_long_word(word: str, max_chars: int) -> list[str]:
    return [word[index : index + max_chars] for index in range(0, len(word), max_chars)]


def _chunk_words(text: str, max_chars: int) -> list[str]:
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
    sentences = [part.strip() for part in _SENTENCE_BREAK_RE.split(text.strip()) if part.strip()]
    if not sentences:
        return _chunk_words(text, max_chars)

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        _append_chunk(chunks, current)

        if len(sentence) > max_chars:
            chunks.extend(_chunk_words(sentence, max_chars))
            current = ""
        else:
            current = sentence

    _append_chunk(chunks, current)
    return chunks


def chunk_text_for_tts(text: str, max_chars: int = DEFAULT_TTS_CHUNK_MAX_CHARS) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if len(normalized) <= max_chars:
        return [normalized]

    paragraphs = [part.strip() for part in _PARAGRAPH_BREAK_RE.split(normalized) if part.strip()]
    if not paragraphs:
        return _chunk_sentences(normalized, max_chars)

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        _append_chunk(chunks, current)

        if len(paragraph) > max_chars:
            chunks.extend(_chunk_sentences(paragraph, max_chars))
            current = ""
        else:
            current = paragraph

    _append_chunk(chunks, current)
    return chunks


def _chunk_filename_stem(
    filename_stem: str | None,
    chunk_index: int,
    chunk_count: int,
) -> str | None:
    if filename_stem is None or chunk_count <= 1:
        return filename_stem
    return f"{filename_stem}-{chunk_index:02d}"


# MARK: Tool Adapters

def tts_status(ctx: Context) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    return runtime.status()


def load_model(ctx: Context) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    try:
        return runtime.load_model()
    except Exception as exc:
        return {"result": "error", "loaded": False, "error": str(exc), **runtime.status()}


def unload_model(ctx: Context) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    try:
        return runtime.unload_model()
    except Exception as exc:
        return {"result": "error", "loaded": True, "error": str(exc), **runtime.status()}


def set_idle_unload_timeout(ctx: Context, seconds: int) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    try:
        status = runtime.set_idle_unload_timeout(seconds)
    except Exception as exc:
        return {"result": "error", "error": str(exc), **runtime.status()}
    return {"result": "success", "info": "idle unload timeout updated", **status}


def generate_audio(
    ctx: Context,
    *,
    text: str,
    voice_description: str,
    language: str = "en",
    output_format: str = "wav",
    filename_stem: str | None = None,
) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    try:
        return runtime.generate_audio(
            text=text,
            voice_description=voice_description,
            language=language,
            output_format=output_format,
            filename_stem=filename_stem,
        )
    except Exception as exc:
        return {"result": "error", "error": str(exc), **runtime.status()}


async def speak_text(
    ctx: Context,
    progress: Progress,
    *,
    text: str,
    voice_description: str,
    language: str = "en",
    output_format: str = "wav",
    filename_stem: str | None = None,
) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    chunks = chunk_text_for_tts(text)
    if not chunks:
        raise ValueError("text must not be empty")

    chunk_queue = deque(chunks)
    chunk_count = len(chunks)
    generated_chunks: list[dict[str, object]] = []
    total_steps = 1 + (chunk_count * 2)

    await progress.set_total(total_steps)
    await progress.set_message(f"Queued {chunk_count} chunk(s) for speech generation")
    await progress.increment()

    chunk_index = 0
    while chunk_queue:
        chunk_index += 1
        chunk_text = chunk_queue.popleft()
        chunk_filename_stem = _chunk_filename_stem(filename_stem, chunk_index, chunk_count)

        await progress.set_message(f"Generating chunk {chunk_index} of {chunk_count}")
        generation_result = await asyncio.to_thread(
            runtime.generate_audio,
            text=chunk_text,
            voice_description=voice_description,
            language=language,
            output_format=output_format,
            filename_stem=chunk_filename_stem,
        )
        await progress.increment()

        await progress.set_message(f"Playing chunk {chunk_index} of {chunk_count}")
        playback_result = await asyncio.to_thread(runtime.play_audio, generation_result["path"])
        await progress.increment()

        generated_chunks.append(
            {
                "index": chunk_index,
                "text": chunk_text,
                "text_length": len(chunk_text),
                "path": generation_result["path"],
                "format": generation_result["format"],
                "sample_rate": generation_result["sample_rate"],
                "sample_count": generation_result["sample_count"],
                "duration_seconds": generation_result["duration_seconds"],
                "model_id": generation_result["model_id"],
                "device": generation_result["device"],
                "language": generation_result["language"],
                "played": playback_result["played"],
                "player": playback_result["player"],
            }
        )

    await progress.set_message("Playback complete")

    last_chunk = generated_chunks[-1]
    return {
        "result": "success",
        "chunked": chunk_count > 1,
        "chunk_count": chunk_count,
        "chunks": generated_chunks,
        "path": last_chunk["path"],
        "format": last_chunk["format"],
        "sample_rate": last_chunk["sample_rate"],
        "sample_count": last_chunk["sample_count"],
        "duration_seconds": last_chunk["duration_seconds"],
        "model_id": last_chunk["model_id"],
        "device": last_chunk["device"],
        "language": last_chunk["language"],
        "played": True,
        "player": last_chunk["player"],
    }
