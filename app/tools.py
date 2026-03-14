from __future__ import annotations

import asyncio
from collections import deque
import datetime
from typing import cast

from fastmcp import Context
from fastmcp.dependencies import Progress

from app.runtime import TTSRuntime
from app.text_chunking import chunk_filename_stem as chunk_filename_stem_for_tts
from app.text_chunking import chunk_text_for_tts


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
        chunk_filename_stem = chunk_filename_stem_for_tts(
            filename_stem,
            chunk_index,
            chunk_count,
        )

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
