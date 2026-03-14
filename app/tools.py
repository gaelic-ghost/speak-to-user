from __future__ import annotations

import asyncio
import datetime
from typing import cast

from fastmcp import Context
from fastmcp.dependencies import Progress

from app.runtime import TTSRuntime
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
) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    chunks = chunk_text_for_tts(text)
    if not chunks:
        raise ValueError("text must not be empty")

    chunk_count = len(chunks)
    total_steps = 3

    await progress.set_total(total_steps)
    await progress.set_message(f"Prepared {chunk_count} chunk(s) for queued speech playback")
    await progress.increment()

    await progress.set_message("Handing speech job to the local playback queue")
    queue_result = await asyncio.to_thread(
        runtime.enqueue_speech,
        chunks=chunks,
        voice_description=voice_description,
        language=language,
    )
    await progress.increment()

    await progress.set_message("Speech job queued")
    await progress.increment()
    return cast(dict[str, object], queue_result)
