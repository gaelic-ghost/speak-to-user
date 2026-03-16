from __future__ import annotations

import datetime
from typing import cast

from fastmcp import Context

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


def speak_text(
    ctx: Context,
    *,
    text: str,
    voice_description: str,
    language: str = "en",
) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("text must not be empty")

    return runtime.speak_text(
        chunks=chunk_text_for_tts(normalized_text),
        voice_description=voice_description,
        language=language,
    )


def speak_text_as_clone(
    ctx: Context,
    *,
    text: str,
    reference_audio_path: str,
    reference_text: str | None = None,
    language: str = "en",
) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("text must not be empty")

    normalized_reference_audio_path = reference_audio_path.strip()
    if not normalized_reference_audio_path:
        raise ValueError("reference_audio_path must not be empty")

    return runtime.speak_text_as_clone(
        chunks=chunk_text_for_tts(normalized_text),
        reference_audio_path=normalized_reference_audio_path,
        reference_text=reference_text,
        language=language,
    )
