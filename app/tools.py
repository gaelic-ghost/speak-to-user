from __future__ import annotations

import datetime
from typing import cast

from fastmcp import Context

from app.runtime import TTSRuntime


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
        chunks=[normalized_text],
        voice_description=voice_description,
        language=language,
    )
