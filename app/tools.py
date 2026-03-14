import datetime
import asyncio
from typing import cast

from fastmcp import Context
from fastmcp.dependencies import Progress

from app.runtime import TTSRuntime


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

    await progress.set_total(3)
    await progress.set_message("Generating audio")
    generation_result = await asyncio.to_thread(
        runtime.generate_audio,
        text=text,
        voice_description=voice_description,
        language=language,
        output_format=output_format,
        filename_stem=filename_stem,
    )
    await progress.increment()

    audio_path = generation_result["path"]
    await progress.set_message("Playing audio locally")
    playback_result = await asyncio.to_thread(runtime.play_audio, audio_path)
    await progress.increment()

    await progress.set_message("Playback complete")
    await progress.increment()

    return {
        "result": "success",
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
