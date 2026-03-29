from __future__ import annotations

# MARK: Imports

import asyncio
import datetime
from typing import Any, cast

from fastmcp import Context

from app.runtime import TTSRuntime


# MARK: Context Helpers

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


def _state_store_from_context(ctx: Context) -> object:
    return ctx.fastmcp._state_store


# MARK: Model Availability Helpers

async def _ensure_required_models_loaded(
    ctx: Context,
    *,
    model_kinds: list[str],
    operation_name: str,
) -> None:
    runtime = _runtime_from_context(ctx)
    missing_model_ids = runtime.missing_required_model_ids(model_kinds=model_kinds)
    if not missing_model_ids:
        return

    fallback_message = runtime.required_models_not_loaded_message(model_kinds=model_kinds)
    joined_model_ids = ", ".join(f"`{model_id}`" for model_id in missing_model_ids)
    model_noun = "model" if len(missing_model_ids) == 1 else "models"
    message = (
        f"`{operation_name}` requires {model_noun} {joined_model_ids}, "
        "but they are not loaded yet. Load them now and continue?"
    )

    try:
        elicitation_result = await ctx.elicit(
            message,
            cast(
                Any,
                {
                    "load": {"title": "Load required model(s)"},
                    "skip": {"title": "Return an error"},
                },
            ),
        )
    except Exception:
        raise RuntimeError(fallback_message) from None

    if getattr(elicitation_result, "action", None) == "accept" and getattr(
        elicitation_result, "data", None
    ) == "load":
        for model_id in missing_model_ids:
            runtime.load_model(model_kind=runtime.model_kind_for_id(model_id))
        return

    raise RuntimeError(fallback_message)


# MARK: Model Tools

def tts_status(ctx: Context) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    return runtime.status()


def load_model(
    ctx: Context,
    *,
    model_id: str,
) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    return runtime.load_model(model_kind=runtime.model_kind_for_id(model_id))


def unload_model(
    ctx: Context,
    *,
    model_id: str,
) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    return runtime.unload_model(model_kind=runtime.model_kind_for_id(model_id))


async def set_startup_model(
    ctx: Context,
    *,
    option: str,
) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    return await runtime.set_startup_model(
        state_store=_state_store_from_context(ctx),
        option=option,
    )


# MARK: Speech Tools

async def speak_text(
    ctx: Context,
    *,
    text: str,
    voice_description: str,
    language: str = "en",
) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    await _ensure_required_models_loaded(
        ctx,
        model_kinds=["voice_design"],
        operation_name="speak_text",
    )

    return await asyncio.to_thread(
        runtime.speak_text,
        text=text,
        voice_description=voice_description,
        language=language,
    )


async def speak_text_as_clone(
    ctx: Context,
    *,
    text: str,
    reference_audio_path: str,
    reference_text: str | None = None,
    language: str = "en",
) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    normalized_reference_audio_path = reference_audio_path.strip()
    if not normalized_reference_audio_path:
        raise ValueError("reference_audio_path must not be empty")
    await _ensure_required_models_loaded(
        ctx,
        model_kinds=["clone"],
        operation_name="speak_text_as_clone",
    )

    return await asyncio.to_thread(
        runtime.speak_text_as_clone,
        text=text,
        reference_audio_path=normalized_reference_audio_path,
        reference_text=reference_text,
        language=language,
    )


# MARK: Profile Tools

async def generate_speech_profile(
    ctx: Context,
    *,
    name: str,
    reference_audio_path: str,
    reference_text: str | None = None,
) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    await _ensure_required_models_loaded(
        ctx,
        model_kinds=["clone"],
        operation_name="generate_speech_profile",
    )
    return await runtime.generate_speech_profile(
        state_store=_state_store_from_context(ctx),
        name=name,
        reference_audio_path=reference_audio_path,
        reference_text=reference_text,
    )


async def generate_speech_profile_from_voice_design(
    ctx: Context,
    *,
    name: str,
    text: str,
    voice_description: str,
    language: str = "en",
) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("text must not be empty")
    normalized_voice_description = voice_description.strip()
    if not normalized_voice_description:
        raise ValueError("voice_description must not be empty")
    await _ensure_required_models_loaded(
        ctx,
        model_kinds=["voice_design", "clone"],
        operation_name="generate_speech_profile_from_voice_design",
    )

    return await runtime.generate_speech_profile_from_voice_design(
        state_store=_state_store_from_context(ctx),
        name=name,
        text=normalized_text,
        voice_description=normalized_voice_description,
        language=language,
    )


async def list_speech_profiles(ctx: Context) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    return await runtime.list_speech_profiles(
        state_store=_state_store_from_context(ctx),
    )


async def delete_speech_profile(
    ctx: Context,
    *,
    name: str,
) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    return await runtime.delete_speech_profile(
        state_store=_state_store_from_context(ctx),
        name=name,
    )


async def speak_with_profile(
    ctx: Context,
    *,
    name: str,
    text: str,
    language: str = "en",
) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    await _ensure_required_models_loaded(
        ctx,
        model_kinds=["clone"],
        operation_name="speak_with_profile",
    )
    return await runtime.speak_with_profile(
        state_store=_state_store_from_context(ctx),
        name=name,
        text=text,
        language=language,
    )
