from __future__ import annotations

import datetime
import json
from typing import Any, cast

from fastmcp import Context

from app.runtime import TTSRuntime
from app.text_chunking import chunk_text_for_tts


# MARK: General Helpers

USAGE_GUIDE_TEXT = """speak-to-user usage guide

Use speak_text for normal narrated replies.
Use speak_text_as_clone for one-off cloning from a real local reference clip.
Use generate_speech_profile for a reusable profile from real reference audio.
Use generate_speech_profile_from_voice_design for a reusable profile
from a short synthetic seed clip.
Use speak_with_profile for repeated playback once a saved profile exists.
Use load_model and unload_model to manage resident models explicitly.
Use set_startup_model to persist which model or models preload on server startup.

Agent defaults:
- prefer language=\"en\" unless the caller needs another language
- check tts_status before assuming speech is broken
- load required models before retrying a model-gated tool
- remember playback is global and serial across clients
- prefer accurate reference_text when cloning from real audio
- prefer short seed text for generate_speech_profile_from_voice_design
  because it is not a long narration path
"""


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


def choose_speak_to_user_workflow_prompt() -> str:
    return (
        "Choose the smallest speak-to-user workflow that fits the job. "
        "Use speak_text for ordinary narration, "
        "speak_text_as_clone for one-off real-audio cloning, "
        "generate_speech_profile for reusable profiles from real reference audio, "
        "generate_speech_profile_from_voice_design for reusable profiles "
        "from a short synthetic seed, "
        "and speak_with_profile once a profile already exists. "
        "Use load_model or unload_model when you need explicit model residency control, "
        "and set_startup_model when you want startup preload behavior to persist. "
        "Check tts_status first if readiness or "
        "busy state is unclear."
    )


def guide_speech_profile_workflow_prompt() -> str:
    return (
        "For speech profiles, prefer real reference audio plus accurate "
        "reference_text when available. "
        "Use generate_speech_profile_from_voice_design only when you need "
        "the server to synthesize a short seed clip for profile creation. "
        "Keep seed text short and focused, then use speak_with_profile for "
        "longer playback after the profile exists. Recreate weak profiles "
        "from better source material instead "
        "of reusing poor seeds."
    )


def usage_guide_resource() -> str:
    return USAGE_GUIDE_TEXT


def status_resource(ctx: Context) -> str:
    return json.dumps(tts_status(ctx), indent=2, sort_keys=True)


async def speech_profiles_resource(ctx: Context) -> str:
    profiles = await list_speech_profiles(ctx)
    return json.dumps(profiles, indent=2, sort_keys=True)


# MARK: Tool Adapters

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


async def speak_text(
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
    await _ensure_required_models_loaded(
        ctx,
        model_kinds=["voice_design"],
        operation_name="speak_text",
    )

    return runtime.speak_text(
        chunks=chunk_text_for_tts(normalized_text),
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
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("text must not be empty")

    normalized_reference_audio_path = reference_audio_path.strip()
    if not normalized_reference_audio_path:
        raise ValueError("reference_audio_path must not be empty")
    await _ensure_required_models_loaded(
        ctx,
        model_kinds=["clone"],
        operation_name="speak_text_as_clone",
    )

    return runtime.speak_text_as_clone(
        chunks=chunk_text_for_tts(normalized_text),
        reference_audio_path=normalized_reference_audio_path,
        reference_text=reference_text,
        language=language,
    )


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
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("text must not be empty")
    await _ensure_required_models_loaded(
        ctx,
        model_kinds=["clone"],
        operation_name="speak_with_profile",
    )

    return await runtime.speak_with_profile(
        state_store=_state_store_from_context(ctx),
        name=name,
        chunks=chunk_text_for_tts(normalized_text),
        language=language,
    )
