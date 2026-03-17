from __future__ import annotations

import datetime
import json
from typing import cast

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

Agent defaults:
- prefer language=\"en\" unless the caller needs another language
- check tts_status before assuming speech is broken
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


def choose_speak_to_user_workflow_prompt() -> str:
    return (
        "Choose the smallest speak-to-user workflow that fits the job. "
        "Use speak_text for ordinary narration, "
        "speak_text_as_clone for one-off real-audio cloning, "
        "generate_speech_profile for reusable profiles from real reference audio, "
        "generate_speech_profile_from_voice_design for reusable profiles "
        "from a short synthetic seed, "
        "and speak_with_profile once a profile already exists. "
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


async def generate_speech_profile(
    ctx: Context,
    *,
    name: str,
    reference_audio_path: str,
    reference_text: str | None = None,
) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    return await runtime.generate_speech_profile(
        state_store=ctx.fastmcp._state_store,
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

    return await runtime.generate_speech_profile_from_voice_design(
        state_store=ctx.fastmcp._state_store,
        name=name,
        text=normalized_text,
        voice_description=normalized_voice_description,
        language=language,
    )


async def list_speech_profiles(ctx: Context) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    return await runtime.list_speech_profiles(
        state_store=ctx.fastmcp._state_store,
    )


async def delete_speech_profile(
    ctx: Context,
    *,
    name: str,
) -> dict[str, object]:
    runtime = _runtime_from_context(ctx)
    return await runtime.delete_speech_profile(
        state_store=ctx.fastmcp._state_store,
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

    return await runtime.speak_with_profile(
        state_store=ctx.fastmcp._state_store,
        name=name,
        chunks=chunk_text_for_tts(normalized_text),
        language=language,
    )
