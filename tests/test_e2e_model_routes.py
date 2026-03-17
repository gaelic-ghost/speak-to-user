from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import uuid
from typing import Any, cast

import numpy as np
import pytest
import soundfile as sf
from fastmcp import Client


pytestmark = [pytest.mark.e2e, pytest.mark.serial]


def _e2e_base_url() -> str:
    base_url = os.getenv("SPEAK_TO_USER_E2E_BASE_URL", "").strip()
    if not base_url:
        pytest.skip("set SPEAK_TO_USER_E2E_BASE_URL to run the optional HTTP e2e suite")
    return base_url


def _tool_payload(result: Any) -> dict[str, Any]:
    payload = getattr(result, "structuredContent", None)
    if isinstance(payload, dict):
        return cast(dict[str, Any], payload)

    for content in getattr(result, "content", []):
        text = getattr(content, "text", None)
        if isinstance(text, str):
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return cast(dict[str, Any], parsed)

    raise AssertionError(f"expected structured tool payload, got {payload!r}")


@pytest.fixture
def reference_audio_path(tmp_path: Path) -> Path:
    sample_rate = 16000
    duration_seconds = 0.9
    frame_count = int(sample_rate * duration_seconds)
    timeline = np.linspace(0.0, duration_seconds, frame_count, endpoint=False, dtype=np.float32)
    waveform = (0.1 * np.sin(2.0 * np.pi * 220.0 * timeline)).astype(np.float32)
    path = tmp_path / "reference.wav"
    sf.write(path, waveform, sample_rate, subtype="PCM_16")
    return path


def test_model_generating_routes_over_http(reference_audio_path: Path) -> None:
    base_url = _e2e_base_url()

    async def run() -> None:
        profile_name = f"e2e-profile-{uuid.uuid4().hex[:8]}"
        designed_profile_name = f"e2e-designed-{uuid.uuid4().hex[:8]}"
        created_profiles: list[str] = []

        async with Client(base_url, timeout=600) as client:
            try:
                status_payload = _tool_payload(await client.call_tool("tts_status"))
                assert status_payload["ready"] is True
                assert status_payload["playback_backend"] == "null"

                speak_text_payload = _tool_payload(
                    await client.call_tool(
                        "speak_text",
                        {
                            "text": "This is a short HTTP e2e narration check.",
                            "voice_description": "Warm, clear, supportive, brisk.",
                            "language": "en",
                        },
                        timeout=600,
                    )
                )
                assert speak_text_payload["result"] == "success"
                assert speak_text_payload["queued"] is True

                speak_clone_payload = _tool_payload(
                    await client.call_tool(
                        "speak_text_as_clone",
                        {
                            "text": "This is a clone playback route check.",
                            "reference_audio_path": str(reference_audio_path),
                            "reference_text": "This is a short reference clip for clone testing.",
                            "language": "en",
                        },
                        timeout=600,
                    )
                )
                assert speak_clone_payload["result"] == "success"
                assert speak_clone_payload["queued"] is True

                generate_profile_payload = _tool_payload(
                    await client.call_tool(
                        "generate_speech_profile",
                        {
                            "name": profile_name,
                            "reference_audio_path": str(reference_audio_path),
                            "reference_text": "This is a short reference clip for clone testing.",
                        },
                        timeout=600,
                    )
                )
                assert generate_profile_payload["result"] == "success"
                created_profiles.append(profile_name)

                list_profiles_payload = _tool_payload(
                    await client.call_tool("list_speech_profiles", timeout=600)
                )
                assert any(
                    profile["name"] == profile_name
                    for profile in cast(list[dict[str, Any]], list_profiles_payload["profiles"])
                )

                voice_designed_payload = _tool_payload(
                    await client.call_tool(
                        "generate_speech_profile_from_voice_design",
                        {
                            "name": designed_profile_name,
                            "text": (
                                "Please create a short, bright, conversational "
                                "seed clip for this reusable profile."
                            ),
                            "voice_description": (
                                "Bright, expressive, articulate, and lightly playful."
                            ),
                            "language": "en",
                        },
                        timeout=600,
                    )
                )
                assert voice_designed_payload["result"] == "success"
                assert voice_designed_payload["profile_source"] == "voice_design"
                created_profiles.append(designed_profile_name)

                speak_with_profile_payload = _tool_payload(
                    await client.call_tool(
                        "speak_with_profile",
                        {
                            "name": profile_name,
                            "text": "This is a reusable profile playback route check.",
                            "language": "en",
                        },
                        timeout=600,
                    )
                )
                assert speak_with_profile_payload["result"] == "success"
                assert speak_with_profile_payload["queued"] is True

                profile_resource = await client.read_resource("state://speak-to-user/profiles")
                profile_resource_text = "\n".join(
                    content.text
                    for content in profile_resource
                    if hasattr(content, "text")
                )
                profile_resource_payload = json.loads(profile_resource_text)
                assert profile_resource_payload["profile_count"] >= 2
            finally:
                for created_profile in reversed(created_profiles):
                    await client.call_tool(
                        "delete_speech_profile",
                        {"name": created_profile},
                        timeout=600,
                    )

    asyncio.run(run())
