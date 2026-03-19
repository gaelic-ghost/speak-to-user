from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Generator, cast
from urllib.parse import urlparse
import uuid

from fastmcp import Client
from fastmcp.exceptions import ToolError
import numpy as np
import pytest
import soundfile as sf


pytestmark = [pytest.mark.e2e, pytest.mark.serial]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_E2E_HOST = "127.0.0.1"
DEFAULT_E2E_PORT = 8876
DEFAULT_E2E_PATH = "/mcp"


def _e2e_host() -> str:
    return os.getenv("SPEAK_TO_USER_E2E_HOST", DEFAULT_E2E_HOST).strip() or DEFAULT_E2E_HOST


def _e2e_port() -> int:
    raw = os.getenv("SPEAK_TO_USER_E2E_PORT", str(DEFAULT_E2E_PORT)).strip()
    return int(raw) if raw else DEFAULT_E2E_PORT


def _e2e_path() -> str:
    raw = os.getenv("SPEAK_TO_USER_E2E_PATH", DEFAULT_E2E_PATH).strip()
    if not raw:
        return DEFAULT_E2E_PATH
    return raw if raw.startswith("/") else f"/{raw}"


def _e2e_base_url() -> str:
    configured = os.getenv("SPEAK_TO_USER_E2E_BASE_URL", "").strip()
    if configured:
        return configured
    return f"http://{_e2e_host()}:{_e2e_port()}{_e2e_path()}"


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


async def _call_tool_payload(
    client: Client,
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    timeout: int = 600,
) -> dict[str, Any]:
    return _tool_payload(await client.call_tool(name, arguments or {}, timeout=timeout))


async def _assert_tool_error(
    client: Client,
    name: str,
    arguments: dict[str, Any] | None,
    match: str,
    *,
    timeout: int = 600,
) -> None:
    with pytest.raises(ToolError, match=match):
        await client.call_tool(name, arguments or {}, timeout=timeout)


class ManagedE2EServer:
    def __init__(self, *, state_dir: Path) -> None:
        self.base_url = _e2e_base_url()
        parsed = urlparse(self.base_url)
        self.host = parsed.hostname or _e2e_host()
        self.port = parsed.port or _e2e_port()
        self.path = parsed.path or _e2e_path()
        self.state_dir = state_dir
        self._log_dir = Path(tempfile.mkdtemp(prefix="speak-to-user-e2e-logs."))
        self._stdout_path = self._log_dir / "server.stdout.log"
        self._stderr_path = self._log_dir / "server.stderr.log"
        self._stdout_handle: Any | None = None
        self._stderr_handle: Any | None = None
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("managed e2e server is already running")

        env = os.environ.copy()
        env.update(
            {
                "SPEAK_TO_USER_HOST": self.host,
                "SPEAK_TO_USER_PORT": str(self.port),
                "SPEAK_TO_USER_MCP_PATH": self.path,
                "SPEAK_TO_USER_PLAYBACK_BACKEND": "null",
                "SPEAK_TO_USER_STATE_DIR": str(self.state_dir),
                "SPEAK_TO_USER_ENABLE_VOICE_DESIGN_MODEL": "true",
                "SPEAK_TO_USER_ENABLE_CLONE_MODEL": "true",
            }
        )

        self._stdout_handle = self._stdout_path.open("w", encoding="utf-8")
        self._stderr_handle = self._stderr_path.open("w", encoding="utf-8")
        self._process = subprocess.Popen(
            [sys.executable, "app/server.py"],
            cwd=REPO_ROOT,
            env=env,
            stdout=self._stdout_handle,
            stderr=self._stderr_handle,
            text=True,
        )

    def stop(self) -> None:
        if self._process is None:
            return

        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        finally:
            self._process = None
            if self._stdout_handle is not None:
                self._stdout_handle.close()
                self._stdout_handle = None
            if self._stderr_handle is not None:
                self._stderr_handle.close()
                self._stderr_handle = None

    def restart(self) -> None:
        self.stop()
        self.start()

    def stderr_text(self) -> str:
        return self._stderr_path.read_text(encoding="utf-8") if self._stderr_path.exists() else ""

    async def wait_until_ready(self, *, timeout_seconds: int = 120) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_result = "unreachable"

        while time.monotonic() < deadline:
            if self._process is None:
                raise RuntimeError("managed e2e server is not running")
            if self._process.poll() is not None:
                raise RuntimeError(
                    "managed e2e server exited before becoming ready\n"
                    f"stderr:\n{self.stderr_text()}"
                )

            try:
                async with Client(self.base_url, timeout=30) as client:
                    payload = await _call_tool_payload(client, "tts_status", timeout=30)
            except Exception as exc:
                last_result = str(exc)
                await asyncio.sleep(2)
                continue

            if payload.get("ready") is True:
                return

            last_result = str(payload.get("speech_last_event") or "not-ready")
            await asyncio.sleep(2)

        raise RuntimeError(
            f"timed out waiting for {self.base_url} to report ready "
            f"({last_result})\nstderr:\n{self.stderr_text()}"
        )


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


@pytest.fixture
def managed_server(tmp_path: Path) -> Generator[ManagedE2EServer, None, None]:
    server = ManagedE2EServer(state_dir=tmp_path / "state")
    server.start()
    asyncio.run(server.wait_until_ready())
    try:
        yield server
    finally:
        server.stop()


def test_http_guidance_and_status_surface(managed_server: ManagedE2EServer) -> None:
    async def run() -> None:
        async with Client(managed_server.base_url, timeout=600) as client:
            prompts = await client.list_prompts()
            prompt_names = {prompt.name for prompt in prompts}
            assert "choose_speak_to_user_workflow" in prompt_names
            assert "guide_speech_profile_workflow" in prompt_names

            workflow_prompt = await client.get_prompt("choose_speak_to_user_workflow")
            workflow_text = "\n".join(
                getattr(message.content, "text", "")
                for message in workflow_prompt.messages
            )
            assert "load_model" in workflow_text
            assert "set_startup_model" in workflow_text

            profile_prompt = await client.get_prompt("guide_speech_profile_workflow")
            profile_text = "\n".join(
                getattr(message.content, "text", "")
                for message in profile_prompt.messages
            )
            assert "generate_speech_profile_from_voice_design" in profile_text

            usage_resource = await client.read_resource("guide://speak-to-user/usage")
            usage_text = "\n".join(
                content.text
                for content in usage_resource
                if hasattr(content, "text")
            )
            assert "load_model and unload_model" in usage_text
            assert "set_startup_model" in usage_text

            health_payload = await _call_tool_payload(client, "health")
            assert health_payload["status"] == "ok"

            status_payload = await _call_tool_payload(client, "tts_status")
            assert status_payload["ready"] is True
            assert status_payload["playback_backend"] == "null"
            assert status_payload["startup_model_option"] == "all"
            assert set(cast(list[str], status_payload["loaded_model_ids"])) == {
                cast(str, status_payload["voice_design_model_id"]),
                cast(str, status_payload["clone_model_id"]),
            }

            status_resource = await client.read_resource("state://speak-to-user/status")
            status_resource_text = "\n".join(
                content.text
                for content in status_resource
                if hasattr(content, "text")
            )
            status_resource_payload = json.loads(status_resource_text)
            assert status_resource_payload["startup_model_option"] == "all"

            profile_resource = await client.read_resource("state://speak-to-user/profiles")
            profile_resource_text = "\n".join(
                content.text
                for content in profile_resource
                if hasattr(content, "text")
            )
            profile_resource_payload = json.loads(profile_resource_text)
            assert profile_resource_payload["profile_count"] == 0

    asyncio.run(run())


def test_speech_and_profile_routes_over_http(
    managed_server: ManagedE2EServer,
    reference_audio_path: Path,
) -> None:
    async def run() -> None:
        profile_name = f"e2e-profile-{uuid.uuid4().hex[:8]}"
        designed_profile_name = f"e2e-designed-{uuid.uuid4().hex[:8]}"

        async with Client(managed_server.base_url, timeout=600) as client:
            try:
                speak_text_payload = await _call_tool_payload(
                    client,
                    "speak_text",
                    {
                        "text": "This is a short HTTP e2e narration check.",
                        "voice_description": "Warm, clear, supportive, brisk.",
                        "language": "en",
                    },
                )
                assert speak_text_payload["result"] == "success"
                assert speak_text_payload["queued"] is True

                speak_clone_payload = await _call_tool_payload(
                    client,
                    "speak_text_as_clone",
                    {
                        "text": "This is a clone playback route check.",
                        "reference_audio_path": str(reference_audio_path),
                        "reference_text": "This is a short reference clip for clone testing.",
                        "language": "en",
                    },
                )
                assert speak_clone_payload["result"] == "success"
                assert speak_clone_payload["queued"] is True

                generate_profile_payload = await _call_tool_payload(
                    client,
                    "generate_speech_profile",
                    {
                        "name": profile_name,
                        "reference_audio_path": str(reference_audio_path),
                        "reference_text": "This is a short reference clip for clone testing.",
                    },
                )
                assert generate_profile_payload["result"] == "success"

                list_profiles_payload = await _call_tool_payload(client, "list_speech_profiles")
                assert any(
                    profile["name"] == profile_name
                    for profile in cast(list[dict[str, Any]], list_profiles_payload["profiles"])
                )

                voice_designed_payload = await _call_tool_payload(
                    client,
                    "generate_speech_profile_from_voice_design",
                    {
                        "name": designed_profile_name,
                        "text": "Please create a short, bright, conversational seed clip.",
                        "voice_description": "Bright, expressive, articulate, and lightly playful.",
                        "language": "en",
                    },
                )
                assert voice_designed_payload["result"] == "success"
                assert voice_designed_payload["profile_source"] == "voice_design"

                speak_with_profile_payload = await _call_tool_payload(
                    client,
                    "speak_with_profile",
                    {
                        "name": profile_name,
                        "text": "This is a reusable profile playback route check.",
                        "language": "en",
                    },
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
                profile_names = {
                    profile["name"]
                    for profile in profile_resource_payload["profiles"]
                }
                assert {profile_name, designed_profile_name}.issubset(profile_names)
            finally:
                await _call_tool_payload(client, "delete_speech_profile", {"name": profile_name})
                await _call_tool_payload(
                    client,
                    "delete_speech_profile",
                    {"name": designed_profile_name},
                )

    asyncio.run(run())


def test_model_management_and_high_signal_failures_over_http(
    managed_server: ManagedE2EServer,
    reference_audio_path: Path,
) -> None:
    async def run() -> None:
        profile_name = f"e2e-dup-{uuid.uuid4().hex[:8]}"

        async with Client(managed_server.base_url, timeout=600) as client:
            clone_model_id = cast(
                str,
                (await _call_tool_payload(client, "tts_status"))["clone_model_id"],
            )
            voice_design_model_id = cast(
                str,
                (await _call_tool_payload(client, "tts_status"))["voice_design_model_id"],
            )

            unload_payload = await _call_tool_payload(
                client,
                "unload_model",
                {"model_id": voice_design_model_id},
            )
            assert unload_payload["result"] == "success"
            assert unload_payload["voice_design_model_loaded"] is False

            await _assert_tool_error(
                client,
                "speak_text",
                {
                    "text": "This should fail without the voice-design model.",
                    "voice_description": "Warm and clear.",
                },
                "required model",
            )

            load_payload = await _call_tool_payload(
                client,
                "load_model",
                {"model_id": voice_design_model_id},
            )
            assert load_payload["result"] == "success"
            assert load_payload["voice_design_model_loaded"] is True

            await _call_tool_payload(
                client,
                "generate_speech_profile",
                {
                    "name": profile_name,
                    "reference_audio_path": str(reference_audio_path),
                    "reference_text": "This is a short reference clip for clone testing.",
                },
            )
            await _assert_tool_error(
                client,
                "generate_speech_profile",
                {
                    "name": profile_name,
                    "reference_audio_path": str(reference_audio_path),
                },
                "already exists",
            )

            await _assert_tool_error(
                client,
                "speak_with_profile",
                {"name": "missing-profile", "text": "hello"},
                "does not exist",
            )
            await _assert_tool_error(
                client,
                "delete_speech_profile",
                {"name": "missing-profile"},
                "does not exist",
            )
            await _assert_tool_error(
                client,
                "set_startup_model",
                {"option": "definitely-not-valid"},
                "startup model option must be one of",
            )

            clone_unload_payload = await _call_tool_payload(
                client,
                "unload_model",
                {"model_id": clone_model_id},
            )
            assert clone_unload_payload["result"] == "success"
            await _call_tool_payload(client, "load_model", {"model_id": clone_model_id})

            await _call_tool_payload(client, "delete_speech_profile", {"name": profile_name})

    asyncio.run(run())


def test_unload_model_rejects_while_jobs_are_queued(managed_server: ManagedE2EServer) -> None:
    async def run() -> None:
        async with Client(managed_server.base_url, timeout=600) as client:
            status_payload = await _call_tool_payload(client, "tts_status")
            voice_design_model_id = cast(str, status_payload["voice_design_model_id"])
            long_text = " ".join("This is a queued e2e unload test sentence." for _ in range(300))

            for _ in range(6):
                await _call_tool_payload(
                    client,
                    "speak_text",
                    {
                        "text": long_text,
                        "voice_description": "Warm, clear, supportive, brisk.",
                        "language": "en",
                    },
                )

            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                queue_status = await _call_tool_payload(client, "tts_status")
                if (
                    cast(int, queue_status["speech_queue_depth"]) > 0
                    or queue_status["speech_in_progress"] is True
                ):
                    break
                await asyncio.sleep(0.2)

            await _assert_tool_error(
                client,
                "unload_model",
                {"model_id": voice_design_model_id},
                "cannot be unloaded",
            )

    asyncio.run(run())


def test_startup_model_and_profiles_persist_across_restart(
    managed_server: ManagedE2EServer,
    reference_audio_path: Path,
) -> None:
    async def run() -> None:
        profile_name = f"e2e-persist-{uuid.uuid4().hex[:8]}"

        async with Client(managed_server.base_url, timeout=600) as client:
            initial_status = await _call_tool_payload(client, "tts_status")
            clone_model_id = cast(str, initial_status["clone_model_id"])

            await _call_tool_payload(
                client,
                "generate_speech_profile",
                {
                    "name": profile_name,
                    "reference_audio_path": str(reference_audio_path),
                    "reference_text": "This is a short reference clip for clone testing.",
                },
            )
            await _call_tool_payload(client, "set_startup_model", {"option": "none"})

        managed_server.restart()
        await managed_server.wait_until_ready()

        async with Client(managed_server.base_url, timeout=600) as client:
            status_after_none = await _call_tool_payload(client, "tts_status")
            assert status_after_none["startup_model_option"] == "none"
            assert status_after_none["loaded_model_ids"] == []

            list_after_restart = await _call_tool_payload(client, "list_speech_profiles")
            assert any(
                profile["name"] == profile_name
                for profile in cast(list[dict[str, Any]], list_after_restart["profiles"])
            )

            await _call_tool_payload(client, "set_startup_model", {"option": clone_model_id})

        managed_server.restart()
        await managed_server.wait_until_ready()

        async with Client(managed_server.base_url, timeout=600) as client:
            status_after_clone = await _call_tool_payload(client, "tts_status")
            assert status_after_clone["startup_model_option"] == clone_model_id
            assert status_after_clone["loaded_model_ids"] == [clone_model_id]

            speak_with_profile_payload = await _call_tool_payload(
                client,
                "speak_with_profile",
                {
                    "name": profile_name,
                    "text": "This persisted profile should still be available after restart.",
                    "language": "en",
                },
            )
            assert speak_with_profile_payload["result"] == "success"

            await _call_tool_payload(client, "delete_speech_profile", {"name": profile_name})

    asyncio.run(run())
