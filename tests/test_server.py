from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import server


class StubRuntime:
    def status(self) -> dict[str, object]:
        return {
            "ready": True,
            "playback_backend": "null",
            "voice_design_model_id": "mlx/test-voice-design",
            "clone_model_id": "mlx/test-clone",
            "speech_in_progress": False,
        }

    async def list_speech_profiles(self, *, state_store: object) -> dict[str, object]:
        del state_store
        return {
            "result": "success",
            "profile_count": 1,
            "profiles": [{"name": "default-femme"}],
        }


class StubStateStore:
    pass


class StubFastMCP:
    def __init__(self) -> None:
        self._state_store = StubStateStore()


class StubContext:
    def __init__(self) -> None:
        self.lifespan_context = {"runtime": StubRuntime()}
        self.fastmcp = StubFastMCP()


def test_server_config_from_env_normalizes_values(monkeypatch) -> None:
    monkeypatch.setenv("SPEAK_TO_USER_HOST", "127.0.0.1")
    monkeypatch.setenv("SPEAK_TO_USER_PORT", "8766")
    monkeypatch.setenv("SPEAK_TO_USER_MCP_PATH", "custom")
    monkeypatch.setenv("SPEAK_TO_USER_STATE_DIR", "~/tmp/speak-to-user-state")

    config = server.server_config_from_env()

    assert config.host == "127.0.0.1"
    assert config.port == 8766
    assert config.path == "/custom"
    assert config.state_dir == Path("~/tmp/speak-to-user-state").expanduser().resolve()


def test_usage_guide_mentions_profile_workflow() -> None:
    guide = server.usage_guide()

    assert "speak_text" in guide
    assert "speak_with_profile" in guide


def test_status_resource_returns_runtime_snapshot() -> None:
    payload = json.loads(server.status_resource(cast(Any, StubContext())))

    assert payload["ready"] is True
    assert payload["playback_backend"] == "null"


def test_speech_profiles_resource_returns_json() -> None:
    payload = json.loads(asyncio.run(server.speech_profiles_resource(cast(Any, StubContext()))))

    assert payload["profile_count"] == 1
    assert payload["profiles"][0]["name"] == "default-femme"
