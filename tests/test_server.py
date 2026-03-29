from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from typing import Any, cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import server


class StubRuntime:
    def status(self) -> dict[str, object]:
        return {
            "ready": True,
            "profiles_dir": "/tmp/profiles",
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
    monkeypatch.setenv("SPEAK_TO_USER_VOICE_DESIGN_MODEL_ID", "mlx/voice")
    monkeypatch.setenv("SPEAK_TO_USER_CLONE_MODEL_ID", "mlx/clone")

    config = server.server_config_from_env()

    assert config.host == "127.0.0.1"
    assert config.port == 8766
    assert config.path == "/custom"
    assert config.state_dir == Path("~/tmp/speak-to-user-state").expanduser().resolve()
    assert config.voice_design_model_id == "mlx/voice"
    assert config.clone_model_id == "mlx/clone"


def test_normalize_http_port_rejects_non_numeric_values() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        server._normalize_http_port("not-a-port")


def test_normalize_http_port_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="between 1 and 65535"):
        server._normalize_http_port("70000")


def test_state_store_metadata_requires_rebuild_when_metadata_points_outside_data_dir(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    data_dir, metadata_dir = server._state_store_subdirectories(state_dir)
    data_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "speech_profiles-info.json").write_text(
        json.dumps(
            {
                "collection": "speech_profiles",
                "directory": str(tmp_path / "somewhere-else"),
            }
        ),
        encoding="utf-8",
    )

    assert server._state_store_metadata_requires_rebuild(state_dir) is True


def test_state_store_metadata_requires_rebuild_for_invalid_json(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    _, metadata_dir = server._state_store_subdirectories(state_dir)
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "speech_profiles-info.json").write_text("{bad", encoding="utf-8")

    assert server._state_store_metadata_requires_rebuild(state_dir) is True


def test_prepare_runtime_state_dir_rebuilds_invalid_metadata(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    _, metadata_dir = server._state_store_subdirectories(state_dir)
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "speech_profiles-info.json").write_text("{bad", encoding="utf-8")

    server._prepare_runtime_state_dir(state_dir)

    assert not metadata_dir.exists()


def test_resolved_runtime_state_dir_prefers_explicit_env_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    explicit_dir = tmp_path / "explicit-state"
    monkeypatch.setenv("SPEAK_TO_USER_STATE_DIR", str(explicit_dir))

    resolved = server._resolved_runtime_state_dir()

    assert resolved == explicit_dir.resolve()


def test_resolved_runtime_state_dir_copies_legacy_store_when_default_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    default_state_dir = tmp_path / "default-state"
    legacy_state_dir = tmp_path / "legacy-state"
    legacy_data_dir, _ = server._state_store_subdirectories(legacy_state_dir)
    legacy_data_dir.mkdir(parents=True)
    profile_dir = legacy_data_dir / "speech_profiles"
    profile_dir.mkdir(parents=True)
    (profile_dir / "saved.json").write_text(
        json.dumps({"value": {"name": "saved"}}),
        encoding="utf-8",
    )

    monkeypatch.delenv("SPEAK_TO_USER_STATE_DIR", raising=False)
    monkeypatch.setattr(server, "DEFAULT_STATE_DIR", default_state_dir)
    monkeypatch.setattr(server, "LEGACY_DEFAULT_STATE_DIR", legacy_state_dir)

    resolved = server._resolved_runtime_state_dir()

    copied_file = resolved / "data" / "speech_profiles" / "saved.json"
    assert copied_file.exists()


def test_resolved_runtime_state_dir_does_not_copy_legacy_when_explicit_dir_is_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    explicit_dir = tmp_path / "explicit-state"
    legacy_state_dir = tmp_path / "legacy-state"
    legacy_data_dir, _ = server._state_store_subdirectories(legacy_state_dir)
    legacy_data_dir.mkdir(parents=True)
    (legacy_data_dir / "speech_profiles").mkdir(parents=True)

    monkeypatch.setenv("SPEAK_TO_USER_STATE_DIR", str(explicit_dir))
    monkeypatch.setattr(server, "LEGACY_DEFAULT_STATE_DIR", legacy_state_dir)

    resolved = server._resolved_runtime_state_dir()

    assert resolved == explicit_dir.resolve()
    assert not (explicit_dir / "data").exists()


def test_usage_guide_mentions_profile_workflow() -> None:
    guide = server.usage_guide()

    assert "speak_text" in guide
    assert "speak_with_profile" in guide


def test_status_resource_returns_runtime_snapshot() -> None:
    payload = json.loads(server.status_resource(cast(Any, StubContext())))

    assert payload["ready"] is True
    assert payload["profiles_dir"] == "/tmp/profiles"


def test_speech_profiles_resource_returns_json() -> None:
    payload = json.loads(asyncio.run(server.speech_profiles_resource(cast(Any, StubContext()))))

    assert payload["profile_count"] == 1
    assert payload["profiles"][0]["name"] == "default-femme"
