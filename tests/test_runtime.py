from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.runtime import (
    CLONE_MODEL_KIND,
    PROFILE_COLLECTION,
    STARTUP_MODEL_COLLECTION,
    STARTUP_MODEL_KEY,
    TTSRuntime,
    VOICE_DESIGN_MODEL_KIND,
)


class FakeModel:
    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> list[SimpleNamespace]:
        self.calls.append(dict(kwargs))
        return [
            SimpleNamespace(
                audio=np.array([0.0, 0.1, 0.2], dtype=np.float32),
                sample_rate=self.sample_rate,
            )
        ]


class FakeStateStore:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str | None], Any] = {}

    async def put(
        self,
        key: str,
        value: Any,
        *,
        collection: str | None = None,
        ttl: object | None = None,
    ) -> None:
        del ttl
        self.values[(key, collection)] = value

    async def get(
        self,
        key: str,
        *,
        collection: str | None = None,
        default: Any = None,
    ) -> Any:
        return self.values.get((key, collection), default)

    async def delete(self, key: str, *, collection: str | None = None) -> bool:
        return self.values.pop((key, collection), None) is not None


def make_runtime(tmp_path: Path) -> TTSRuntime:
    return TTSRuntime(
        voice_design_model_id="mlx/test-voice-design",
        clone_model_id="mlx/test-clone",
        enable_voice_design_model=True,
        enable_clone_model=True,
        playback_backend="null",
        state_dir=tmp_path / "state",
        tts_chunk_max_chars=24,
    )


def test_from_env_reads_mlx_audio_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SPEAK_TO_USER_ENABLE_VOICE_DESIGN_MODEL", "false")
    monkeypatch.setenv("SPEAK_TO_USER_ENABLE_CLONE_MODEL", "true")
    monkeypatch.setenv("SPEAK_TO_USER_VOICE_DESIGN_MODEL_ID", "mlx/voice")
    monkeypatch.setenv("SPEAK_TO_USER_CLONE_MODEL_ID", "mlx/clone")
    monkeypatch.setenv("SPEAK_TO_USER_PLAYBACK_BACKEND", "null")
    monkeypatch.setenv("SPEAK_TO_USER_STATE_DIR", str(tmp_path / "profiles"))
    monkeypatch.setenv("SPEAK_TO_USER_TTS_CHUNK_MAX_CHARS", "320")

    runtime = TTSRuntime.from_env()

    assert runtime.enable_voice_design_model is False
    assert runtime.enable_clone_model is True
    assert runtime.voice_design_model_id == "mlx/voice"
    assert runtime.clone_model_id == "mlx/clone"
    assert runtime.playback_backend == "null"
    assert runtime.tts_chunk_max_chars == 320
    assert runtime.state_dir == (tmp_path / "profiles").resolve()


def test_preload_uses_saved_startup_option(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    state_store.values[(STARTUP_MODEL_KEY, STARTUP_MODEL_COLLECTION)] = runtime.clone_model_id
    loaded_model_kinds: list[str] = []

    def fake_load_model(model_kind: str) -> FakeModel:
        loaded_model_kinds.append(model_kind)
        return FakeModel()

    monkeypatch.setattr(runtime, "_load_model_impl", fake_load_model)

    import asyncio

    asyncio.run(runtime.preload(state_store=state_store))

    assert loaded_model_kinds == [CLONE_MODEL_KIND]
    assert runtime.status()["startup_model_option"] == runtime.clone_model_id


def test_speak_text_uses_mlx_audio_generate_and_records_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    fake_model = FakeModel()
    runtime._models[VOICE_DESIGN_MODEL_KIND] = fake_model

    played_audio: list[tuple[np.ndarray, int]] = []
    def fake_play_audio(waveform: np.ndarray, sample_rate: int) -> None:
        played_audio.append((waveform, sample_rate))

    monkeypatch.setattr(runtime, "_play_audio", fake_play_audio)

    result = runtime.speak_text(
        chunks=["Hello there.", "General Kenobi."],
        voice_description="warm and calm",
        language="en",
    )

    assert result["result"] == "success"
    assert result["mode"] == "voice_design"
    assert result["chunk_count"] == 2
    assert result["speech_jobs_completed"] == 1
    assert played_audio[0][1] == 24000
    assert fake_model.calls[0]["instruct"] == "warm and calm"
    assert fake_model.calls[0]["lang_code"] == "en"


def test_speak_text_as_clone_passes_reference_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    fake_model = FakeModel()
    runtime._models[CLONE_MODEL_KIND] = fake_model
    monkeypatch.setattr(runtime, "_play_audio", lambda waveform, sample_rate: None)

    reference_path = tmp_path / "voice.wav"
    reference_path.write_bytes(b"fake")

    result = runtime.speak_text_as_clone(
        chunks=["Hello clone."],
        reference_audio_path=str(reference_path),
        reference_text="Hello clone.",
        language="en",
    )

    assert result["result"] == "success"
    assert result["mode"] == "clone"
    assert fake_model.calls[0]["ref_audio"] == str(reference_path.resolve())
    assert fake_model.calls[0]["ref_text"] == "Hello clone."


def test_profile_lifecycle_uses_state_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    runtime._models[VOICE_DESIGN_MODEL_KIND] = FakeModel()

    reference_path = tmp_path / "sample.wav"
    reference_waveform = np.array([0.0, 0.1, -0.1], dtype=np.float32)
    import soundfile as sf

    sf.write(reference_path, reference_waveform, 24000)

    import asyncio

    created = asyncio.run(
        runtime.generate_speech_profile(
            state_store=state_store,
            name="default-femme",
            reference_audio_path=str(reference_path),
            reference_text="sample",
        )
    )
    assert created["result"] == "success"

    listed = asyncio.run(runtime.list_speech_profiles(state_store=state_store))
    assert listed["profile_count"] == 1
    profiles = listed["profiles"]
    assert isinstance(profiles, list)
    assert profiles[0]["name"] == "default-femme"

    resolved = asyncio.run(
        runtime.resolve_profile(state_store=state_store, name="default-femme")
    )
    assert resolved.reference_text == "sample"

    deleted = asyncio.run(
        runtime.delete_speech_profile(state_store=state_store, name="default-femme")
    )
    assert deleted["deleted"] is True
    assert state_store.values.get(("default-femme", PROFILE_COLLECTION)) is None


def test_generate_speech_profile_from_voice_design_writes_seed_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    runtime._models[VOICE_DESIGN_MODEL_KIND] = FakeModel()

    import asyncio

    created = asyncio.run(
        runtime.generate_speech_profile_from_voice_design(
            state_store=state_store,
            name="seeded",
            text="A short seed clip.",
            voice_description="bright and supportive",
            language="en",
        )
    )

    profile = created["profile"]
    assert isinstance(profile, dict)
    profile_path = Path(profile["reference_audio_path"])
    assert profile_path.exists()
    assert profile["source_kind"] == "voice_design_seed"
    assert profile["voice_description"] == "bright and supportive"


def test_list_profiles_reads_filetree_metadata(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    runtime.state_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = runtime.state_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    profile = {
        "name": "saved",
        "model_id": runtime.clone_model_id,
        "reference_audio_path": str(tmp_path / "saved.wav"),
        "reference_text": "saved",
        "language": "en",
        "source_kind": "reference_audio",
        "created_at": "2026-03-29T00:00:00+00:00",
        "seed_text": None,
        "voice_description": None,
    }
    state_store.values[("saved", PROFILE_COLLECTION)] = profile
    (metadata_dir / "saved.json").write_text(
        json.dumps({"key": "saved", "collection": PROFILE_COLLECTION}),
        encoding="utf-8",
    )

    import asyncio

    listed = asyncio.run(runtime.list_speech_profiles(state_store=state_store))

    assert listed["profile_count"] == 1
    profiles = listed["profiles"]
    assert isinstance(profiles, list)
    assert profiles[0]["name"] == "saved"
