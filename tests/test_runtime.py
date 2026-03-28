from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
from typing import Any, cast

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.runtime as runtime_module
from app.runtime import CLONE_MODEL_KIND, TTSRuntime, VOICE_DESIGN_MODEL_KIND
from fastmcp.server.server import StateValue


class FakeVoiceDesignModel:
    def __init__(self, *, waveform: list[float] | None = None, sample_rate: int = 24000) -> None:
        self.waveform = waveform or [0.0, 0.1, 0.2]
        self.sample_rate = sample_rate
        self.calls: list[dict[str, object]] = []

    def generate_voice_design(
        self,
        *,
        text: str | list[str],
        language: str | list[str],
        instruct: str | list[str],
        max_new_tokens: int,
        non_streaming_mode: bool,
    ) -> tuple[list[list[float]], int]:
        self.calls.append(
            {
                "text": text,
                "language": language,
                "instruct": instruct,
                "max_new_tokens": max_new_tokens,
                "non_streaming_mode": non_streaming_mode,
            }
        )
        batch_size = len(text) if isinstance(text, list) else 1
        return [list(self.waveform) for _ in range(batch_size)], self.sample_rate


class FakeCloneModel:
    def __init__(self, *, waveform: list[float] | None = None, sample_rate: int = 24000) -> None:
        self.waveform = waveform or [0.3, 0.4, 0.5]
        self.sample_rate = sample_rate
        self.calls: list[dict[str, object]] = []

    def generate_voice_clone(self, **kwargs: object) -> tuple[list[list[float]], int]:
        self.calls.append(dict(kwargs))
        return [list(self.waveform)], self.sample_rate

    def create_voice_clone_prompt(self, **kwargs: object) -> list[SimpleNamespace]:
        self.calls.append(dict(kwargs))
        return [
            SimpleNamespace(
                ref_code=np.array([[1, 2], [3, 4]], dtype=np.int64),
                ref_spk_embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
                x_vector_only_mode=bool(kwargs["x_vector_only_mode"]),
                icl_mode=not bool(kwargs["x_vector_only_mode"]),
                ref_text=kwargs.get("ref_text"),
            )
        ]


class FakeStateStore:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str | None], StateValue] = {}

    async def put(
        self,
        key: str,
        value: StateValue,
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
        default: StateValue | None = None,
    ) -> StateValue | None:
        return self.values.get((key, collection), default)

    async def delete(self, key: str, *, collection: str | None = None) -> bool:
        return self.values.pop((key, collection), None) is not None


def make_runtime(tmp_path: Path) -> TTSRuntime:
    del tmp_path
    return TTSRuntime(
        voice_design_model_id="Qwen/test-voice-design",
        clone_model_id="Qwen/test-clone",
        enable_voice_design_model=True,
        enable_clone_model=True,
        device_preference="cpu",
    )


@pytest.fixture(autouse=True)
def runtime_dependencies_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_module, "_runtime_dependencies_ready", lambda: True)
    monkeypatch.setattr(
        runtime_module,
        "_runtime_dependency_status",
        lambda: {
            "sox_available": True,
            "qwen_tts_available": True,
            "torch_available": True,
            "torchaudio_available": True,
            "soundfile_available": True,
        },
    )


# MARK: Module Helpers


def test_stdout_suppression_uses_module_level_safe_print() -> None:
    original_print = runtime_module.builtins.print

    with runtime_module._suppress_default_stdout_prints_for_current_thread():
        assert runtime_module.builtins.print is runtime_module.safe_print

    assert runtime_module.builtins.print is original_print


# MARK: Config


def test_from_env_reads_dual_model_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEAK_TO_USER_ENABLE_VOICE_DESIGN_MODEL", "false")
    monkeypatch.setenv("SPEAK_TO_USER_ENABLE_CLONE_MODEL", "true")
    monkeypatch.setenv("SPEAK_TO_USER_VOICE_DESIGN_MODEL_ID", "Qwen/voice-design")
    monkeypatch.setenv("SPEAK_TO_USER_CLONE_MODEL_ID", "Qwen/clone")

    runtime = TTSRuntime.from_env()

    assert runtime.enable_voice_design_model is False
    assert runtime.enable_clone_model is True
    assert runtime.voice_design_model_id == "Qwen/voice-design"
    assert runtime.clone_model_id == "Qwen/clone"


def test_from_env_reads_wavbuffer_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEAK_TO_USER_PLAYBACK_BACKEND", "wavbuffer")
    monkeypatch.setenv("SPEAK_TO_USER_WAVBUFFER_BINARY_PATH", "/tmp/wavbuffer")
    monkeypatch.setenv("SPEAK_TO_USER_WAVBUFFER_QUEUE_DEPTH", "4")
    monkeypatch.setenv("SPEAK_TO_USER_WAVBUFFER_PREROLL_MODE", "buffers")

    runtime = TTSRuntime.from_env()

    assert runtime.playback_backend == "wavbuffer"
    assert runtime.wavbuffer_binary_path == "/tmp/wavbuffer"
    assert runtime.wavbuffer_queue_depth == 4
    assert runtime.wavbuffer_preroll_mode == "buffers"


def test_from_env_reads_tts_chunk_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEAK_TO_USER_TTS_CHUNK_MAX_CHARS", "320")
    monkeypatch.setenv("SPEAK_TO_USER_TTS_MAX_NEW_TOKENS", "512")
    monkeypatch.setenv("SPEAK_TO_USER_TTS_MAX_CHUNK_SYNTH_SECONDS", "45")
    monkeypatch.setenv("SPEAK_TO_USER_TTS_MAX_CHUNK_AUDIO_SECONDS", "18")

    runtime = TTSRuntime.from_env()

    assert runtime.tts_chunk_max_chars == 320
    assert runtime.tts_max_new_tokens == 512
    assert runtime.tts_max_chunk_synth_seconds == 45.0
    assert runtime.tts_max_chunk_audio_seconds == 18.0


def test_from_env_reads_null_playback_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEAK_TO_USER_PLAYBACK_BACKEND", "null")

    runtime = TTSRuntime.from_env()

    assert runtime.playback_backend == "null"


def test_from_env_defaults_wavbuffer_binary_to_bundled_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPEAK_TO_USER_WAVBUFFER_BINARY_PATH", raising=False)

    runtime = TTSRuntime.from_env()

    expected_path = (
        Path(runtime_module.__file__).resolve().parent
        / "vendor"
        / "wavbuffer"
        / "macos-arm64"
        / "wavbuffer"
    )
    assert runtime.wavbuffer_binary_path == str(expected_path)


# MARK: Runtime Lifecycle


def test_load_voice_design_model_success_sets_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    fake_model = FakeVoiceDesignModel()
    monkeypatch.setattr(runtime, "_load_model_impl", lambda model_kind: fake_model)

    result = runtime.load_model(model_kind=VOICE_DESIGN_MODEL_KIND)

    assert result["result"] == "success"
    assert result["loaded"] is True
    status = runtime.status()
    assert status["voice_design_model_loaded"] is True
    assert status["model_loaded"] is True
    assert status["voice_design_last_error"] is None


def test_load_clone_model_success_sets_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    fake_model = FakeCloneModel()
    monkeypatch.setattr(runtime, "_load_model_impl", lambda model_kind: fake_model)

    result = runtime.load_model(model_kind=CLONE_MODEL_KIND)

    assert result["result"] == "success"
    assert result["loaded"] is True
    status = runtime.status()
    assert status["clone_model_loaded"] is True
    assert status["clone_last_error"] is None


def test_unload_model_clears_loaded_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    fake_model = FakeVoiceDesignModel()
    released: list[tuple[object | None, str | None]] = []
    runtime._model_state(VOICE_DESIGN_MODEL_KIND)["model"] = fake_model
    runtime._model_state(VOICE_DESIGN_MODEL_KIND)["resolved_device"] = "cpu"
    monkeypatch.setattr(
        runtime,
        "_release_model_resources",
        lambda model, *, resolved_device: released.append((model, resolved_device)),
    )

    result = runtime.unload_model(model_kind=VOICE_DESIGN_MODEL_KIND)

    assert result["result"] == "success"
    assert result["unloaded"] is True
    assert runtime.status()["voice_design_model_loaded"] is False
    assert released == [(fake_model, "cpu")]


def test_unload_model_rejects_when_jobs_are_queued(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    runtime._model_state(VOICE_DESIGN_MODEL_KIND)["model"] = FakeVoiceDesignModel()
    runtime._model_state(VOICE_DESIGN_MODEL_KIND)["resolved_device"] = "cpu"
    runtime._queued_jobs_by_mode[VOICE_DESIGN_MODEL_KIND] = 1

    with pytest.raises(RuntimeError, match="queued speech jobs"):
        runtime.unload_model(model_kind=VOICE_DESIGN_MODEL_KIND)


def test_preload_starts_worker_and_loads_enabled_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    calls: list[str] = []

    def fake_start_speech_worker() -> bool:
        calls.append("worker")
        return True

    monkeypatch.setattr(runtime, "start_speech_worker", fake_start_speech_worker)

    def fake_load_model(*, model_kind: str = VOICE_DESIGN_MODEL_KIND) -> dict[str, object]:
        calls.append(model_kind)
        return {"result": "success", "loaded": True}

    monkeypatch.setattr(runtime, "load_model", fake_load_model)
    monkeypatch.setattr(runtime, "status", lambda: {"ready": True})

    result = asyncio.run(runtime.preload(state_store=state_store))

    assert result["result"] == "success"
    assert calls == ["worker", VOICE_DESIGN_MODEL_KIND, CLONE_MODEL_KIND]


def test_preload_uses_persisted_startup_model_option(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    asyncio.run(
        state_store.put(
            runtime_module.STARTUP_MODEL_OPTION_KEY,
            StateValue(value=runtime.clone_model_id),
            collection=runtime_module.RUNTIME_CONFIGURATION_COLLECTION,
        )
    )
    loaded_model_kinds: list[str] = []

    monkeypatch.setattr(runtime, "start_speech_worker", lambda: True)

    def fake_load_model(
        *,
        model_kind: str = VOICE_DESIGN_MODEL_KIND,
    ) -> dict[str, object]:
        loaded_model_kinds.append(model_kind)
        return {"result": "success", "loaded": True}

    monkeypatch.setattr(
        runtime,
        "load_model",
        fake_load_model,
    )
    monkeypatch.setattr(runtime, "status", lambda: {"ready": True})

    result = asyncio.run(runtime.preload(state_store=state_store))

    assert result["startup_model_option"] == runtime.clone_model_id
    assert loaded_model_kinds == [CLONE_MODEL_KIND]


def test_set_startup_model_persists_option_and_updates_status(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()

    result = asyncio.run(
        runtime.set_startup_model(
            state_store=state_store,
            option=runtime.voice_design_model_id,
        )
    )

    stored_value = asyncio.run(
        state_store.get(
            runtime_module.STARTUP_MODEL_OPTION_KEY,
            collection=runtime_module.RUNTIME_CONFIGURATION_COLLECTION,
        )
    )
    assert stored_value is not None
    assert stored_value.value == runtime.voice_design_model_id
    assert result["startup_model_option"] == runtime.voice_design_model_id
    assert result["voice_design_model_startup_enabled"] is True
    assert result["clone_model_startup_enabled"] is False


def test_shutdown_releases_loaded_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    runtime._model_state(VOICE_DESIGN_MODEL_KIND)["model"] = FakeVoiceDesignModel()
    runtime._model_state(VOICE_DESIGN_MODEL_KIND)["resolved_device"] = "cpu"
    runtime._model_state(CLONE_MODEL_KIND)["model"] = FakeCloneModel()
    runtime._model_state(CLONE_MODEL_KIND)["resolved_device"] = "cpu"
    released: list[tuple[object | None, str | None]] = []
    monkeypatch.setattr(
        runtime,
        "_release_model_resources",
        lambda model, *, resolved_device: released.append((model, resolved_device)),
    )

    runtime.shutdown()

    status = runtime.status()
    assert status["voice_design_model_loaded"] is False
    assert status["clone_model_loaded"] is False
    assert len(released) == 2


# MARK: Queue


def test_speak_text_enqueues_one_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    runtime._model_state(VOICE_DESIGN_MODEL_KIND)["model"] = FakeVoiceDesignModel()
    runtime._model_state(VOICE_DESIGN_MODEL_KIND)["resolved_device"] = "cpu"
    worker_starts = {"value": 0}

    def fake_start_speech_worker() -> bool:
        worker_starts["value"] += 1
        return True

    monkeypatch.setattr(runtime, "start_speech_worker", fake_start_speech_worker)

    result = runtime.speak_text(
        chunks=["Hello there"],
        voice_description="Warm and calm",
        language="en",
    )

    assert result["result"] == "success"
    assert result["queued"] is True
    assert result["chunk_count"] == 1
    assert result["mode"] == VOICE_DESIGN_MODEL_KIND
    assert result["speech_jobs_queued"] == 1
    assert worker_starts["value"] == 1


def test_speak_text_as_clone_enqueues_one_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    runtime._model_state(CLONE_MODEL_KIND)["model"] = FakeCloneModel()
    runtime._model_state(CLONE_MODEL_KIND)["resolved_device"] = "cpu"
    monkeypatch.setattr(runtime, "start_speech_worker", lambda: True)
    reference_audio = (np.array([0.0, 0.1], dtype=np.float32), 16000)
    monkeypatch.setattr(runtime, "_decode_reference_audio_file", lambda path: reference_audio)

    result = runtime.speak_text_as_clone(
        chunks=["Hello there"],
        reference_audio_path="voice.wav",
        reference_text="Reference text",
        language="en",
    )

    assert result["result"] == "success"
    assert result["mode"] == CLONE_MODEL_KIND
    assert result["clone_mode"] == "reference_text"
    job = cast(dict[str, object], runtime._speech_queue.get_nowait())
    assert job["mode"] == CLONE_MODEL_KIND
    assert job["reference_audio"] == reference_audio
    assert job["reference_text"] == "Reference text"


def test_generate_speech_profile_persists_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    fake_model = FakeCloneModel()
    runtime._model_state(CLONE_MODEL_KIND)["model"] = fake_model
    runtime._model_state(CLONE_MODEL_KIND)["resolved_device"] = "cpu"
    monkeypatch.setattr(
        runtime,
        "_decode_reference_audio_file",
        lambda path: (np.array([0.0, 0.1], dtype=np.float32), 16000),
    )

    result = asyncio.run(
        runtime.generate_speech_profile(
            state_store=state_store,
            name="demo",
            reference_audio_path="voice.wav",
            reference_text="reference text",
        )
    )

    assert result["result"] == "success"
    stored_profile = asyncio.run(
        state_store.get(
            "profile:demo",
            collection=runtime_module.SPEECH_PROFILE_COLLECTION,
        )
    )
    assert stored_profile is not None
    profile_payload = cast(dict[str, object], stored_profile.value)
    assert profile_payload["clone_model_id"] == "Qwen/test-clone"
    assert profile_payload["clone_mode"] == "reference_text"
    assert profile_payload["profile_source"] is None
    assert profile_payload["seed_text"] is None
    assert profile_payload["voice_description"] is None


def test_generate_speech_profile_rejects_duplicate_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    fake_model = FakeCloneModel()
    runtime._model_state(CLONE_MODEL_KIND)["model"] = fake_model
    runtime._model_state(CLONE_MODEL_KIND)["resolved_device"] = "cpu"
    monkeypatch.setattr(
        runtime,
        "_decode_reference_audio_file",
        lambda path: (np.array([0.0, 0.1], dtype=np.float32), 16000),
    )

    asyncio.run(
        runtime.generate_speech_profile(
            state_store=state_store,
            name="demo",
            reference_audio_path="voice.wav",
        )
    )

    with pytest.raises(RuntimeError, match="already exists"):
        asyncio.run(
            runtime.generate_speech_profile(
                state_store=state_store,
                name="demo",
                reference_audio_path="voice.wav",
            )
        )


def test_generate_speech_profile_from_voice_design_persists_seed_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    fake_clone_model = FakeCloneModel()
    runtime._model_state(CLONE_MODEL_KIND)["model"] = fake_clone_model
    runtime._model_state(CLONE_MODEL_KIND)["resolved_device"] = "cpu"
    temp_wav_path = tmp_path / "generated.wav"
    temp_wav_path.write_bytes(b"stub")

    monkeypatch.setattr(
        runtime,
        "_synthesize_voice_design_reference_audio",
        lambda **kwargs: (np.array([0.0, 0.1, 0.2], dtype=np.float32), 24000),
    )
    monkeypatch.setattr(
        runtime,
        "_write_temp_reference_audio_file",
        lambda **kwargs: str(temp_wav_path),
    )
    monkeypatch.setattr(
        runtime,
        "_decode_reference_audio_file",
        lambda path: (np.array([0.0, 0.1, 0.2], dtype=np.float32), 24000),
    )

    result = asyncio.run(
        runtime.generate_speech_profile_from_voice_design(
            state_store=state_store,
            name="demo",
            text="hello there",
            voice_description="warm and bright",
            language="en",
        )
    )

    assert result["result"] == "success"
    assert result["profile_source"] == "voice_design"
    assert result["seed_text_stored"] is True
    assert result["voice_description_stored"] is True
    stored_profile = asyncio.run(
        state_store.get(
            "profile:demo",
            collection=runtime_module.SPEECH_PROFILE_COLLECTION,
        )
    )
    assert stored_profile is not None
    profile_payload = cast(dict[str, object], stored_profile.value)
    assert profile_payload["seed_text"] == "hello there"
    assert profile_payload["voice_description"] == "warm and bright"
    assert profile_payload["profile_source"] == "voice_design"
    assert not temp_wav_path.exists()


def test_generate_speech_profile_from_voice_design_passes_one_short_seed_chunk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    fake_clone_model = FakeCloneModel()
    runtime._model_state(CLONE_MODEL_KIND)["model"] = fake_clone_model
    runtime._model_state(CLONE_MODEL_KIND)["resolved_device"] = "cpu"
    captured_chunks: list[str] = []
    temp_wav_path = tmp_path / "generated.wav"
    temp_wav_path.write_bytes(b"stub")

    def fake_synthesize(**kwargs: object) -> tuple[np.ndarray, int]:
        captured_chunks.extend(cast(list[str], kwargs["chunks"]))
        return np.array([0.0, 0.1, 0.2], dtype=np.float32), 24000

    monkeypatch.setattr(runtime, "_synthesize_voice_design_reference_audio", fake_synthesize)
    monkeypatch.setattr(
        runtime,
        "_write_temp_reference_audio_file",
        lambda **kwargs: str(temp_wav_path),
    )
    monkeypatch.setattr(
        runtime,
        "_decode_reference_audio_file",
        lambda path: (np.array([0.0, 0.1, 0.2], dtype=np.float32), 24000),
    )

    text = "First sentence. Second sentence. Third sentence."
    asyncio.run(
        runtime.generate_speech_profile_from_voice_design(
            state_store=state_store,
            name="demo",
            text=text,
            voice_description="warm and bright",
            language="en",
        )
    )

    assert captured_chunks == [text]


def test_generate_speech_profile_from_voice_design_rejects_overlong_seed_text(
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    overlong_text = "a" * (runtime_module.VOICE_DESIGN_PROFILE_SEED_MAX_CHARS + 1)

    with pytest.raises(
        ValueError,
        match="text must be 240 characters or fewer for voice-designed profile seeds",
    ):
        asyncio.run(
            runtime.generate_speech_profile_from_voice_design(
                state_store=state_store,
                name="demo",
                text=overlong_text,
                voice_description="warm and bright",
                language="en",
            )
        )


def test_generate_speech_profile_from_voice_design_deletes_temp_wav_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    temp_wav_path = tmp_path / "generated.wav"
    temp_wav_path.write_bytes(b"stub")

    monkeypatch.setattr(
        runtime,
        "_synthesize_voice_design_reference_audio",
        lambda **kwargs: (np.array([0.0, 0.1, 0.2], dtype=np.float32), 24000),
    )
    monkeypatch.setattr(
        runtime,
        "_write_temp_reference_audio_file",
        lambda **kwargs: str(temp_wav_path),
    )
    monkeypatch.setattr(
        runtime,
        "_decode_reference_audio_file",
        lambda path: (_ for _ in ()).throw(RuntimeError("decode failed")),
    )

    with pytest.raises(RuntimeError, match="decode failed"):
        asyncio.run(
            runtime.generate_speech_profile_from_voice_design(
                state_store=state_store,
                name="demo",
                text="hello there",
                voice_description="warm and bright",
                language="en",
            )
        )

    assert not temp_wav_path.exists()


def test_list_and_delete_speech_profiles(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    profile = runtime_module.StoredSpeechProfile(
        name="demo",
        clone_model_id="Qwen/test-clone",
        clone_mode="x_vector_only",
        created_at="2026-03-14T00:00:00+00:00",
        updated_at="2026-03-14T00:00:00+00:00",
        prompt_items=[
            runtime_module.StoredSpeechProfilePromptItem(
                ref_code=None,
                ref_spk_embedding=runtime_module.SerializedTensor(
                    dtype="float32",
                    data=[0.1, 0.2],
                ),
                x_vector_only_mode=True,
                icl_mode=False,
                ref_text=None,
            )
        ],
    )
    asyncio.run(runtime._save_speech_profile(state_store=state_store, profile=profile))
    asyncio.run(runtime._save_speech_profile_names(state_store, ["demo"]))

    list_result = asyncio.run(runtime.list_speech_profiles(state_store=state_store))
    delete_result = asyncio.run(runtime.delete_speech_profile(state_store=state_store, name="demo"))

    assert list_result["profile_count"] == 1
    assert delete_result["deleted"] is True
    assert asyncio.run(
        state_store.get(
            "profile:demo",
            collection=runtime_module.SPEECH_PROFILE_COLLECTION,
        )
    ) is None


def test_speak_with_profile_enqueues_clone_job(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    runtime._model_state(CLONE_MODEL_KIND)["model"] = FakeCloneModel()
    runtime._model_state(CLONE_MODEL_KIND)["resolved_device"] = "cpu"
    profile = runtime_module.StoredSpeechProfile(
        name="demo",
        clone_model_id="Qwen/test-clone",
        clone_mode="x_vector_only",
        created_at="2026-03-14T00:00:00+00:00",
        updated_at="2026-03-14T00:00:00+00:00",
        prompt_items=[
            runtime_module.StoredSpeechProfilePromptItem(
                ref_code=None,
                ref_spk_embedding=runtime_module.SerializedTensor(
                    dtype="float32",
                    data=[0.1, 0.2],
                ),
                x_vector_only_mode=True,
                icl_mode=False,
                ref_text=None,
            )
        ],
    )
    asyncio.run(runtime._save_speech_profile(state_store=state_store, profile=profile))
    asyncio.run(runtime._save_speech_profile_names(state_store, ["demo"]))
    runtime.start_speech_worker = lambda: False  # type: ignore[method-assign]

    result = asyncio.run(
        runtime.speak_with_profile(
            state_store=state_store,
            name="demo",
            chunks=["Hello there"],
            language="en",
        )
    )

    assert result["result"] == "success"
    job = cast(dict[str, object], runtime._speech_queue.get_nowait())
    assert job["mode"] == CLONE_MODEL_KIND
    assert job["profile_name"] == "demo"
    assert "voice_clone_prompt_items" in job


def test_speak_text_assigns_unique_job_ids_under_concurrent_enqueue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    runtime._model_state(VOICE_DESIGN_MODEL_KIND)["model"] = FakeVoiceDesignModel()
    runtime._model_state(VOICE_DESIGN_MODEL_KIND)["resolved_device"] = "cpu"
    monkeypatch.setattr(runtime, "start_speech_worker", lambda: True)

    real_put_nowait = runtime._speech_queue.put_nowait
    first_put_started = threading.Event()
    allow_first_put_to_finish = threading.Event()

    def controlled_put_nowait(job: dict[str, object] | None) -> None:
        if (
            job is not None
            and cast(int, job["job_id"]) == 1
            and not first_put_started.is_set()
        ):
            first_put_started.set()
            allow_first_put_to_finish.wait(timeout=1)
        real_put_nowait(job)

    monkeypatch.setattr(runtime._speech_queue, "put_nowait", controlled_put_nowait)

    results: list[dict[str, object]] = [{}, {}]

    def enqueue(index: int) -> None:
        results[index] = runtime.speak_text(
            chunks=[f"chunk {index}"],
            voice_description="Warm and calm",
            language="en",
        )

    first_thread = threading.Thread(target=enqueue, args=(0,))
    second_thread = threading.Thread(target=enqueue, args=(1,))

    first_thread.start()
    first_put_started.wait(timeout=1)
    second_thread.start()
    allow_first_put_to_finish.set()
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)

    assert [cast(int, result["job_id"]) for result in results] == [1, 2]


def test_speech_worker_dispatches_modes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    runtime._model_state(VOICE_DESIGN_MODEL_KIND)["model"] = FakeVoiceDesignModel()
    runtime._model_state(VOICE_DESIGN_MODEL_KIND)["resolved_device"] = "cpu"
    runtime._model_state(CLONE_MODEL_KIND)["model"] = FakeCloneModel()
    runtime._model_state(CLONE_MODEL_KIND)["resolved_device"] = "cpu"
    played_jobs: list[dict[str, object]] = []
    monkeypatch.setattr(runtime, "start_speech_worker", lambda: True)
    monkeypatch.setattr(
        runtime,
        "_decode_reference_audio_file",
        lambda path: (np.array([0.0, 0.1], dtype=np.float32), 16000),
    )

    def fake_play_speech_chunks(**kwargs: object) -> dict[str, object]:
        played_jobs.append(dict(kwargs))
        return {
            "played": True,
            "sample_rate": 24000,
            "sample_count": 3,
            "duration_seconds": 0.0,
            "model_id": "Qwen/test-model",
            "device": "cpu",
            "language": kwargs["language"],
            "player": "sounddevice-stream",
            "mode": kwargs["mode"],
        }

    monkeypatch.setattr(runtime, "play_speech_chunks", fake_play_speech_chunks)

    runtime.speak_text(chunks=["first"], voice_description="warm", language="en")
    runtime.speak_text_as_clone(chunks=["second"], reference_audio_path="voice.wav", language="en")
    runtime._speech_queue.put_nowait(None)
    runtime._speech_worker_loop()

    assert played_jobs[0]["mode"] == VOICE_DESIGN_MODEL_KIND
    assert played_jobs[1]["mode"] == CLONE_MODEL_KIND
    assert played_jobs[1]["reference_text"] is None
    status = runtime.status()
    assert status["speech_jobs_completed"] == 2
    assert status["speech_phase"] == "idle"


# MARK: Synthesis


def test_synthesize_voice_design_chunk_uses_non_streaming_mode(
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    fake_model = FakeVoiceDesignModel()
    runtime._model_state(VOICE_DESIGN_MODEL_KIND)["model"] = fake_model
    runtime._model_state(VOICE_DESIGN_MODEL_KIND)["resolved_device"] = "cpu"

    result = runtime._synthesize_audio_chunk(
        mode=VOICE_DESIGN_MODEL_KIND,
        text="hello",
        voice_description="warm",
        language="English",
    )

    assert result["model_id"] == "Qwen/test-voice-design"
    assert fake_model.calls == [
        {
            "text": "hello",
            "language": "English",
            "instruct": "warm",
            "max_new_tokens": runtime.tts_max_new_tokens,
            "non_streaming_mode": True,
        }
    ]


def test_synthesize_clone_chunk_uses_x_vector_only_without_reference_text(
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    fake_model = FakeCloneModel()
    runtime._model_state(CLONE_MODEL_KIND)["model"] = fake_model
    runtime._model_state(CLONE_MODEL_KIND)["resolved_device"] = "cpu"

    reference_audio = (np.array([0.0, 0.1], dtype=np.float32), 16000)
    result = runtime._synthesize_audio_chunk(
        mode=CLONE_MODEL_KIND,
        text="hello",
        language="English",
        reference_audio=reference_audio,
    )

    assert result["model_id"] == "Qwen/test-clone"
    assert fake_model.calls == [
        {
            "text": "hello",
            "language": "English",
            "ref_audio": reference_audio,
            "x_vector_only_mode": True,
            "max_new_tokens": runtime.tts_max_new_tokens,
            "non_streaming_mode": True,
        }
    ]


def test_synthesize_clone_chunk_uses_reference_text_when_present(
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    fake_model = FakeCloneModel()
    runtime._model_state(CLONE_MODEL_KIND)["model"] = fake_model
    runtime._model_state(CLONE_MODEL_KIND)["resolved_device"] = "cpu"

    reference_audio = (np.array([0.0, 0.1], dtype=np.float32), 16000)
    runtime._synthesize_audio_chunk(
        mode=CLONE_MODEL_KIND,
        text="hello",
        language="English",
        reference_audio=reference_audio,
        reference_text="reference text",
    )

    assert fake_model.calls == [
        {
            "text": "hello",
            "language": "English",
            "ref_audio": reference_audio,
            "ref_text": "reference text",
            "x_vector_only_mode": False,
            "max_new_tokens": runtime.tts_max_new_tokens,
            "non_streaming_mode": True,
        }
    ]


def test_synthesize_chunk_rejects_audio_duration_over_guardrail(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    runtime.tts_max_chunk_audio_seconds = 0.001
    fake_model = FakeCloneModel(waveform=[0.0] * 128, sample_rate=24000)
    runtime._model_state(CLONE_MODEL_KIND)["model"] = fake_model
    runtime._model_state(CLONE_MODEL_KIND)["resolved_device"] = "cpu"

    with pytest.raises(RuntimeError, match="chunk audio duration exceeded"):
        runtime._synthesize_audio_chunk(
            mode=CLONE_MODEL_KIND,
            text="hello",
            language="English",
            reference_audio=(np.array([0.0, 0.1], dtype=np.float32), 16000),
        )


def test_synthesize_chunk_rejects_synth_duration_over_guardrail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    runtime.tts_max_chunk_synth_seconds = 0.001
    fake_model = FakeCloneModel()
    runtime._model_state(CLONE_MODEL_KIND)["model"] = fake_model
    runtime._model_state(CLONE_MODEL_KIND)["resolved_device"] = "cpu"

    timestamps = iter(
        [
            dt.datetime(2026, 3, 14, 0, 0, 0, tzinfo=dt.UTC),
            dt.datetime(2026, 3, 14, 0, 0, 1, tzinfo=dt.UTC),
            dt.datetime(2026, 3, 14, 0, 0, 1, tzinfo=dt.UTC),
        ]
    )

    def fake_utc_now() -> dt.datetime:
        try:
            return next(timestamps)
        except StopIteration:
            return dt.datetime(2026, 3, 14, 0, 0, 1, tzinfo=dt.UTC)

    monkeypatch.setattr(runtime_module, "_utc_now", fake_utc_now)

    with pytest.raises(RuntimeError, match="chunk synthesis exceeded"):
        runtime._synthesize_audio_chunk(
            mode=CLONE_MODEL_KIND,
            text="hello",
            language="English",
            reference_audio=(np.array([0.0, 0.1], dtype=np.float32), 16000),
        )


def test_synthesize_clone_chunk_retries_with_cpu_fallback_for_meta_tensor_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    runtime.device_preference = "auto"
    first_model = SimpleNamespace()
    second_model = FakeCloneModel()

    def failing_generate_voice_clone(**kwargs: object) -> tuple[list[list[float]], int]:
        del kwargs
        raise RuntimeError("meta tensor failure during tensor.item()")

    first_model.generate_voice_clone = failing_generate_voice_clone
    runtime._model_state(CLONE_MODEL_KIND)["model"] = first_model
    runtime._model_state(CLONE_MODEL_KIND)["resolved_device"] = "mps"

    def reload_with_cpu_fallback(model_kind: str) -> None:
        runtime._model_state(model_kind)["model"] = second_model
        runtime._model_state(model_kind)["resolved_device"] = "cpu"
        runtime._model_state(model_kind)["cpu_fallback_active"] = True

    monkeypatch.setattr(runtime, "_reload_model_with_cpu_fallback", reload_with_cpu_fallback)

    result = runtime._synthesize_audio_chunk(
        mode=CLONE_MODEL_KIND,
        text="hello",
        language="English",
        reference_audio=(np.array([0.0, 0.1], dtype=np.float32), 16000),
    )

    assert result["device"] == "cpu"
    assert second_model.calls[0]["x_vector_only_mode"] is True


def test_synthesize_chunk_requires_model_to_be_loaded_first(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)

    with pytest.raises(RuntimeError, match="call load_model"):
        runtime._synthesize_audio_chunk(
            mode=VOICE_DESIGN_MODEL_KIND,
            text="hello",
            voice_description="warm",
            language="English",
        )


# MARK: Playback


def test_wavbuffer_command_uses_seconds_preroll_by_default(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    runtime.playback_backend = "wavbuffer"
    runtime.wavbuffer_binary_path = "/tmp/wavbuffer"
    runtime.wavbuffer_queue_depth = 6
    runtime.playback_preroll_seconds = 1.5

    assert runtime._wavbuffer_command() == [
        "/tmp/wavbuffer",
        "--queue-depth",
        "6",
        "--preroll-seconds",
        "1.5",
    ]


def test_wavbuffer_command_uses_buffer_preroll_when_configured(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    runtime.wavbuffer_binary_path = "/tmp/wavbuffer"
    runtime.wavbuffer_preroll_mode = "buffers"
    runtime.playback_preroll_chunks = 3

    assert runtime._wavbuffer_command(chunk_count=4) == [
        "/tmp/wavbuffer",
        "--queue-depth",
        "8",
        "--preroll-buffers",
        "2",
    ]


def test_effective_preroll_chunk_target_balances_short_and_long_jobs(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    runtime.playback_preroll_chunks = 3

    assert runtime._effective_preroll_chunk_target(1) == 1
    assert runtime._effective_preroll_chunk_target(2) == 1
    assert runtime._effective_preroll_chunk_target(3) == 2
    assert runtime._effective_preroll_chunk_target(7) == 2


def test_playback_start_chunk_target_holds_longer_replies_longer(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)

    assert runtime._playback_start_chunk_target(1) == 1
    assert runtime._playback_start_chunk_target(2) == 2
    assert runtime._playback_start_chunk_target(3) == 3
    assert runtime._playback_start_chunk_target(4) == 4
    assert runtime._playback_start_chunk_target(7) == 4


def test_required_start_buffer_seconds_scales_with_observed_deficit(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    job_metrics = runtime._new_job_metrics(
        mode=CLONE_MODEL_KIND,
        normalized_chunks=["a", "b", "c", "d"],
        effective_preroll_chunks=2,
        total_chunk_char_count=4,
        job_started_at=dt.datetime.now(dt.UTC),
    )

    assert (
        runtime._required_start_buffer_seconds(job_metrics=job_metrics, total_chunk_count=2)
        == 0.0
    )
    assert (
        runtime._required_start_buffer_seconds(job_metrics=job_metrics, total_chunk_count=4)
        == 18.0
    )

    job_metrics["min_real_time_margin_ms"] = -10000
    assert (
        runtime._required_start_buffer_seconds(job_metrics=job_metrics, total_chunk_count=2)
        == 0.0
    )
    assert (
        runtime._required_start_buffer_seconds(job_metrics=job_metrics, total_chunk_count=3)
        == 0.0
    )
    assert (
        runtime._required_start_buffer_seconds(job_metrics=job_metrics, total_chunk_count=4)
        == 18.0
    )


def test_evaluate_playback_start_admission_defers_until_safe_buffer(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    job_metrics = runtime._new_job_metrics(
        mode=CLONE_MODEL_KIND,
        normalized_chunks=["a", "b", "c", "d"],
        effective_preroll_chunks=2,
        total_chunk_char_count=4,
        job_started_at=dt.datetime.now(dt.UTC),
    )
    job_metrics["min_real_time_margin_ms"] = -14886
    job_metrics["max_synth_to_audio_ratio"] = 1.994

    deferred = runtime._evaluate_playback_start_admission(
        job_metrics=job_metrics,
        total_chunk_count=4,
        buffered_chunk_count=2,
        buffered_audio_seconds=18.96,
        synthesis_done=False,
    )
    assert deferred["admit"] is False
    assert deferred["defer_reason"] == "waiting_for_chunk_target"

    admitted = runtime._evaluate_playback_start_admission(
        job_metrics=job_metrics,
        total_chunk_count=4,
        buffered_chunk_count=4,
        buffered_audio_seconds=30.96,
        synthesis_done=False,
    )
    assert admitted["admit"] is True
    assert admitted["reason"] == "safe_buffer_threshold_met"

    fully_buffered = runtime._evaluate_playback_start_admission(
        job_metrics=job_metrics,
        total_chunk_count=4,
        buffered_chunk_count=4,
        buffered_audio_seconds=43.12,
        synthesis_done=True,
    )
    assert fully_buffered["admit"] is True
    assert fully_buffered["reason"] == "all_chunks_buffered"


def test_encode_waveform_as_wav_bytes_uses_float32_pcm(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)

    wav_bytes = runtime._encode_waveform_as_wav_bytes(
        waveform=np.array([0.0, 0.25, -0.5], dtype=np.float32),
        sample_rate=24000,
    )

    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"
    assert wav_bytes[12:16] == b"fmt "
    assert int.from_bytes(wav_bytes[20:22], "little") == 3
    assert int.from_bytes(wav_bytes[22:24], "little") == 1
    assert int.from_bytes(wav_bytes[24:28], "little") == 24000
    assert wav_bytes[36:40] == b"data"


def test_parse_wavbuffer_event_line_handles_quoted_fields(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)

    parsed = runtime._parse_wavbuffer_event_line(
        'wavbuffer event=error reason="stream_starved" description="hello world"'
    )

    assert parsed == {
        "event": "error",
        "reason": "stream_starved",
        "description": "hello world",
    }


def test_play_speech_chunks_generates_and_writes_clone_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    played: list[list[float]] = []
    opened_streams: list[FakeStream] = []

    class FakeStream:
        def __init__(self) -> None:
            self.closed = False
            self.stopped = False

        def start(self) -> None:
            return None

        def write(self, waveform: np.ndarray) -> bool:
            played.append(waveform.tolist())
            return False

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            self.closed = True

    def open_output_stream(**kwargs: object) -> FakeStream:
        del kwargs
        stream = FakeStream()
        opened_streams.append(stream)
        return stream

    def synthesize_audio_chunk(**kwargs: object) -> dict[str, object]:
        waveform = [0.0, 0.1, 0.2] if kwargs["text"] == "Hello there" else [0.3, 0.4, 0.5]
        return {
            "waveform": np.array(waveform, dtype=np.float32),
            "sample_rate": 24000,
            "model_id": "Qwen/test-clone",
            "device": "cpu",
            "language": kwargs["language"],
        }

    monkeypatch.setattr(runtime, "_synthesize_audio_chunk", synthesize_audio_chunk)
    monkeypatch.setattr(runtime, "_open_output_stream", open_output_stream)

    result = runtime.play_speech_chunks(
        mode=CLONE_MODEL_KIND,
        chunks=["Hello there", "General Kenobi"],
        reference_audio=(np.array([0.0, 0.1], dtype=np.float32), 16000),
        reference_text="Reference text",
        language="en",
    )

    assert result["played"] is True
    assert result["mode"] == CLONE_MODEL_KIND
    assert len(opened_streams) == 1
    assert opened_streams[0].stopped is True
    assert opened_streams[0].closed is True
    assert played[0] == pytest.approx([0.0, 0.1, 0.2])
    assert played[1] == pytest.approx([0.3, 0.4, 0.5])
    recent_events = cast(list[dict[str, object]], runtime.status()["recent_events"])
    assert any(event["event"] == "speech_chunk_ready_for_playback" for event in recent_events)
    assert any(event["event"] == "speech_playback_start_admitted" for event in recent_events)
    assert any(event["event"] == "speech_first_audio_started" for event in recent_events)
    assert any(event["event"] == "speech_job_metrics_summary" for event in recent_events)


def test_play_speech_chunks_recovers_from_output_underflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    played: list[list[float]] = []
    opened_streams: list[Any] = []

    class FakeStream:
        def __init__(self, *, underflow_first_write: bool) -> None:
            self.closed = False
            self.stopped = False
            self._underflow_first_write = underflow_first_write
            self._write_count = 0

        def start(self) -> None:
            return None

        def write(self, waveform: np.ndarray) -> bool:
            self._write_count += 1
            played.append(waveform.tolist())
            if self._underflow_first_write and self._write_count == 1:
                return True
            return False

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            self.closed = True

    def open_output_stream(**kwargs: object) -> FakeStream:
        del kwargs
        stream = FakeStream(underflow_first_write=not opened_streams)
        opened_streams.append(stream)
        return stream

    monkeypatch.setattr(
        runtime,
        "_synthesize_audio_chunk",
        lambda **kwargs: {
            "waveform": np.array([0.0, 0.1, 0.2], dtype=np.float32),
            "sample_rate": 24000,
            "model_id": "Qwen/test-clone",
            "device": "cpu",
            "language": kwargs["language"],
        },
    )
    monkeypatch.setattr(runtime, "_open_output_stream", open_output_stream)

    result = runtime.play_speech_chunks(
        mode=CLONE_MODEL_KIND,
        chunks=["Hello there"],
        reference_audio=(np.array([0.0, 0.1], dtype=np.float32), 16000),
        reference_text="Reference text",
        language="en",
    )

    assert result["played"] is True
    assert len(opened_streams) == 2
    assert opened_streams[0].stopped is True
    assert opened_streams[0].closed is True
    assert opened_streams[1].stopped is True
    assert opened_streams[1].closed is True
    assert played == [
        pytest.approx([0.0, 0.1, 0.2]),
        pytest.approx([0.0, 0.1, 0.2]),
    ]

    recent_events = cast(list[dict[str, object]], runtime.status()["recent_events"])
    assert any(event["event"] == "speech_chunk_playback_underflow" for event in recent_events)
    assert any(event["event"] == "speech_chunk_playback_retrying" for event in recent_events)
    assert any(
        event["event"] == "speech_chunk_playback_handoff_completed"
        for event in recent_events
    )


def test_play_speech_chunks_with_wavbuffer_retries_after_stream_starvation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    runtime.playback_backend = "wavbuffer"
    runtime.wavbuffer_binary_path = "/tmp/wavbuffer"
    written_chunks: list[bytes] = []

    class FakeStderr:
        def __init__(self, lines: list[bytes]) -> None:
            self._lines = list(lines)

        def readline(self) -> bytes:
            if self._lines:
                return self._lines.pop(0)
            return b""

    class FakeStdin:
        def __init__(self, *, fail_on_write: bool = False) -> None:
            self._fail_on_write = fail_on_write
            self.closed = False

        def write(self, payload: bytes) -> int:
            if self._fail_on_write:
                raise BrokenPipeError("stream starved")
            written_chunks.append(payload)
            return len(payload)

        def flush(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def __init__(
            self,
            *,
            pid: int,
            fail_on_write: bool,
            returncode: int,
            stderr_lines: list[bytes],
        ) -> None:
            self.pid = pid
            self.stdin = FakeStdin(fail_on_write=fail_on_write)
            self.stderr = FakeStderr(stderr_lines)
            self._returncode = returncode

        def poll(self) -> int | None:
            if self.stdin.closed:
                return self._returncode
            return None

        def wait(self) -> int:
            return self._returncode

    processes = [
        FakeProcess(
            pid=101,
            fail_on_write=True,
            returncode=1,
            stderr_lines=[
                b'wavbuffer event=underrun buffered_buffers=0\n',
                b'wavbuffer event=error reason="stream_starved" description="starved"\n',
            ],
        ),
        FakeProcess(
            pid=102,
            fail_on_write=False,
            returncode=0,
            stderr_lines=[
                b'wavbuffer event=engine_started format="pcm"\n',
                (
                    b'wavbuffer event=playback_started buffered_buffers=1 '
                    b'buffered_seconds=0.5 reason="forced_input_completed"\n'
                ),
                b'wavbuffer event=completed preroll="seconds"\n',
            ],
        ),
    ]

    monkeypatch.setattr(runtime, "_spawn_wavbuffer_process", lambda command: processes.pop(0))
    monkeypatch.setattr(
        runtime,
        "_synthesize_audio_chunk",
        lambda **kwargs: {
            "waveform": np.array([0.0, 0.1, 0.2], dtype=np.float32),
            "sample_rate": 24000,
            "model_id": "Qwen/test-clone",
            "device": "cpu",
            "language": kwargs["language"],
        },
    )

    result = runtime.play_speech_chunks(
        mode=CLONE_MODEL_KIND,
        chunks=["Hello there"],
        reference_audio=(np.array([0.0, 0.1], dtype=np.float32), 16000),
        language="en",
    )

    assert result["played"] is True
    assert result["player"] == "wavbuffer-subprocess"
    assert len(written_chunks) == 1
    recent_events = cast(list[dict[str, object]], runtime.status()["recent_events"])
    assert any(event["event"] == "speech_wavbuffer_event_received" for event in recent_events)
    assert any(event["event"] == "speech_chunk_playback_retrying" for event in recent_events)
    assert any(event["event"] == "speech_playback_start_admitted" for event in recent_events)
    assert any(
        event["event"] == "speech_first_audio_started"
        and event.get("reason") == "forced_input_completed"
        for event in recent_events
    )
    assert any(
        event["event"] == "speech_job_metrics_summary"
        and event.get("wavbuffer_forced_input_completed") is True
        for event in recent_events
    )
    assert any(
        event["event"] == "speech_chunk_playback_handoff_completed"
        for event in recent_events
    )


def test_play_speech_chunks_with_wavbuffer_fails_after_retry_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    runtime.playback_backend = "wavbuffer"
    runtime.wavbuffer_binary_path = "/tmp/wavbuffer"
    runtime.playback_underflow_retries = 1

    class FakeStderr:
        def __init__(self, lines: list[bytes]) -> None:
            self._lines = list(lines)

        def readline(self) -> bytes:
            if self._lines:
                return self._lines.pop(0)
            return b""

    class FakeStdin:
        def write(self, payload: bytes) -> int:
            del payload
            raise BrokenPipeError("stream starved")

        def flush(self) -> None:
            return None

        def close(self) -> None:
            return None

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.stdin = FakeStdin()
            self.stderr = FakeStderr(
                [
                    b'wavbuffer event=underrun buffered_buffers=0\n',
                    b'wavbuffer event=error reason="stream_starved" description="starved"\n',
                ]
            )

        def poll(self) -> int | None:
            return 1

        def wait(self) -> int:
            return 1

    processes = [FakeProcess(201), FakeProcess(202)]
    monkeypatch.setattr(runtime, "_spawn_wavbuffer_process", lambda command: processes.pop(0))
    monkeypatch.setattr(
        runtime,
        "_synthesize_audio_chunk",
        lambda **kwargs: {
            "waveform": np.array([0.0, 0.1, 0.2], dtype=np.float32),
            "sample_rate": 24000,
            "model_id": "Qwen/test-clone",
            "device": "cpu",
            "language": kwargs["language"],
        },
    )

    with pytest.raises(
        RuntimeError,
        match="audio output underflowed during streamed playback after 2 attempt\\(s\\)",
    ):
        runtime.play_speech_chunks(
            mode=CLONE_MODEL_KIND,
            chunks=["Hello there"],
            reference_audio=(np.array([0.0, 0.1], dtype=np.float32), 16000),
            language="en",
        )


def test_play_speech_chunks_with_null_backend_discards_audio(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    runtime.playback_backend = "null"
    fake_model = FakeVoiceDesignModel()
    runtime._model_state(VOICE_DESIGN_MODEL_KIND)["model"] = fake_model
    runtime._model_state(VOICE_DESIGN_MODEL_KIND)["resolved_device"] = "cpu"

    result = runtime.play_speech_chunks(
        mode=VOICE_DESIGN_MODEL_KIND,
        chunks=["hello"],
        voice_description="warm",
        language="en",
    )

    assert result["played"] is True
    assert result["player"] == "null-sink"
    assert result["sample_rate"] == 24000
    recent_events = cast(list[dict[str, object]], runtime.status()["recent_events"])
    assert any(
        event["event"] == "speech_chunk_playback_handoff_completed"
        and event.get("player") == "null-sink"
        for event in recent_events
    )


def test_play_speech_chunks_fails_after_exhausting_underflow_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    runtime.playback_underflow_retries = 1
    opened_streams: list[Any] = []

    class FakeStream:
        def __init__(self) -> None:
            self.closed = False
            self.stopped = False

        def start(self) -> None:
            return None

        def write(self, waveform: np.ndarray) -> bool:
            del waveform
            return True

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            self.closed = True

    def open_output_stream(**kwargs: object) -> FakeStream:
        del kwargs
        stream = FakeStream()
        opened_streams.append(stream)
        return stream

    monkeypatch.setattr(
        runtime,
        "_synthesize_audio_chunk",
        lambda **kwargs: {
            "waveform": np.array([0.0, 0.1, 0.2], dtype=np.float32),
            "sample_rate": 24000,
            "model_id": "Qwen/test-clone",
            "device": "cpu",
            "language": kwargs["language"],
        },
    )
    monkeypatch.setattr(runtime, "_open_output_stream", open_output_stream)

    with pytest.raises(
        RuntimeError,
        match="audio output underflowed during streamed playback after 2 attempt\\(s\\)",
    ):
        runtime.play_speech_chunks(
            mode=CLONE_MODEL_KIND,
            chunks=["Hello there"],
            reference_audio=(np.array([0.0, 0.1], dtype=np.float32), 16000),
            reference_text="Reference text",
            language="en",
        )

    assert len(opened_streams) == 2
    assert all(stream.stopped is True for stream in opened_streams)
    assert all(stream.closed is True for stream in opened_streams)


# MARK: Status


def test_status_ready_requires_all_startup_models_loaded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)

    assert runtime.status()["ready"] is False

    def load_impl(model_kind: str) -> object:
        if model_kind == VOICE_DESIGN_MODEL_KIND:
            return FakeVoiceDesignModel()
        return FakeCloneModel()

    monkeypatch.setattr(runtime, "_load_model_impl", load_impl)
    runtime.load_model(model_kind=VOICE_DESIGN_MODEL_KIND)

    assert runtime.status()["ready"] is False

    runtime.load_model(model_kind=CLONE_MODEL_KIND)
    assert runtime.status()["ready"] is True


def test_status_ready_when_no_startup_models_are_configured(tmp_path: Path) -> None:
    runtime = TTSRuntime(
        voice_design_model_id="Qwen/test-voice-design",
        clone_model_id="Qwen/test-clone",
        enable_voice_design_model=False,
        enable_clone_model=False,
        device_preference="cpu",
    )

    assert runtime.status()["ready"] is True
    assert runtime.status()["startup_model_option"] == runtime_module.STARTUP_MODEL_OPTION_NONE


def test_emit_runtime_event_records_recent_event_and_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    printed: list[str] = []

    monkeypatch.setattr(
        runtime_module,
        "_ORIGINAL_PRINT",
        lambda message, **kwargs: printed.append(str(message)),
    )

    runtime._emit_runtime_event("custom_event", level="info", job_id=7)

    status = runtime.status()
    recent_events = cast(list[dict[str, object]], status["recent_events"])
    assert recent_events[-1]["event"] == "custom_event"
    assert recent_events[-1]["job_id"] == 7
    assert '"event": "custom_event"' in printed[-1]


def test_current_process_memory_snapshot_reads_rss_and_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(
        mps=SimpleNamespace(
            current_allocated_memory=lambda: 1024,
            driver_allocated_memory=lambda: 2048,
        )
    )
    monkeypatch.setattr(
        runtime_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="512\n"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    snapshot = runtime_module._current_process_memory_snapshot()

    assert snapshot["process_rss_kib"] == 512
    assert snapshot["process_rss_bytes"] == 512 * 1024
    assert snapshot["torch_mps_current_allocated_bytes"] == 1024
    assert snapshot["torch_mps_driver_allocated_bytes"] == 2048


def test_emit_runtime_memory_snapshot_records_snapshot_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    monkeypatch.setattr(
        runtime_module,
        "_current_process_memory_snapshot",
        lambda: {"process_rss_bytes": 4096, "process_rss_mib": 0.0},
    )

    runtime._emit_runtime_memory_snapshot(
        level="info",
        snapshot_event="speech_chunk_synthesis_started",
        chunk_index=1,
    )

    recent_events = cast(list[dict[str, object]], runtime.status()["recent_events"])
    assert recent_events[-1]["event"] == "speech_memory_snapshot"
    assert recent_events[-1]["snapshot_event"] == "speech_chunk_synthesis_started"
    assert recent_events[-1]["process_rss_bytes"] == 4096


def test_recent_event_buffer_is_bounded(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)

    for index in range(runtime_module.DEFAULT_RECENT_EVENT_LIMIT + 5):
        runtime._emit_runtime_event("bounded_event", level="info", event_index=index)

    recent_events = cast(list[dict[str, object]], runtime.status()["recent_events"])
    assert len(recent_events) == runtime_module.DEFAULT_RECENT_EVENT_LIMIT
    assert recent_events[0]["event_index"] == 5


def test_status_reports_timestamps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    fake_now = dt.datetime(2026, 3, 14, 17, 0, tzinfo=dt.UTC)
    monkeypatch.setattr("app.runtime._utc_now", lambda: fake_now)
    monkeypatch.setattr(runtime, "_load_model_impl", lambda model_kind: FakeVoiceDesignModel())

    runtime.load_model(model_kind=VOICE_DESIGN_MODEL_KIND)
    status = runtime.status()

    assert status["last_loaded_at"] == fake_now.isoformat()
    assert status["voice_design_last_loaded_at"] == fake_now.isoformat()


# MARK: Reference Audio


def test_decode_reference_audio_file_rejects_missing_path(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)

    with pytest.raises(ValueError, match="reference audio file does not exist"):
        runtime._decode_reference_audio_file(str(tmp_path / "missing.wav"))
