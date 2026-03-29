from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any, cast

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.runtime import (
    CLONE_MODEL_KIND,
    DEFAULT_STARTUP_MODEL_OPTION,
    PROFILE_COLLECTION,
    STARTUP_MODEL_COLLECTION,
    STARTUP_MODEL_KEY,
    _prompt_item_from_payload,
    _serialized_array_from_payload,
    _speech_profile_from_payload,
    StoredSpeechProfilePromptItem,
    SerializedArray,
    SpeechProfile,
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
        profiles_dir=tmp_path / "state" / "profile_audio",
        playback_backend="null",
    )


def make_prompt_item(*, with_ref_code: bool) -> StoredSpeechProfilePromptItem:
    ref_code = (
        SerializedArray(dtype="float32", data=[[[0.1, 0.2], [0.3, 0.4]]])
        if with_ref_code
        else None
    )
    return StoredSpeechProfilePromptItem(
        ref_code=ref_code,
        ref_spk_embedding=SerializedArray(dtype="float32", data=[[0.5, 0.6, 0.7]]),
        x_vector_only_mode=not with_ref_code,
        icl_mode=with_ref_code,
        ref_text="reference text" if with_ref_code else None,
    )


def test_runtime_constructor_uses_explicit_model_ids(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)

    assert runtime.voice_design_model_id == "mlx/test-voice-design"
    assert runtime.clone_model_id == "mlx/test-clone"
    assert runtime.profiles_dir == tmp_path / "state" / "profile_audio"


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


def test_preload_defaults_to_all_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    loaded_model_kinds: list[str] = []

    def fake_load_model(model_kind: str) -> FakeModel:
        loaded_model_kinds.append(model_kind)
        return FakeModel()

    monkeypatch.setattr(runtime, "_load_model_impl", fake_load_model)

    import asyncio

    asyncio.run(runtime.preload(state_store=state_store))

    assert loaded_model_kinds == [VOICE_DESIGN_MODEL_KIND, CLONE_MODEL_KIND]
    assert runtime.status()["startup_model_option"] == DEFAULT_STARTUP_MODEL_OPTION


def test_preload_reads_startup_option_from_runtime_config_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    runtime._runtime_config_data_dir().mkdir(parents=True, exist_ok=True)
    runtime._startup_model_option_file_path().write_text(
        json.dumps({"value": {"option": "none"}}),
        encoding="utf-8",
    )
    loaded_model_kinds: list[str] = []

    def fake_load_model(model_kind: str) -> FakeModel:
        loaded_model_kinds.append(model_kind)
        return FakeModel()

    monkeypatch.setattr(runtime, "_load_model_impl", fake_load_model)

    import asyncio

    asyncio.run(runtime.preload(state_store=state_store))

    assert loaded_model_kinds == []
    assert runtime.status()["startup_model_option"] == "none"


def test_read_startup_model_option_uses_wrapped_state_payload(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    state_store.values[(STARTUP_MODEL_KEY, STARTUP_MODEL_COLLECTION)] = {
        "value": {"option": runtime.clone_model_id}
    }

    import asyncio

    option = asyncio.run(runtime._read_startup_model_option(state_store))

    assert option == runtime.clone_model_id


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
        text="Hello there. General Kenobi.",
        voice_description="warm and calm",
        language="en",
    )

    assert result["result"] == "success"
    assert result["mode"] == "voice_design"
    assert result["speech_jobs_completed"] == 1
    assert played_audio[0][1] == 24000
    assert fake_model.calls[0]["instruct"] == "warm and calm"
    assert fake_model.calls[0]["lang_code"] == "en"
    assert len(fake_model.calls) == 1


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
        text="Hello clone.",
        reference_audio_path=str(reference_path),
        reference_text="Hello clone.",
        language="en",
    )

    assert result["result"] == "success"
    assert result["mode"] == "clone"
    assert fake_model.calls[0]["ref_audio"] == str(reference_path.resolve())
    assert fake_model.calls[0]["ref_text"] == "Hello clone."


def test_generate_speech_profile_persists_reusable_prompt_items(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    prompt_items = [make_prompt_item(with_ref_code=True)]
    monkeypatch.setattr(runtime, "_create_voice_clone_prompt_items", lambda **kwargs: prompt_items)

    import asyncio

    created = asyncio.run(
        runtime.generate_speech_profile(
            state_store=state_store,
            name="default-femme",
            reference_audio_path=str(tmp_path / "sample.wav"),
            reference_text="reference text",
        )
    )

    assert created["result"] == "success"
    assert created["clone_mode"] == "reference_text"
    stored_payload = state_store.values[("default-femme", PROFILE_COLLECTION)]
    assert stored_payload["clone_model_id"] == "mlx/test-clone"
    assert stored_payload["prompt_items"][0]["ref_text"] == "reference text"
    assert stored_payload["prompt_items"][0]["icl_mode"] is True


def test_profile_lifecycle_uses_state_store_and_prompt_items(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    runtime._models[CLONE_MODEL_KIND] = FakeModel()
    monkeypatch.setattr(runtime, "_play_audio", lambda waveform, sample_rate: None)
    monkeypatch.setattr(
        runtime,
        "_create_voice_clone_prompt_items",
        lambda **kwargs: [make_prompt_item(with_ref_code=False)],
    )

    synth_calls: list[dict[str, object]] = []

    def fake_synthesize_with_prompt_items(
        *,
        model: object,
        text: str,
        language: str,
        prompt_items: list[StoredSpeechProfilePromptItem],
    ) -> tuple[np.ndarray, int]:
        synth_calls.append(
            {
                "model": model,
                "text": text,
                "language": language,
                "prompt_items": prompt_items,
            }
        )
        return np.array([0.0, 0.1, -0.1], dtype=np.float32), 24000

    monkeypatch.setattr(runtime, "_synthesize_with_prompt_items", fake_synthesize_with_prompt_items)

    import asyncio

    created = asyncio.run(
        runtime.generate_speech_profile(
            state_store=state_store,
            name="default-femme",
            reference_audio_path=str(tmp_path / "sample.wav"),
            reference_text=None,
        )
    )
    assert created["result"] == "success"

    listed = asyncio.run(runtime.list_speech_profiles(state_store=state_store))
    assert listed["profile_count"] == 1
    profiles = listed["profiles"]
    assert isinstance(profiles, list)
    assert profiles[0]["name"] == "default-femme"

    spoken = asyncio.run(
        runtime.speak_with_profile(
            state_store=state_store,
            name="default-femme",
            text="Profile playback text.",
            language="en",
        )
    )
    assert spoken["result"] == "success"
    assert synth_calls[0]["text"] == "Profile playback text."
    prompt_items = synth_calls[0]["prompt_items"]
    assert isinstance(prompt_items, list)
    assert prompt_items[0].x_vector_only_mode is True

    deleted = asyncio.run(
        runtime.delete_speech_profile(state_store=state_store, name="default-femme")
    )
    assert deleted["deleted"] is True
    assert state_store.values.get(("default-femme", PROFILE_COLLECTION)) is None


def test_generate_speech_profile_from_voice_design_persists_seed_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    temp_wav_path = tmp_path / "generated.wav"
    temp_wav_path.write_bytes(b"stub")

    monkeypatch.setattr(
        runtime,
        "_synthesize_text",
        lambda **kwargs: (np.array([0.0, 0.1, 0.2], dtype=np.float32), 24000),
    )
    monkeypatch.setattr(
        runtime,
        "_write_temp_reference_audio_file",
        lambda **kwargs: str(temp_wav_path),
    )
    monkeypatch.setattr(
        runtime,
        "_create_voice_clone_prompt_items",
        lambda **kwargs: [make_prompt_item(with_ref_code=True)],
    )

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

    assert created["result"] == "success"
    assert created["profile_source"] == "voice_design"
    assert created["seed_text_stored"] is True
    assert created["voice_description_stored"] is True
    assert not temp_wav_path.exists()

    stored_payload = state_store.values[("seeded", PROFILE_COLLECTION)]
    assert stored_payload["seed_text"] == "A short seed clip."
    assert stored_payload["voice_description"] == "bright and supportive"


def test_list_profiles_reads_profiles_from_filetree_data_directory(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    runtime.profiles_dir.parent.mkdir(parents=True, exist_ok=True)
    profile_data_dir = runtime._profile_data_dir()
    profile_data_dir.mkdir(parents=True, exist_ok=True)

    profile = asdict(
        SpeechProfile(
            name="saved",
            clone_model_id=runtime.clone_model_id,
            clone_mode="reference_text",
            created_at="2026-03-29T00:00:00+00:00",
            updated_at="2026-03-29T00:00:00+00:00",
            prompt_items=[make_prompt_item(with_ref_code=True)],
        )
    )
    (profile_data_dir / "saved.json").write_text(
        json.dumps({"value": profile}),
        encoding="utf-8",
    )

    import asyncio

    listed = asyncio.run(runtime.list_speech_profiles(state_store=state_store))

    assert listed["profile_count"] == 1
    profiles = listed["profiles"]
    assert isinstance(profiles, list)
    assert profiles[0]["name"] == "saved"


def test_get_profile_reads_wrapped_payload_from_profile_file(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    runtime._profile_data_dir().mkdir(parents=True, exist_ok=True)
    stored_profile = asdict(
        SpeechProfile(
            name="wrapped",
            clone_model_id=runtime.clone_model_id,
            clone_mode="reference_text",
            created_at="2026-03-29T00:00:00+00:00",
            updated_at="2026-03-29T00:00:00+00:00",
            prompt_items=[make_prompt_item(with_ref_code=True)],
        )
    )
    runtime._profile_file_path("wrapped").write_text(
        json.dumps({"value": stored_profile}),
        encoding="utf-8",
    )

    import asyncio

    profile = asyncio.run(runtime._get_profile(state_store, "wrapped"))

    assert profile is not None
    assert profile.name == "wrapped"
    assert profile.clone_mode == "reference_text"


def test_list_profiles_uses_in_memory_state_store_when_no_profile_directory(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    state_store = FakeStateStore()
    state_store.values[("memory-only", PROFILE_COLLECTION)] = asdict(
        SpeechProfile(
            name="memory-only",
            clone_model_id=runtime.clone_model_id,
            clone_mode="x_vector_only",
            created_at="2026-03-29T00:00:00+00:00",
            updated_at="2026-03-29T00:00:00+00:00",
            prompt_items=[make_prompt_item(with_ref_code=False)],
        )
    )

    import asyncio

    listed = asyncio.run(runtime.list_speech_profiles(state_store=state_store))

    assert listed["profile_count"] == 1
    profiles = cast(list[dict[str, object]], listed["profiles"])
    assert profiles[0]["name"] == "memory-only"


def test_read_json_file_raises_helpful_error_for_invalid_json(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    broken_path = tmp_path / "broken.json"
    broken_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="contains unreadable JSON"):
        runtime._read_json_file(
            broken_path,
            not_found_message="missing",
            unreadable_message="contains unreadable JSON",
        )


def test_read_json_file_raises_helpful_error_for_non_object_payload(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    broken_path = tmp_path / "broken.json"
    broken_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Expected a JSON object"):
        runtime._read_json_file(
            broken_path,
            not_found_message="missing",
            unreadable_message="contains unreadable JSON",
        )


def test_synthesize_with_prompt_items_dispatches_to_x_vector_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    prompt_item = make_prompt_item(with_ref_code=False)
    calls: list[str] = []

    def fake_x_vector(**kwargs: object) -> tuple[np.ndarray, int]:
        del kwargs
        calls.append("x_vector")
        return np.array([0.0], dtype=np.float32), 24000

    def fake_icl(**kwargs: object) -> tuple[np.ndarray, int]:
        del kwargs
        calls.append("icl")
        return np.array([0.0], dtype=np.float32), 24000

    monkeypatch.setattr(
        runtime,
        "_synthesize_with_x_vector_prompt_item",
        fake_x_vector,
    )
    monkeypatch.setattr(
        runtime,
        "_synthesize_with_icl_prompt_item",
        fake_icl,
    )

    result = runtime._synthesize_with_prompt_items(
        model=object(),
        text="hello",
        language="en",
        prompt_items=[prompt_item],
    )

    assert result[1] == 24000
    assert calls == ["x_vector"]


def test_synthesize_with_prompt_items_dispatches_to_icl_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    prompt_item = make_prompt_item(with_ref_code=True)
    calls: list[str] = []

    def fake_x_vector(**kwargs: object) -> tuple[np.ndarray, int]:
        del kwargs
        calls.append("x_vector")
        return np.array([0.0], dtype=np.float32), 24000

    def fake_icl(**kwargs: object) -> tuple[np.ndarray, int]:
        del kwargs
        calls.append("icl")
        return np.array([0.0], dtype=np.float32), 24000

    monkeypatch.setattr(
        runtime,
        "_synthesize_with_x_vector_prompt_item",
        fake_x_vector,
    )
    monkeypatch.setattr(
        runtime,
        "_synthesize_with_icl_prompt_item",
        fake_icl,
    )

    result = runtime._synthesize_with_prompt_items(
        model=object(),
        text="hello",
        language="en",
        prompt_items=[prompt_item],
    )

    assert result[1] == 24000
    assert calls == ["icl"]


def test_synthesize_with_prompt_items_rejects_empty_prompt_items(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)

    with pytest.raises(RuntimeError, match="does not contain any reusable clone artifacts"):
        runtime._synthesize_with_prompt_items(
            model=object(),
            text="hello",
            language="en",
            prompt_items=[],
        )


def test_speech_profile_from_payload_requires_prompt_items() -> None:
    with pytest.raises(RuntimeError, match="missing reusable clone artifacts"):
        _speech_profile_from_payload(
            {
                "name": "broken",
                "clone_model_id": "mlx/test-clone",
                "clone_mode": "reference_text",
                "created_at": "2026-03-29T00:00:00+00:00",
                "updated_at": "2026-03-29T00:00:00+00:00",
            }
        )


def test_prompt_item_from_payload_requires_embedding() -> None:
    with pytest.raises(RuntimeError, match="missing `ref_spk_embedding`"):
        _prompt_item_from_payload(
            {
                "icl_mode": False,
                "x_vector_only_mode": True,
            }
        )


def test_serialized_array_from_payload_requires_dtype_and_list_data() -> None:
    with pytest.raises(RuntimeError, match="tensor payload is unreadable"):
        _serialized_array_from_payload({"dtype": 7, "data": "bad"})
