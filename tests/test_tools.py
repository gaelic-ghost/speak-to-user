from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any, cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import tools
from app.text_chunking import chunk_text_for_tts


class FakeRuntime:
    def __init__(self) -> None:
        self.loaded_model_ids: list[str] = []
        self.missing_model_ids: list[str] = []
        self.speak_calls: list[dict[str, object]] = []
        self.profile_calls: list[dict[str, object]] = []
        self.startup_calls: list[dict[str, object]] = []

    def missing_required_model_ids(self, *, model_kinds: list[str]) -> list[str]:
        del model_kinds
        return list(self.missing_model_ids)

    def required_models_not_loaded_message(self, *, model_kinds: list[str]) -> str:
        del model_kinds
        return "Required TTS models are not loaded yet: " + ", ".join(self.missing_model_ids)

    def model_kind_for_id(self, model_id: str) -> str:
        return f"kind:{model_id}"

    def load_model(self, *, model_kind: str) -> dict[str, object]:
        self.loaded_model_ids.append(model_kind)
        return {"result": "success", "model_kind": model_kind}

    def status(self) -> dict[str, object]:
        return {"ready": True}

    def speak_text(self, *, text: str, voice_description: str, language: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "text": text,
            "voice_description": voice_description,
            "language": language,
        }
        self.speak_calls.append(payload)
        return {"result": "success", **payload}

    async def generate_speech_profile_from_voice_design(
        self,
        *,
        state_store: object,
        name: str,
        text: str,
        voice_description: str,
        language: str,
    ) -> dict[str, object]:
        payload = {
            "state_store": state_store,
            "name": name,
            "text": text,
            "voice_description": voice_description,
            "language": language,
        }
        self.profile_calls.append(payload)
        return {"result": "success", **payload}

    async def generate_speech_profile(
        self,
        *,
        state_store: object,
        name: str,
        reference_audio_path: str,
        reference_text: str | None,
    ) -> dict[str, object]:
        payload = {
            "state_store": state_store,
            "name": name,
            "reference_audio_path": reference_audio_path,
            "reference_text": reference_text,
        }
        self.profile_calls.append(payload)
        return {"result": "success", **payload}

    async def set_startup_model(self, *, state_store: object, option: str) -> dict[str, object]:
        payload = {"state_store": state_store, "option": option}
        self.startup_calls.append(payload)
        return {"result": "success", **payload}


class FakeContext:
    def __init__(
        self,
        *,
        runtime: FakeRuntime | None = None,
        state_store: object | None = None,
        elicit_result: object | Exception | None = None,
    ) -> None:
        self.lifespan_context = {}
        if runtime is not None:
            self.lifespan_context["runtime"] = runtime
        self.fastmcp = SimpleNamespace(_state_store=state_store or object())
        self._elicit_result = elicit_result
        self.elicit_messages: list[str] = []

    async def elicit(self, message: str, response_type: object) -> object:
        del response_type
        self.elicit_messages.append(message)
        if isinstance(self._elicit_result, Exception):
            raise self._elicit_result
        return self._elicit_result


def test_chunk_text_for_tts_keeps_short_paragraphs_in_one_chunk_when_they_fit() -> None:
    text = "A short paragraph. A second sentence."

    result = chunk_text_for_tts(text, max_chars=200)

    assert result == ["A short paragraph. A second sentence."]


def test_chunk_text_for_tts_packs_adjacent_sentences_up_to_the_limit() -> None:
    text = "First sentence. Second sentence. Third sentence."

    result = chunk_text_for_tts(text, max_chars=36)

    assert result == [
        "First sentence. Second sentence.",
        "Third sentence.",
    ]


def test_chunk_text_for_tts_prefers_nearest_sentence_end_before_limit() -> None:
    text = "One. Two. Three. Four."

    result = chunk_text_for_tts(text, max_chars=12)

    assert result == [
        "One. Two.",
        "Three. Four.",
    ]


def test_chunk_text_for_tts_splits_only_overlong_sentences_by_words() -> None:
    text = (
        "This is the first sentence. "
        "This second sentence is longer than the limit. "
        "Supercalifragilisticexpialidocious"
    )

    result = chunk_text_for_tts(text, max_chars=20)

    assert result == [
        "This is the first",
        "sentence.",
        "This second sentence",
        "is longer than the",
        "limit.",
        "Supercalifragilistic",
        "expialidocious",
    ]


def test_runtime_from_context_requires_runtime() -> None:
    with pytest.raises(RuntimeError, match="lifespan context"):
        tools._runtime_from_context(cast(Any, FakeContext()))


def test_ensure_required_models_loaded_returns_immediately_when_models_are_ready() -> None:
    runtime = FakeRuntime()
    ctx = FakeContext(runtime=runtime)

    asyncio.run(
        tools._ensure_required_models_loaded(
            cast(Any, ctx),
            model_kinds=["voice_design"],
            operation_name="speak_text",
        )
    )

    assert ctx.elicit_messages == []
    assert runtime.loaded_model_ids == []


def test_ensure_required_models_loaded_uses_elicitation_to_load_missing_models() -> None:
    runtime = FakeRuntime()
    runtime.missing_model_ids = ["voice", "clone"]
    ctx = FakeContext(runtime=runtime, elicit_result=SimpleNamespace(action="accept", data="load"))

    asyncio.run(
        tools._ensure_required_models_loaded(
            cast(Any, ctx),
            model_kinds=["voice_design", "clone"],
            operation_name="speak_text",
        )
    )

    assert len(ctx.elicit_messages) == 1
    assert runtime.loaded_model_ids == ["kind:voice", "kind:clone"]


def test_ensure_required_models_loaded_raises_when_elicitation_declines() -> None:
    runtime = FakeRuntime()
    runtime.missing_model_ids = ["voice"]
    ctx = FakeContext(runtime=runtime, elicit_result=SimpleNamespace(action="decline", data=None))

    with pytest.raises(RuntimeError, match="Required TTS models are not loaded yet"):
        asyncio.run(
            tools._ensure_required_models_loaded(
                cast(Any, ctx),
                model_kinds=["voice_design"],
                operation_name="speak_text",
            )
        )


def test_ensure_required_models_loaded_raises_fallback_message_when_elicitation_errors() -> None:
    runtime = FakeRuntime()
    runtime.missing_model_ids = ["voice"]
    ctx = FakeContext(runtime=runtime, elicit_result=RuntimeError("broken UI"))

    with pytest.raises(RuntimeError, match="Required TTS models are not loaded yet"):
        asyncio.run(
            tools._ensure_required_models_loaded(
                cast(Any, ctx),
                model_kinds=["voice_design"],
                operation_name="speak_text",
            )
        )


def test_speak_text_tool_forwards_to_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    ctx = FakeContext(runtime=runtime)
    ensure_calls: list[dict[str, object]] = []

    async def fake_ensure_required_models_loaded(
        ctx: object,
        *,
        model_kinds: list[str],
        operation_name: str,
    ) -> None:
        ensure_calls.append(
            {
                "ctx": ctx,
                "model_kinds": model_kinds,
                "operation_name": operation_name,
            }
        )

    monkeypatch.setattr(tools, "_ensure_required_models_loaded", fake_ensure_required_models_loaded)

    result = asyncio.run(
        tools.speak_text(
            cast(Any, ctx),
            text="hello",
            voice_description="warm",
            language="en",
        )
    )

    assert result["result"] == "success"
    assert ensure_calls == [
        {"ctx": ctx, "model_kinds": ["voice_design"], "operation_name": "speak_text"}
    ]
    assert runtime.speak_calls == [
        {"text": "hello", "voice_description": "warm", "language": "en"}
    ]


def test_generate_speech_profile_from_voice_design_trims_inputs_and_forwards_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime()
    state_store = object()
    ctx = FakeContext(runtime=runtime, state_store=state_store)

    async def fake_ensure_required_models_loaded(
        ctx: object,
        *,
        model_kinds: list[str],
        operation_name: str,
    ) -> None:
        del ctx, model_kinds, operation_name

    monkeypatch.setattr(tools, "_ensure_required_models_loaded", fake_ensure_required_models_loaded)

    result = asyncio.run(
        tools.generate_speech_profile_from_voice_design(
            cast(Any, ctx),
            name="default-femme",
            text="  hello there  ",
            voice_description="  warm and clear  ",
            language="en",
        )
    )

    assert result["result"] == "success"
    assert runtime.profile_calls == [
        {
            "state_store": state_store,
            "name": "default-femme",
            "text": "hello there",
            "voice_description": "warm and clear",
            "language": "en",
        }
    ]


def test_generate_speech_profile_from_voice_design_rejects_empty_text() -> None:
    runtime = FakeRuntime()
    ctx = FakeContext(runtime=runtime)

    with pytest.raises(ValueError, match="text must not be empty"):
        asyncio.run(
            tools.generate_speech_profile_from_voice_design(
                cast(Any, ctx),
                name="default-femme",
                text="   ",
                voice_description="warm",
            )
        )


def test_generate_speech_profile_from_voice_design_rejects_empty_voice_description() -> None:
    runtime = FakeRuntime()
    ctx = FakeContext(runtime=runtime)

    with pytest.raises(ValueError, match="voice_description must not be empty"):
        asyncio.run(
            tools.generate_speech_profile_from_voice_design(
                cast(Any, ctx),
                name="default-femme",
                text="hello",
                voice_description="   ",
            )
        )


def test_generate_speech_profile_tool_forwards_state_store() -> None:
    runtime = FakeRuntime()
    state_store = object()
    ctx = FakeContext(runtime=runtime, state_store=state_store)

    result = asyncio.run(
        tools.generate_speech_profile(
            cast(Any, ctx),
            name="default-femme",
            reference_audio_path="/tmp/ref.wav",
            reference_text="hello",
        )
    )

    assert result["result"] == "success"
    assert runtime.profile_calls == [
        {
            "state_store": state_store,
            "name": "default-femme",
            "reference_audio_path": "/tmp/ref.wav",
            "reference_text": "hello",
        }
    ]


def test_set_startup_model_tool_forwards_state_store() -> None:
    runtime = FakeRuntime()
    state_store = object()
    ctx = FakeContext(runtime=runtime, state_store=state_store)

    result = asyncio.run(tools.set_startup_model(cast(Any, ctx), option="none"))

    assert result["result"] == "success"
    assert runtime.startup_calls == [{"state_store": state_store, "option": "none"}]
