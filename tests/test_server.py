from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from typing import cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastmcp import Context

from app import server


class StubRuntime:
    def __init__(self) -> None:
        self.status_payload = {
            "ready": True,
            "model_loaded": True,
            "device": "cpu",
            "model_id": "Qwen/test-voice-design",
            "torch_dtype": None,
            "last_used_at": None,
            "last_loaded_at": None,
            "last_error": None,
            "voice_design_model_enabled": True,
            "voice_design_model_loaded": True,
            "voice_design_model_id": "Qwen/test-voice-design",
            "voice_design_device": "cpu",
            "voice_design_last_used_at": None,
            "voice_design_last_loaded_at": None,
            "voice_design_last_error": None,
            "voice_design_cpu_fallback_active": False,
            "clone_model_enabled": True,
            "clone_model_loaded": True,
            "clone_model_id": "Qwen/test-clone",
            "clone_device": "cpu",
            "clone_last_used_at": None,
            "clone_last_loaded_at": None,
            "clone_last_error": None,
            "clone_cpu_fallback_active": False,
            "clone_in_progress": False,
            "speech_in_progress": False,
            "speech_phase": "idle",
            "speech_current_job_id": None,
            "speech_current_chunk_index": None,
            "speech_current_chunk_count": None,
            "speech_current_mode": None,
            "speech_queue_depth": 0,
            "speech_queue_maxsize": 32,
            "speech_jobs_queued": 0,
            "speech_jobs_completed": 0,
            "speech_jobs_failed": 0,
            "speech_last_enqueued_at": None,
            "speech_last_completed_at": None,
            "speech_last_error": None,
            "speech_last_event_at": None,
            "speech_last_event": None,
            "recent_events": [],
            "current_job_started_at": None,
            "current_chunk_started_at": None,
            "current_phase_started_at": None,
        }
        self.queued_jobs: list[dict[str, object]] = []
        self.generated_profiles: list[dict[str, object]] = []
        self.generated_voice_designed_profiles: list[dict[str, object]] = []
        self.deleted_profiles: list[str] = []

    def status(self) -> dict[str, object]:
        return dict(self.status_payload)

    def speak_text(self, **kwargs: object) -> dict[str, object]:
        chunks = list(cast(list[str], kwargs.get("chunks", [])))
        self.queued_jobs.append(
            {
                "mode": "voice_design",
                "chunks": chunks,
                "voice_description": kwargs.get("voice_description", ""),
                "language": kwargs.get("language", "en"),
            }
        )
        return {
            "result": "success",
            "queued": True,
            "job_id": len(self.queued_jobs),
            "chunked": len(chunks) > 1,
            "chunk_count": len(chunks),
            "language": kwargs.get("language", "en"),
            "enqueued_at": "2026-03-14T00:00:00+00:00",
            "playback_mode": "in-process-queue",
            "mode": "voice_design",
            "speech_in_progress": False,
            "speech_phase": "idle",
            "speech_current_job_id": None,
            "speech_current_chunk_index": None,
            "speech_current_chunk_count": None,
            "speech_queue_depth": len(self.queued_jobs),
            "speech_queue_maxsize": 32,
            "speech_jobs_queued": len(self.queued_jobs),
            "speech_jobs_completed": 0,
            "speech_jobs_failed": 0,
            "speech_last_enqueued_at": "2026-03-14T00:00:00+00:00",
            "speech_last_completed_at": None,
            "speech_last_error": None,
            "speech_last_event_at": None,
            "speech_last_event": "speech_job_queued",
        }

    def speak_text_as_clone(self, **kwargs: object) -> dict[str, object]:
        chunks = list(cast(list[str], kwargs.get("chunks", [])))
        self.queued_jobs.append(
            {
                "mode": "clone",
                "chunks": chunks,
                "reference_audio_path": kwargs.get("reference_audio_path", ""),
                "reference_text": kwargs.get("reference_text"),
                "language": kwargs.get("language", "en"),
            }
        )
        return {
            "result": "success",
            "queued": True,
            "job_id": len(self.queued_jobs),
            "chunked": len(chunks) > 1,
            "chunk_count": len(chunks),
            "language": kwargs.get("language", "en"),
            "enqueued_at": "2026-03-14T00:00:00+00:00",
            "playback_mode": "in-process-queue",
            "mode": "clone",
            "clone_mode": "reference_text"
            if kwargs.get("reference_text") is not None
            else "x_vector_only",
            "reference_audio_path": kwargs.get("reference_audio_path", ""),
            "reference_text_included": kwargs.get("reference_text") is not None,
            "clone_model_id": "Qwen/test-clone",
            "speech_in_progress": False,
            "speech_phase": "idle",
            "speech_current_job_id": None,
            "speech_current_chunk_index": None,
            "speech_current_chunk_count": None,
            "speech_queue_depth": len(self.queued_jobs),
            "speech_queue_maxsize": 32,
            "speech_jobs_queued": len(self.queued_jobs),
            "speech_jobs_completed": 0,
            "speech_jobs_failed": 0,
            "speech_last_enqueued_at": "2026-03-14T00:00:00+00:00",
            "speech_last_completed_at": None,
            "speech_last_error": None,
            "speech_last_event_at": None,
            "speech_last_event": "speech_job_queued",
        }

    async def generate_speech_profile(self, **kwargs: object) -> dict[str, object]:
        self.generated_profiles.append(dict(kwargs))
        return {
            "result": "success",
            "name": kwargs["name"],
            "clone_model_id": "Qwen/test-clone",
            "clone_mode": "reference_text"
            if kwargs.get("reference_text") is not None
            else "x_vector_only",
            "created_at": "2026-03-14T00:00:00+00:00",
            "updated_at": "2026-03-14T00:00:00+00:00",
            "reference_text_included": kwargs.get("reference_text") is not None,
        }

    async def generate_speech_profile_from_voice_design(
        self,
        **kwargs: object,
    ) -> dict[str, object]:
        self.generated_voice_designed_profiles.append(dict(kwargs))
        return {
            "result": "success",
            "name": kwargs["name"],
            "clone_model_id": "Qwen/test-clone",
            "clone_mode": "reference_text",
            "created_at": "2026-03-14T00:00:00+00:00",
            "updated_at": "2026-03-14T00:00:00+00:00",
            "reference_text_included": True,
            "profile_source": "voice_design",
            "seed_text_stored": True,
            "voice_description_stored": True,
        }

    async def list_speech_profiles(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "result": "success",
            "profiles": [
                {
                    "name": "demo",
                    "clone_model_id": "Qwen/test-clone",
                    "clone_mode": "x_vector_only",
                    "created_at": "2026-03-14T00:00:00+00:00",
                    "updated_at": "2026-03-14T00:00:00+00:00",
                    "reference_text_included": False,
                    "profile_source": None,
                }
            ],
            "profile_count": 1,
        }

    async def delete_speech_profile(self, **kwargs: object) -> dict[str, object]:
        name = cast(str, kwargs["name"])
        self.deleted_profiles.append(name)
        return {
            "result": "success",
            "deleted": True,
            "name": name,
            "profile_count": 0,
        }

    async def speak_with_profile(self, **kwargs: object) -> dict[str, object]:
        chunks = list(cast(list[str], kwargs.get("chunks", [])))
        self.queued_jobs.append(
            {
                "mode": "profile",
                "name": kwargs["name"],
                "chunks": chunks,
                "language": kwargs.get("language", "en"),
            }
        )
        return {
            "result": "success",
            "queued": True,
            "job_id": len(self.queued_jobs),
            "chunked": len(chunks) > 1,
            "chunk_count": len(chunks),
            "language": kwargs.get("language", "en"),
            "enqueued_at": "2026-03-14T00:00:00+00:00",
            "playback_mode": "in-process-queue",
            "mode": "clone",
            "profile_name": kwargs["name"],
            "clone_mode": "x_vector_only",
            "clone_model_id": "Qwen/test-clone",
            "speech_in_progress": False,
            "speech_phase": "idle",
            "speech_current_job_id": None,
            "speech_current_chunk_index": None,
            "speech_current_chunk_count": None,
            "speech_queue_depth": len(self.queued_jobs),
            "speech_queue_maxsize": 32,
            "speech_jobs_queued": len(self.queued_jobs),
            "speech_jobs_completed": 0,
            "speech_jobs_failed": 0,
            "speech_last_enqueued_at": "2026-03-14T00:00:00+00:00",
            "speech_last_completed_at": None,
            "speech_last_error": None,
            "speech_last_event_at": None,
            "speech_last_event": "speech_job_queued",
        }


class StubStateStore:
    async def put(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        return None

    async def get(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        return None

    async def delete(self, *args: object, **kwargs: object) -> bool:
        del args, kwargs
        return True


class StubFastMCP:
    def __init__(self) -> None:
        self._state_store = StubStateStore()


class StubContext:
    def __init__(self, runtime: object) -> None:
        self.lifespan_context = {"runtime": runtime}
        self.fastmcp = StubFastMCP()


def test_tts_status_uses_lifespan_runtime() -> None:
    runtime = StubRuntime()
    ctx = StubContext(runtime)

    result = server.tts_status(cast(Context, ctx))

    assert result["model_id"] == "Qwen/test-voice-design"
    assert result["clone_model_id"] == "Qwen/test-clone"


def test_app_lifespan_preloads_and_shuts_down(monkeypatch) -> None:
    events: list[str] = []

    class FakeRuntime:
        def preload(self) -> dict[str, object]:
            events.append("preload")
            return {"result": "success", "loaded": True}

        def shutdown(self) -> None:
            events.append("shutdown")

    monkeypatch.setattr(server.TTSRuntime, "from_env", classmethod(lambda cls: FakeRuntime()))

    async def run_lifespan() -> None:
        async with server.app_lifespan(server.mcp) as context:
            assert "runtime" in context
            events.append("yielded")

    asyncio.run(run_lifespan())

    assert events == ["preload", "yielded", "shutdown"]


def test_server_module_imports_when_loaded_from_file_path() -> None:
    server_path = Path(__file__).resolve().parents[1] / "app" / "server.py"
    spec = importlib.util.spec_from_file_location("server_file_import_test", server_path)

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.mcp.name == "speak-to-user"


def test_server_config_from_env_uses_http_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPEAK_TO_USER_HOST", raising=False)
    monkeypatch.delenv("SPEAK_TO_USER_PORT", raising=False)
    monkeypatch.delenv("SPEAK_TO_USER_MCP_PATH", raising=False)

    config = server.server_config_from_env()

    assert config == server.ServerConfig(
        host="127.0.0.1",
        port=8765,
        path="/mcp",
    )


def test_server_config_from_env_normalizes_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEAK_TO_USER_HOST", "127.0.0.1")
    monkeypatch.setenv("SPEAK_TO_USER_PORT", "8766")
    monkeypatch.setenv("SPEAK_TO_USER_MCP_PATH", "custom-mcp")

    config = server.server_config_from_env()

    assert config == server.ServerConfig(
        host="127.0.0.1",
        port=8766,
        path="/custom-mcp",
    )


def test_server_config_from_env_rejects_invalid_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEAK_TO_USER_PORT", "not-a-port")

    with pytest.raises(ValueError, match="SPEAK_TO_USER_PORT must be an integer"):
        server.server_config_from_env()


def test_main_runs_http_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(**kwargs: object) -> None:
        calls.append(dict(kwargs))

    monkeypatch.setattr(
        server,
        "server_config_from_env",
        lambda: server.ServerConfig(host="127.0.0.1", port=8766, path="/mcp"),
    )
    monkeypatch.setattr(server.mcp, "run", fake_run)

    server.main()

    assert calls == [
        {
            "transport": "http",
            "host": "127.0.0.1",
            "port": 8766,
            "path": "/mcp",
        }
    ]


def test_speak_text_is_plain_tool() -> None:
    async def run() -> None:
        tool = await server.mcp.get_tool("speak_text")
        assert tool is not None
        assert tool.task_config is not None
        assert tool.task_config.mode == "forbidden"

    asyncio.run(run())


def test_speak_text_as_clone_is_plain_tool() -> None:
    async def run() -> None:
        tool = await server.mcp.get_tool("speak_text_as_clone")
        assert tool is not None
        assert tool.task_config is not None
        assert tool.task_config.mode == "forbidden"
        properties = cast(dict[str, object], tool.parameters["properties"])
        assert "reference_audio_path" in properties
        assert "reference_text" in properties

    asyncio.run(run())


def test_generate_speech_profile_is_plain_tool() -> None:
    async def run() -> None:
        tool = await server.mcp.get_tool("generate_speech_profile")
        assert tool is not None
        assert tool.task_config is not None
        assert tool.task_config.mode == "forbidden"
        properties = cast(dict[str, object], tool.parameters["properties"])
        assert "name" in properties
        assert "reference_audio_path" in properties

    asyncio.run(run())


def test_generate_speech_profile_from_voice_design_is_plain_tool() -> None:
    async def run() -> None:
        tool = await server.mcp.get_tool("generate_speech_profile_from_voice_design")
        assert tool is not None
        assert tool.task_config is not None
        assert tool.task_config.mode == "forbidden"
        properties = cast(dict[str, object], tool.parameters["properties"])
        assert "name" in properties
        assert "text" in properties
        assert "voice_description" in properties

    asyncio.run(run())


def test_prompts_are_registered() -> None:
    async def run() -> None:
        prompts = await server.mcp.list_prompts()
        prompt_names = {prompt.name for prompt in prompts}
        assert "choose_speak_to_user_workflow" in prompt_names
        assert "guide_speech_profile_workflow" in prompt_names

    asyncio.run(run())


def test_resources_are_registered() -> None:
    async def run() -> None:
        resources = await server.mcp.list_resources()
        resource_uris = {str(resource.uri) for resource in resources}
        assert "guide://speak-to-user/usage" in resource_uris
        assert "state://speak-to-user/status" in resource_uris
        assert "state://speak-to-user/profiles" in resource_uris

    asyncio.run(run())


def test_choose_speak_to_user_workflow_prompt_renders_expected_guidance() -> None:
    async def run() -> None:
        result = await server.mcp.render_prompt("choose_speak_to_user_workflow")
        text = getattr(result.messages[0].content, "text", "")
        assert "speak_text" in text
        assert "speak_text_as_clone" in text
        assert "speak_with_profile" in text
        assert "tts_status" in text

    asyncio.run(run())


def test_usage_guide_resource_reads_expected_content() -> None:
    async def run() -> None:
        result = await server.mcp.read_resource("guide://speak-to-user/usage")
        assert "generate_speech_profile_from_voice_design" in result.contents[0].content
        assert "playback is global and serial" in result.contents[0].content

    asyncio.run(run())


def test_speak_with_profile_is_plain_tool() -> None:
    async def run() -> None:
        tool = await server.mcp.get_tool("speak_with_profile")
        assert tool is not None
        assert tool.task_config is not None
        assert tool.task_config.mode == "forbidden"
        properties = cast(dict[str, object], tool.parameters["properties"])
        assert "name" in properties
        assert "text" in properties

    asyncio.run(run())


def test_speak_text_queues_audio() -> None:
    runtime = StubRuntime()
    ctx = StubContext(runtime)

    result = server.speak_text(
        "hello",
        "warm",
        ctx=cast(Context, ctx),
    )

    assert result["result"] == "success"
    assert result["queued"] is True
    assert runtime.queued_jobs == [
        {
            "mode": "voice_design",
            "chunks": ["hello"],
            "voice_description": "warm",
            "language": "en",
        }
    ]


def test_speak_text_as_clone_queues_audio() -> None:
    runtime = StubRuntime()
    ctx = StubContext(runtime)

    result = server.speak_text_as_clone(
        "hello",
        "voice.wav",
        "reference text",
        ctx=cast(Context, ctx),
    )

    assert result["result"] == "success"
    assert result["clone_mode"] == "reference_text"
    assert runtime.queued_jobs == [
        {
            "mode": "clone",
            "chunks": ["hello"],
            "reference_audio_path": "voice.wav",
            "reference_text": "reference text",
            "language": "en",
        }
    ]


def test_generate_speech_profile_uses_runtime() -> None:
    runtime = StubRuntime()
    ctx = StubContext(runtime)

    result = asyncio.run(
        server.generate_speech_profile(
            "demo",
            "voice.wav",
            "reference text",
            ctx=cast(Context, ctx),
        )
    )

    assert result["result"] == "success"
    assert runtime.generated_profiles == [
        {
            "state_store": ctx.fastmcp._state_store,
            "name": "demo",
            "reference_audio_path": "voice.wav",
            "reference_text": "reference text",
        }
    ]


def test_generate_speech_profile_from_voice_design_uses_runtime() -> None:
    runtime = StubRuntime()
    ctx = StubContext(runtime)

    result = asyncio.run(
        server.generate_speech_profile_from_voice_design(
            "demo",
            "hello there",
            "warm and bright",
            "en",
            ctx=cast(Context, ctx),
        )
    )

    assert result["result"] == "success"
    assert result["profile_source"] == "voice_design"
    assert runtime.generated_voice_designed_profiles == [
        {
            "state_store": ctx.fastmcp._state_store,
            "name": "demo",
            "text": "hello there",
            "voice_description": "warm and bright",
            "language": "en",
        }
    ]


def test_list_speech_profiles_uses_runtime() -> None:
    runtime = StubRuntime()
    ctx = StubContext(runtime)

    result = asyncio.run(server.list_speech_profiles(ctx=cast(Context, ctx)))

    assert result["result"] == "success"
    assert result["profile_count"] == 1


def test_status_resource_uses_runtime_status() -> None:
    runtime = StubRuntime()
    ctx = StubContext(runtime)

    result = server.status_resource(ctx=cast(Context, ctx))

    payload = json.loads(result)
    assert payload["ready"] is True
    assert payload["clone_model_id"] == "Qwen/test-clone"


def test_speech_profiles_resource_uses_runtime_profile_summary() -> None:
    runtime = StubRuntime()
    ctx = StubContext(runtime)

    result = asyncio.run(server.speech_profiles_resource(ctx=cast(Context, ctx)))

    payload = json.loads(result)
    assert payload["result"] == "success"
    assert payload["profile_count"] == 1
    assert payload["profiles"][0]["name"] == "demo"


def test_delete_speech_profile_uses_runtime() -> None:
    runtime = StubRuntime()
    ctx = StubContext(runtime)

    result = asyncio.run(
        server.delete_speech_profile(
            "demo",
            ctx=cast(Context, ctx),
        )
    )

    assert result["deleted"] is True
    assert runtime.deleted_profiles == ["demo"]


def test_speak_with_profile_queues_audio() -> None:
    runtime = StubRuntime()
    ctx = StubContext(runtime)

    result = asyncio.run(
        server.speak_with_profile(
            "demo",
            "hello",
            ctx=cast(Context, ctx),
        )
    )

    assert result["result"] == "success"
    assert result["profile_name"] == "demo"
    assert runtime.queued_jobs[-1] == {
        "mode": "profile",
        "name": "demo",
        "chunks": ["hello"],
        "language": "en",
    }
