from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastmcp import Context

from app import server


class StubRuntime:
    def __init__(self) -> None:
        self.status_payload = {
            "ready": True,
            "model_loaded": True,
            "device": "cpu",
            "model_id": "Qwen/test-model",
            "last_used_at": None,
            "last_loaded_at": None,
            "last_error": None,
            "speech_in_progress": False,
            "speech_phase": "idle",
            "speech_current_job_id": None,
            "speech_current_chunk_index": None,
            "speech_current_chunk_count": None,
            "speech_queue_depth": 0,
            "speech_queue_maxsize": 32,
            "speech_jobs_queued": 0,
            "speech_jobs_completed": 0,
            "speech_jobs_failed": 0,
            "speech_last_enqueued_at": None,
            "speech_last_completed_at": None,
            "speech_last_error": None,
        }
        self.queued_jobs: list[dict[str, object]] = []

    def status(self) -> dict[str, object]:
        return dict(self.status_payload)

    def speak_text(self, **kwargs: object) -> dict[str, object]:
        chunks = list(cast(list[str], kwargs.get("chunks", [])))
        self.queued_jobs.append(
            {
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
        }


class StubContext:
    def __init__(self, runtime: object) -> None:
        self.lifespan_context = {"runtime": runtime}


def test_tts_status_uses_lifespan_runtime() -> None:
    runtime = StubRuntime()
    ctx = StubContext(runtime)

    result = server.tts_status(cast(Context, ctx))

    assert result["model_id"] == "Qwen/test-model"
    assert result["device"] == "cpu"


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


def test_speak_text_is_plain_tool() -> None:
    async def run() -> None:
        tool = await server.mcp.get_tool("speak_text")
        assert tool is not None
        assert tool.task_config is not None
        assert tool.task_config.mode == "forbidden"
        assert "output_format" not in tool.parameters
        assert "filename_stem" not in tool.parameters
        assert "Queue one full text job for local audio playback" in (tool.description or "")

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
    assert result["chunk_count"] == 1
    assert runtime.queued_jobs == [
        {
            "chunks": ["hello"],
            "voice_description": "warm",
            "language": "en",
        }
    ]


def test_speak_text_passes_full_text_as_one_request() -> None:
    runtime = StubRuntime()
    ctx = StubContext(runtime)
    from app.tools import speak_text as speak_text_tool

    result = speak_text_tool(
        cast(Context, ctx),
        text="First chunk. Second chunk. Third chunk.",
        voice_description="warm",
    )

    assert result["result"] == "success"
    assert result["chunked"] is False
    assert result["chunk_count"] == 1
    assert runtime.queued_jobs == [
        {
            "chunks": ["First chunk. Second chunk. Third chunk."],
            "voice_description": "warm",
            "language": "en",
        }
    ]


def test_speak_text_chunks_long_text_before_queueing() -> None:
    runtime = StubRuntime()
    ctx = StubContext(runtime)
    from app.tools import speak_text as speak_text_tool

    long_text = ("Sentence one. " * 120).strip()

    result = speak_text_tool(
        cast(Context, ctx),
        text=long_text,
        voice_description="warm",
    )

    assert result["result"] == "success"
    assert result["chunked"] is True
    assert cast(int, result["chunk_count"]) > 1
    assert len(runtime.queued_jobs) == 1
    assert len(cast(list[str], runtime.queued_jobs[0]["chunks"])) > 1
