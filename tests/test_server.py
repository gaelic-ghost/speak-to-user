from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastmcp import Context
from fastmcp.dependencies import Progress

from app import server


class StubRuntime:
    def __init__(self) -> None:
        self.status_payload = {
            "ready": True,
            "model_loaded": False,
            "preload_in_progress": False,
            "device": "cpu",
            "model_id": "Qwen/test-model",
            "idle_unload_seconds": 1200,
            "last_used_at": None,
            "last_loaded_at": None,
            "last_unloaded_at": None,
            "preload_started_at": None,
            "preload_completed_at": None,
            "output_dir": "/tmp/generated-audio",
            "last_error": None,
        }
        self.generated_texts: list[str] = []
        self.generated_batches: list[list[str]] = []
        self.played_buffers: list[dict[str, object]] = []
        self.enqueued_jobs: list[dict[str, object]] = []

    def status(self) -> dict[str, object]:
        return dict(self.status_payload)

    def load_model(self) -> dict[str, object]:
        return {"result": "success", "loaded": True, **self.status_payload}

    def unload_model(self) -> dict[str, object]:
        return {"result": "success", "loaded": False, **self.status_payload}

    def set_idle_unload_timeout(self, seconds: int) -> dict[str, object]:
        payload = dict(self.status_payload)
        payload["idle_unload_seconds"] = seconds
        return payload

    def generate_audio(self, **kwargs: object) -> dict[str, object]:
        text = str(kwargs.get("text", ""))
        chunk_number = len(self.generated_texts) + 1
        self.generated_texts.append(text)
        return {
            "result": "success",
            "path": f"/tmp/generated-audio/test-{chunk_number}.wav",
            "format": "wav",
            "sample_rate": 24000,
            "sample_count": len(text),
            "duration_seconds": 0.001,
            "model_id": "Qwen/test-model",
            "device": "cpu",
            "language": kwargs.get("language", "en"),
            **kwargs,
        }

    def generate_speech_buffer(self, **kwargs: object) -> dict[str, object]:
        chunks = list(cast(list[str], kwargs.get("chunks", [])))
        self.generated_batches.append(chunks)
        return {
            "result": "success",
            "format": "wav",
            "sample_rate": 24000,
            "chunked": len(chunks) > 1,
            "chunk_count": len(chunks),
            "sample_count": sum(len(chunk) for chunk in chunks),
            "duration_seconds": 0.003,
            "model_id": "Qwen/test-model",
            "device": "cpu",
            "language": kwargs.get("language", "en"),
            "waveform": [0.1] * max(sum(len(chunk) for chunk in chunks), 1),
        }

    def play_audio_buffer(self, waveform: object, sample_rate: int) -> dict[str, object]:
        self.played_buffers.append({"waveform": waveform, "sample_rate": sample_rate})
        return {
            "result": "success",
            "played": True,
            "player": "sounddevice",
        }

    def enqueue_speech(self, **kwargs: object) -> dict[str, object]:
        chunks = list(cast(list[str], kwargs.get("chunks", [])))
        self.enqueued_jobs.append(
            {
                "chunks": chunks,
                "voice_description": kwargs.get("voice_description", ""),
                "language": kwargs.get("language", "en"),
            }
        )
        return {
            "result": "success",
            "queued": True,
            "job_id": len(self.enqueued_jobs),
            "chunked": len(chunks) > 1,
            "chunk_count": len(chunks),
            "language": kwargs.get("language", "en"),
            "enqueued_at": "2026-03-14T00:00:00+00:00",
            "speech_in_progress": False,
            "speech_current_job_id": None,
            "speech_queue_depth": 1,
            "speech_jobs_queued": len(self.enqueued_jobs),
            "speech_jobs_completed": 0,
            "speech_jobs_failed": 0,
            "speech_last_enqueued_at": "2026-03-14T00:00:00+00:00",
            "speech_last_completed_at": None,
            "speech_last_error": None,
        }


class StubProgress:
    def __init__(self) -> None:
        self.total: int | None = None
        self.current = 0
        self.messages: list[str | None] = []

    async def set_total(self, total: int) -> None:
        self.total = total

    async def increment(self, amount: int = 1) -> None:
        self.current += amount

    async def set_message(self, message: str | None) -> None:
        self.messages.append(message)


class StubContext:
    def __init__(self, runtime: object) -> None:
        self.lifespan_context = {"runtime": runtime}


def test_set_idle_unload_timeout_returns_error_payload() -> None:
    runtime = StubRuntime()
    ctx = StubContext(runtime)

    def boom(seconds: int) -> dict[str, object]:
        raise ValueError("bad timeout")

    runtime.set_idle_unload_timeout = boom  # type: ignore[method-assign]

    result = server.set_idle_unload_timeout(0, cast(Context, ctx))

    assert result["result"] == "error"
    assert result["error"] == "bad timeout"


def test_generate_audio_returns_runtime_error_payload() -> None:
    runtime = StubRuntime()
    ctx = StubContext(runtime)

    def boom(**kwargs: object) -> dict[str, object]:
        raise RuntimeError("generation failed")

    runtime.generate_audio = boom  # type: ignore[method-assign]

    result = server.generate_audio("hello", "warm", ctx=cast(Context, ctx))

    assert result["result"] == "error"
    assert result["error"] == "generation failed"


def test_tts_status_uses_lifespan_runtime() -> None:
    runtime = StubRuntime()
    ctx = StubContext(runtime)

    result = server.tts_status(cast(Context, ctx))

    assert result["model_id"] == "Qwen/test-model"
    assert result["device"] == "cpu"


def test_app_lifespan_starts_background_preload_and_shuts_down(monkeypatch) -> None:
    events: list[str] = []

    class FakeRuntime:
        def start_background_preload(self) -> bool:
            events.append("background-preload")
            return True

        def shutdown(self) -> None:
            events.append("shutdown")

    monkeypatch.setattr(server.TTSRuntime, "from_env", classmethod(lambda cls: FakeRuntime()))

    async def run_lifespan() -> None:
        async with server.app_lifespan(server.mcp) as context:
            assert "runtime" in context
            events.append("yielded")

    asyncio.run(run_lifespan())

    assert events == ["background-preload", "yielded", "shutdown"]


def test_server_module_imports_when_loaded_from_file_path() -> None:
    server_path = Path(__file__).resolve().parents[1] / "app" / "server.py"
    spec = importlib.util.spec_from_file_location("server_file_import_test", server_path)

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.mcp.name == "speak-to-user"


def test_speak_text_requires_background_task_mode() -> None:
    async def run() -> None:
        tool = await server.mcp.get_tool("speak_text")
        assert tool is not None
        assert tool.task_config is not None
        assert tool.task_config.mode == "optional"
        assert "output_format" not in tool.parameters
        assert "filename_stem" not in tool.parameters

    asyncio.run(run())


def test_speak_text_reports_progress_and_queues_audio() -> None:
    runtime = StubRuntime()
    ctx = StubContext(runtime)
    progress = StubProgress()

    async def run() -> dict[str, object]:
        return await server.speak_text(
            "hello",
            "warm",
            ctx=cast(Context, ctx),
            progress=cast(Progress, progress),
        )

    result = asyncio.run(run())

    assert result["result"] == "success"
    assert result["queued"] is True
    assert result["chunk_count"] == 1
    assert runtime.enqueued_jobs == [
        {
            "chunks": ["hello"],
            "voice_description": "warm",
            "language": "en",
        }
    ]
    assert progress.total == 3
    assert progress.current == 3
    assert progress.messages == [
        "Prepared 1 chunk(s) for queued speech playback",
        "Handing speech job to the local playback queue",
        "Speech job queued",
    ]


def test_speak_text_chunks_and_queues_in_fifo_order(monkeypatch) -> None:
    runtime = StubRuntime()
    ctx = StubContext(runtime)
    progress = StubProgress()
    chunked_text = ["First chunk.", "Second chunk.", "Third chunk."]

    monkeypatch.setattr("app.tools.chunk_text_for_tts", lambda text: list(chunked_text))

    async def run() -> dict[str, object]:
        from app.tools import speak_text as speak_text_tool

        return await speak_text_tool(
            cast(Context, ctx),
            cast(Progress, progress),
            text="ignored",
            voice_description="warm",
        )

    result = asyncio.run(run())

    assert result["result"] == "success"
    assert result["chunked"] is True
    assert result["chunk_count"] == 3
    assert runtime.enqueued_jobs == [
        {
            "chunks": chunked_text,
            "voice_description": "warm",
            "language": "en",
        }
    ]
    assert progress.total == 3
    assert progress.current == 3
