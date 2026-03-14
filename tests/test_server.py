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
        self.played_paths: list[str] = []

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

    def play_audio(self, path: str) -> dict[str, object]:
        self.played_paths.append(path)
        return {"result": "success", "path": path, "played": True, "player": "afplay"}


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

        def unload_model(self, reason: str = "manual") -> dict[str, object]:
            events.append(f"unload:{reason}")
            return {"result": "success"}

        def shutdown(self) -> None:
            events.append("shutdown")

    monkeypatch.setattr(server.TTSRuntime, "from_env", classmethod(lambda cls: FakeRuntime()))

    async def run_lifespan() -> None:
        async with server.app_lifespan(server.mcp) as context:
            assert "runtime" in context
            events.append("yielded")

    asyncio.run(run_lifespan())

    assert events == ["background-preload", "yielded", "unload:server shutdown", "shutdown"]


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
        assert tool.task_config.mode == "required"

    asyncio.run(run())


def test_speak_text_reports_progress_and_plays_audio() -> None:
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
    assert result["played"] is True
    assert progress.total == 3
    assert progress.current == 3
    assert progress.messages == [
        "Queued 1 chunk(s) for speech generation",
        "Generating chunk 1 of 1",
        "Playing chunk 1 of 1",
        "Playback complete",
    ]


def test_speak_text_chunks_and_plays_in_fifo_order(monkeypatch) -> None:
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
    assert runtime.generated_texts == chunked_text
    assert runtime.played_paths == [
        "/tmp/generated-audio/test-1.wav",
        "/tmp/generated-audio/test-2.wav",
        "/tmp/generated-audio/test-3.wav",
    ]
    assert progress.total == 7
    assert progress.current == 7
