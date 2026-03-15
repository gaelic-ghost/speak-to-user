from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
from typing import cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.runtime as runtime_module
from app.runtime import TTSRuntime


class FakeModel:
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
    ) -> tuple[list[list[float]], int]:
        self.calls.append({"text": text, "language": language, "instruct": instruct})
        batch_size = len(text) if isinstance(text, list) else 1
        return [list(self.waveform) for _ in range(batch_size)], self.sample_rate


def make_runtime(tmp_path: Path) -> TTSRuntime:
    del tmp_path
    return TTSRuntime(
        model_id="Qwen/test-model",
        device_preference="cpu",
    )


# MARK: Module Helpers

def test_stdout_suppression_uses_module_level_safe_print() -> None:
    original_print = runtime_module.builtins.print

    with runtime_module._suppress_default_stdout_prints_for_current_thread():
        assert runtime_module.builtins.print is runtime_module.safe_print

    assert runtime_module.builtins.print is original_print


# MARK: Runtime Lifecycle

def test_load_model_success_sets_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    fake_model = FakeModel()
    monkeypatch.setattr(runtime, "_load_model_impl", lambda: fake_model)

    result = runtime.load_model()

    assert result["result"] == "success"
    assert result["loaded"] is True
    assert runtime.status()["model_loaded"] is True
    assert runtime.status()["last_error"] is None


def test_load_model_failure_records_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)

    def boom() -> FakeModel:
        raise RuntimeError("load failed")

    monkeypatch.setattr(runtime, "_load_model_impl", boom)

    with pytest.raises(RuntimeError, match="load failed"):
        runtime.load_model()

    assert runtime.status()["last_error"] == "load failed"


def test_preload_starts_worker_and_loads_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    calls = {"worker": 0, "load": 0}

    def fake_start_speech_worker() -> bool:
        calls["worker"] = 1
        return True

    def fake_load_model() -> dict[str, object]:
        calls["load"] = 1
        return {"result": "success", "loaded": True}

    monkeypatch.setattr(runtime, "start_speech_worker", fake_start_speech_worker)
    monkeypatch.setattr(runtime, "load_model", fake_load_model)

    result = runtime.preload()

    assert result == {"result": "success", "loaded": True}
    assert calls == {"worker": 1, "load": 1}


def test_shutdown_releases_loaded_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    monkeypatch.setattr(runtime, "_load_model_impl", lambda: FakeModel())
    released: list[tuple[object | None, str | None]] = []
    monkeypatch.setattr(
        runtime,
        "_release_model_resources",
        lambda model, *, resolved_device: released.append((model, resolved_device)),
    )

    runtime.load_model()
    runtime.shutdown()

    assert runtime.status()["model_loaded"] is False
    assert len(released) == 1


# MARK: Queue

def test_speak_text_enqueues_one_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
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
    assert result["speech_phase"] == "idle"
    assert result["speech_jobs_queued"] == 1
    assert result["speech_queue_depth"] == 1
    assert worker_starts["value"] == 1


def test_speak_text_assigns_unique_job_ids_under_concurrent_enqueue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
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
    assert runtime.status()["speech_jobs_queued"] == 2
    queued_jobs = [runtime._speech_queue.get_nowait(), runtime._speech_queue.get_nowait()]
    assert [cast(dict[str, object], job)["job_id"] for job in queued_jobs] == [1, 2]


def test_speech_worker_plays_queued_jobs_in_fifo_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    played_jobs: list[dict[str, object]] = []
    monkeypatch.setattr(runtime, "start_speech_worker", lambda: True)

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
        }

    monkeypatch.setattr(runtime, "play_speech_chunks", fake_play_speech_chunks)

    runtime.speak_text(chunks=["first"], voice_description="warm", language="en")
    runtime.speak_text(chunks=["second"], voice_description="warm", language="en")
    runtime._speech_queue.put_nowait(None)
    runtime._speech_worker_loop()

    assert played_jobs == [
        {"chunks": ["first"], "voice_description": "warm", "language": "English"},
        {"chunks": ["second"], "voice_description": "warm", "language": "English"},
    ]
    status = runtime.status()
    assert status["speech_jobs_completed"] == 2
    assert status["speech_phase"] == "idle"
    assert status["speech_queue_depth"] == 0


def test_speech_worker_records_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    monkeypatch.setattr(runtime, "start_speech_worker", lambda: True)

    def boom(**kwargs: object) -> dict[str, object]:
        del kwargs
        raise RuntimeError("playback failed")

    monkeypatch.setattr(runtime, "play_speech_chunks", boom)

    runtime.speak_text(chunks=["Hello there"], voice_description="warm", language="en")
    runtime._speech_queue.put_nowait(None)
    runtime._speech_worker_loop()

    status = runtime.status()
    assert status["speech_jobs_failed"] == 1
    assert status["speech_phase"] == "idle"
    assert status["speech_last_error"] == "playback failed"
    assert status["last_error"] == "playback failed"


def test_speech_worker_reports_synthesizing_phase_before_playback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    phases: list[str] = []
    monkeypatch.setattr(runtime, "start_speech_worker", lambda: True)

    def fake_synthesize_audio_chunk(**kwargs: object) -> dict[str, object]:
        del kwargs
        phases.append(cast(str, runtime.status()["speech_phase"]))
        return {
            "waveform": runtime_module.np.array([0.0, 0.1, 0.2], dtype=runtime_module.np.float32),
            "sample_rate": 24000,
            "model_id": "Qwen/test-model",
            "device": "cpu",
            "language": "English",
        }

    class FakeStream:
        def start(self) -> None:
            return None

        def write(self, waveform: runtime_module.np.ndarray) -> bool:
            del waveform
            phases.append(cast(str, runtime.status()["speech_phase"]))
            return False

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_synthesize_audio_chunk", fake_synthesize_audio_chunk)
    monkeypatch.setattr(runtime, "_open_output_stream", lambda **kwargs: FakeStream())

    runtime.speak_text(chunks=["Hello there"], voice_description="warm", language="en")
    runtime._speech_queue.put_nowait(None)
    runtime._speech_worker_loop()

    assert phases == ["synthesizing", "playing"]


# MARK: Playback

def test_play_speech_chunks_generates_and_writes_one_chunk_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    generated_chunks: list[dict[str, object]] = []
    played: list[dict[str, object]] = []
    opened_streams: list[FakeStream] = []

    class FakeStream:
        def __init__(self) -> None:
            self.closed = False
            self.stopped = False

        def start(self) -> None:
            return None

        def write(self, waveform: runtime_module.np.ndarray) -> bool:
            played.append({"waveform": waveform.tolist()})
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
        generated_chunks.append(
            {
                "text": cast(str, kwargs["text"]),
                "language": kwargs["language"],
                "voice_description": kwargs["voice_description"],
            }
        )
        waveform = [0.0, 0.1, 0.2] if kwargs["text"] == "Hello there" else [0.3, 0.4, 0.5]
        return {
            "waveform": runtime_module.np.array(waveform, dtype=runtime_module.np.float32),
            "sample_rate": 24000,
            "model_id": "Qwen/test-model",
            "device": "cpu",
            "language": kwargs["language"],
        }

    monkeypatch.setattr(runtime, "_synthesize_audio_chunk", synthesize_audio_chunk)
    monkeypatch.setattr(runtime, "_open_output_stream", open_output_stream)

    result = runtime.play_speech_chunks(
        chunks=["Hello there", "General Kenobi"],
        voice_description="Warm and calm",
        language="en",
    )

    assert result["played"] is True
    assert result["sample_count"] == 6
    assert result["player"] == "sounddevice-stream"
    assert generated_chunks == [
        {
            "text": "Hello there",
            "language": "English",
            "voice_description": "Warm and calm",
        },
        {
            "text": "General Kenobi",
            "language": "English",
            "voice_description": "Warm and calm",
        }
    ]
    assert len(opened_streams) == 1
    assert opened_streams[0].stopped is True
    assert opened_streams[0].closed is True
    assert len(played) == 2
    assert cast(list[float], played[0]["waveform"]) == pytest.approx([0.0, 0.1, 0.2])
    assert cast(list[float], played[1]["waveform"]) == pytest.approx([0.3, 0.4, 0.5])


def test_play_speech_chunks_starts_playback_after_small_preroll_then_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    events: list[str] = []

    class FakeStream:
        def start(self) -> None:
            return None

        def write(self, waveform: runtime_module.np.ndarray) -> bool:
            del waveform
            events.append("write")
            return False

        def stop(self) -> None:
            events.append("stop")

        def close(self) -> None:
            events.append("close")

    def synthesize_audio_chunk(**kwargs: object) -> dict[str, object]:
        events.append(f"synthesize:{cast(str, kwargs['text'])}")
        waveform = runtime_module.np.zeros(6000, dtype=runtime_module.np.float32)
        return {
            "waveform": waveform,
            "sample_rate": 24000,
            "model_id": "Qwen/test-model",
            "device": "cpu",
            "language": kwargs["language"],
        }

    def open_output_stream(**kwargs: object) -> FakeStream:
        del kwargs
        events.append("open")
        return FakeStream()

    monkeypatch.setattr(runtime, "_synthesize_audio_chunk", synthesize_audio_chunk)
    monkeypatch.setattr(runtime, "_open_output_stream", open_output_stream)

    runtime.play_speech_chunks(
        chunks=["First chunk", "Second chunk", "Third chunk"],
        voice_description="Warm and calm",
        language="en",
    )

    assert events[:2] == ["synthesize:First chunk", "synthesize:Second chunk"]
    assert "open" in events
    open_index = events.index("open")
    first_write_index = events.index("write")
    assert open_index < first_write_index
    assert events[-2:] == ["stop", "close"]
    assert events.count("write") == 3


def test_normalize_language_accepts_locale_variants() -> None:
    assert runtime_module._normalize_language("en-US") == "English"
    assert runtime_module._normalize_language("pt_BR") == "Portuguese"


# MARK: Status

def test_status_ready_requires_loaded_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)

    assert runtime.status()["ready"] is False

    monkeypatch.setattr(runtime, "_load_model_impl", lambda: FakeModel())
    runtime.load_model()

    assert runtime.status()["ready"] is True

    runtime.shutdown()
    assert runtime.status()["ready"] is False


def test_load_model_releases_model_when_shutdown_requested_mid_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    released: list[tuple[object | None, str | None]] = []

    def load_impl() -> FakeModel:
        runtime._shutdown_requested.set()
        return FakeModel()

    monkeypatch.setattr(runtime, "_load_model_impl", load_impl)
    monkeypatch.setattr(
        runtime,
        "_release_model_resources",
        lambda model, *, resolved_device: released.append((model, resolved_device)),
    )

    result = runtime.load_model()

    assert result["result"] == "success"
    assert result["loaded"] is False
    assert result["info"] == "model load aborted during shutdown"
    assert runtime.status()["model_loaded"] is False
    assert len(released) == 1
    assert isinstance(released[0][0], FakeModel)
    assert released[0][1] is None


def test_status_reports_timestamps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    fake_now = dt.datetime(2026, 3, 14, 17, 0, tzinfo=dt.UTC)
    monkeypatch.setattr("app.runtime._utc_now", lambda: fake_now)
    monkeypatch.setattr(runtime, "_load_model_impl", lambda: FakeModel())

    runtime.load_model()
    status = runtime.status()

    assert status["last_loaded_at"] == fake_now.isoformat()
    assert status["last_used_at"] == fake_now.isoformat()


# MARK: Device Resolution

def test_resolve_device_prefers_mps_when_available(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    runtime.device_preference = "auto"
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: True),
        )
    )

    resolved = runtime._resolve_device(fake_torch)

    assert resolved == "mps"
