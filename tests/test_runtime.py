from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace
import sys
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


def test_speech_worker_plays_queued_jobs_in_fifo_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    played_jobs: list[dict[str, object]] = []

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

    def fake_synthesize_audio_batch(**kwargs: object) -> dict[str, object]:
        del kwargs
        phases.append(cast(str, runtime.status()["speech_phase"]))
        return {
            "waveforms": [
                runtime_module.np.array([0.0, 0.1, 0.2], dtype=runtime_module.np.float32)
            ],
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

        def close(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_synthesize_audio_batch", fake_synthesize_audio_batch)
    monkeypatch.setattr(runtime, "_open_output_stream", lambda **kwargs: FakeStream())

    runtime.speak_text(chunks=["Hello there"], voice_description="warm", language="en")
    runtime._speech_queue.put_nowait(None)
    runtime._speech_worker_loop()

    assert phases == ["synthesizing", "playing"]


# MARK: Playback

def test_play_speech_chunks_uses_one_batch_model_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    generated_batches: list[dict[str, object]] = []
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

    def synthesize_audio_batch(**kwargs: object) -> dict[str, object]:
        generated_batches.append(
            {
                "texts": list(cast(list[str], kwargs["texts"])),
                "language": kwargs["language"],
                "voice_description": kwargs["voice_description"],
            }
        )
        return {
            "waveforms": [
                runtime_module.np.array([0.0, 0.1, 0.2], dtype=runtime_module.np.float32),
                runtime_module.np.array([0.3, 0.4, 0.5], dtype=runtime_module.np.float32),
            ],
            "sample_rate": 24000,
            "model_id": "Qwen/test-model",
            "device": "cpu",
            "language": kwargs["language"],
        }

    monkeypatch.setattr(runtime, "_synthesize_audio_batch", synthesize_audio_batch)
    monkeypatch.setattr(runtime, "_open_output_stream", open_output_stream)

    result = runtime.play_speech_chunks(
        chunks=["Hello there", "General Kenobi"],
        voice_description="Warm and calm",
        language="en",
    )

    assert result["played"] is True
    assert result["sample_count"] == 6
    assert result["player"] == "sounddevice-stream"
    assert runtime.status()["speech_phase"] == "playing"
    assert generated_batches == [
        {
            "texts": ["Hello there", "General Kenobi"],
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
