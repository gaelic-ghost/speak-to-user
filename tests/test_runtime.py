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
        text: str,
        language: str,
        instruct: str,
    ) -> tuple[list[list[float]], int]:
        self.calls.append({"text": text, "language": language, "instruct": instruct})
        return [self.waveform], self.sample_rate


def make_runtime(tmp_path: Path) -> TTSRuntime:
    return TTSRuntime(
        model_id="Qwen/test-model",
        idle_unload_seconds=10,
        output_dir=tmp_path / "generated-audio",
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
    assert runtime.status()["preload_in_progress"] is False


def test_unload_model_clears_loaded_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    monkeypatch.setattr(runtime, "_load_model_impl", lambda: FakeModel())
    runtime.load_model()

    result = runtime.unload_model()

    assert result["result"] == "success"
    assert result["loaded"] is False
    assert runtime.status()["model_loaded"] is False


def test_set_idle_timeout_rejects_non_positive(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)

    with pytest.raises(ValueError, match="greater than zero"):
        runtime.set_idle_unload_timeout(0)


# MARK: Audio Generation

def test_generate_audio_writes_file_and_returns_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    fake_model = FakeModel()
    writes: list[dict[str, object]] = []

    monkeypatch.setattr(runtime, "_load_model_impl", lambda: fake_model)
    monkeypatch.setattr(
        "app.runtime.sf.write",
        lambda path, waveform, sample_rate: writes.append(
            {"path": Path(path), "waveform": list(waveform), "sample_rate": sample_rate}
        ),
    )

    result = runtime.generate_audio(
        text="Hello there",
        voice_description="Warm and calm",
        language="en",
        output_format="wav",
        filename_stem="greeting",
    )

    assert result["result"] == "success"
    assert result["path"].endswith("greeting.wav")
    assert result["sample_rate"] == 24000
    assert result["sample_count"] == 3
    assert cast(Path, writes[0]["path"]).name == "greeting.wav"
    assert fake_model.calls[0]["language"] == "English"


def test_generate_audio_reloads_after_idle_unload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    models = [FakeModel(), FakeModel()]
    load_count = {"value": 0}

    def load_impl() -> FakeModel:
        model = models[load_count["value"]]
        load_count["value"] += 1
        return model

    monkeypatch.setattr(runtime, "_load_model_impl", load_impl)
    monkeypatch.setattr("app.runtime.sf.write", lambda *args, **kwargs: None)

    runtime.generate_audio(text="First", voice_description="Plain")
    runtime.unload_model(reason="idle timeout exceeded")
    runtime.generate_audio(text="Second", voice_description="Plain")

    assert load_count["value"] == 2


def test_watchdog_unloads_after_idle_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    monkeypatch.setattr(runtime, "_load_model_impl", lambda: FakeModel())
    runtime.load_model()
    runtime._last_used_monotonic = time_marker = 100.0

    monotonic_values = iter([time_marker + 11.0])
    monkeypatch.setattr("app.runtime.time.monotonic", lambda: next(monotonic_values))
    wait_values = iter([False, True])
    monkeypatch.setattr(runtime._watchdog_stop, "wait", lambda timeout: next(wait_values))

    runtime._watchdog_loop()

    assert runtime.status()["model_loaded"] is False


# MARK: Background Preload

def test_start_background_preload_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    thread_starts = {"value": 0}

    class FakeThread:
        ident = 999

        def __init__(self, *, target: object, name: str, daemon: bool) -> None:
            self.target = target
            self.name = name
            self.daemon = daemon

        def start(self) -> None:
            thread_starts["value"] += 1

        def join(self, timeout: float | None = None) -> None:
            return None

    monkeypatch.setattr("app.runtime.threading.Thread", FakeThread)

    assert runtime.start_background_preload() is True
    assert runtime.start_background_preload() is False
    assert thread_starts["value"] == 2
    assert runtime.status()["preload_in_progress"] is True


def test_background_preload_worker_updates_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    fake_now = iter(
        [
            dt.datetime(2026, 3, 14, 17, 0, tzinfo=dt.UTC),
            dt.datetime(2026, 3, 14, 17, 0, 5, tzinfo=dt.UTC),
            dt.datetime(2026, 3, 14, 17, 0, 10, tzinfo=dt.UTC),
        ]
    )

    monkeypatch.setattr("app.runtime._utc_now", lambda: next(fake_now))
    monkeypatch.setattr(runtime, "_load_model_impl", lambda: FakeModel())

    class FakeThread:
        ident = 999

        def __init__(self, *, target: object, name: str, daemon: bool) -> None:
            self.target = target
            self.name = name
            self.daemon = daemon

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            return None

    monkeypatch.setattr("app.runtime.threading.Thread", FakeThread)

    runtime.start_background_preload()
    runtime._preload_thread = threading.current_thread()
    runtime._background_preload_worker()

    status = runtime.status()
    assert status["model_loaded"] is True
    assert status["preload_in_progress"] is False
    assert status["preload_started_at"] == "2026-03-14T17:00:00+00:00"
    assert status["preload_completed_at"] == "2026-03-14T17:00:10+00:00"


def test_background_preload_worker_records_error_without_raising(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)

    def boom() -> FakeModel:
        raise RuntimeError("preload failed")

    monkeypatch.setattr(runtime, "_load_model_impl", boom)

    runtime._preload_in_progress = True
    runtime._background_preload_worker()

    status = runtime.status()
    assert status["model_loaded"] is False
    assert status["preload_in_progress"] is False
    assert status["last_error"] == "preload failed"


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
