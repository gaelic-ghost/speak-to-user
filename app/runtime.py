from __future__ import annotations

import builtins
import contextlib
import datetime as dt
import gc
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]
import soundfile as sf  # type: ignore[import-untyped]

# MARK: Constants

DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_IDLE_UNLOAD_SECONDS = 1200
ALLOWED_OUTPUT_FORMATS = {"wav", "flac"}
LANGUAGE_ALIASES = {
    "auto": "Auto",
    "en": "English",
    "english": "English",
    "zh": "Chinese",
    "chinese": "Chinese",
}

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
_ORIGINAL_PRINT = builtins.print
_PRINT_REDIRECT_LOCK = threading.RLock()
_PRINT_REDIRECT_THREAD_COUNTS: dict[int, int] = {}
_ACTIVE_PRINT_REDIRECTS = 0


# MARK: Module Helpers

def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _timestamp_value(value: dt.datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _normalize_device(raw: str | None) -> str:
    value = (raw or "auto").strip().lower()
    if value not in {"auto", "cpu", "mps"}:
        raise ValueError("SPEAK_TO_USER_DEVICE must be one of: auto, cpu, mps")
    return value


def _normalize_language(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("language must not be empty")
    return LANGUAGE_ALIASES.get(normalized.lower(), normalized)


def _normalize_filename_stem(value: str | None) -> str:
    if not value:
        return _utc_now().strftime("%Y%m%d-%H%M%S")

    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip())
    cleaned = cleaned.strip("-_")
    if not cleaned:
        raise ValueError("filename_stem must contain at least one alphanumeric character")
    return cleaned[:80]


def safe_print(*args: Any, **kwargs: Any) -> None:
    file = kwargs.get("file")
    thread_id = threading.get_ident()
    with _PRINT_REDIRECT_LOCK:
        should_redirect = _PRINT_REDIRECT_THREAD_COUNTS.get(thread_id, 0) > 0

    if should_redirect and (file is None or file is sys.stdout):
        kwargs["file"] = sys.stderr

    _ORIGINAL_PRINT(*args, **kwargs)


@contextlib.contextmanager
def _suppress_default_stdout_prints_for_current_thread() -> Any:
    thread_id = threading.get_ident()
    global _ACTIVE_PRINT_REDIRECTS

    with _PRINT_REDIRECT_LOCK:
        _PRINT_REDIRECT_THREAD_COUNTS[thread_id] = (
            _PRINT_REDIRECT_THREAD_COUNTS.get(thread_id, 0) + 1
        )
        _ACTIVE_PRINT_REDIRECTS += 1
        builtins.print = safe_print
    try:
        yield
    finally:
        with _PRINT_REDIRECT_LOCK:
            remaining = _PRINT_REDIRECT_THREAD_COUNTS[thread_id] - 1
            if remaining > 0:
                _PRINT_REDIRECT_THREAD_COUNTS[thread_id] = remaining
            else:
                del _PRINT_REDIRECT_THREAD_COUNTS[thread_id]

            _ACTIVE_PRINT_REDIRECTS -= 1
            if _ACTIVE_PRINT_REDIRECTS == 0:
                builtins.print = _ORIGINAL_PRINT


# MARK: TTS Runtime


class TTSRuntime:
    def __init__(
        self,
        *,
        model_id: str,
        idle_unload_seconds: int,
        output_dir: Path,
        device_preference: str,
    ) -> None:
        self.model_id = model_id
        self.idle_unload_seconds = idle_unload_seconds
        self.output_dir = output_dir
        self.device_preference = device_preference

        self._lock = threading.RLock()
        self._watchdog_started = False
        self._watchdog_stop = threading.Event()
        self._shutdown_requested = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._preload_complete = threading.Event()
        self._preload_thread: threading.Thread | None = None
        self._preload_in_progress = False
        self._speech_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._speech_worker_started = False
        self._speech_worker_thread: threading.Thread | None = None
        self._speech_in_progress = False
        self._speech_current_job_id: int | None = None
        self._speech_jobs_queued = 0
        self._speech_jobs_completed = 0
        self._speech_jobs_failed = 0
        self._speech_last_enqueued_at: dt.datetime | None = None
        self._speech_last_completed_at: dt.datetime | None = None
        self._speech_last_error: str | None = None

        self._model: Any | None = None
        self._resolved_device: str | None = None
        self._last_error: str | None = None
        self._last_used_monotonic: float | None = None
        self._last_used_at: dt.datetime | None = None
        self._last_loaded_at: dt.datetime | None = None
        self._last_unloaded_at: dt.datetime | None = None
        self._preload_started_at: dt.datetime | None = None
        self._preload_completed_at: dt.datetime | None = None

    @classmethod
    def from_env(cls) -> TTSRuntime:
        idle_unload_seconds = int(
            os.getenv("SPEAK_TO_USER_IDLE_UNLOAD_SECONDS", str(DEFAULT_IDLE_UNLOAD_SECONDS))
        )
        if idle_unload_seconds <= 0:
            raise ValueError("SPEAK_TO_USER_IDLE_UNLOAD_SECONDS must be a positive integer")

        output_dir = Path(os.getenv("SPEAK_TO_USER_OUTPUT_DIR", "generated-audio")).expanduser()
        if not output_dir.is_absolute():
            output_dir = (Path.cwd() / output_dir).resolve()

        return cls(
            model_id=os.getenv("SPEAK_TO_USER_MODEL_ID", DEFAULT_MODEL_ID),
            idle_unload_seconds=idle_unload_seconds,
            output_dir=output_dir,
            device_preference=_normalize_device(os.getenv("SPEAK_TO_USER_DEVICE")),
        )

    def start_watchdog(self) -> None:
        with self._lock:
            if self._watchdog_started:
                return
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                name="speak-to-user-idle-watchdog",
                daemon=True,
            )
            self._watchdog_started = True
            self._watchdog_thread.start()

    def start_speech_worker(self) -> None:
        with self._lock:
            if self._speech_worker_started:
                return
            self._speech_worker_thread = threading.Thread(
                target=self._speech_worker_loop,
                name="speak-to-user-speech-worker",
                daemon=True,
            )
            self._speech_worker_started = True
            self._speech_worker_thread.start()

    def preload(self) -> None:
        self.start_background_preload()

    def start_background_preload(self) -> bool:
        self.start_watchdog()
        self.start_speech_worker()
        with self._lock:
            if self._shutdown_requested.is_set():
                return False
            if self._model is not None or self._preload_in_progress:
                return False

            self._preload_in_progress = True
            self._preload_started_at = _utc_now()
            self._preload_completed_at = None
            self._preload_complete.clear()
            self._preload_thread = threading.Thread(
                target=self._background_preload_worker,
                name="speak-to-user-preload",
                daemon=True,
            )
            self._preload_thread.start()
            return True

    def shutdown(self) -> None:
        self._shutdown_requested.set()
        self._watchdog_stop.set()
        if self._speech_worker_started:
            self._speech_queue.put(None)
        preload_thread = self._preload_thread
        if preload_thread is not None:
            preload_thread.join(timeout=1)
        with self._lock:
            if self._model is not None:
                self._unload_locked(reason="server shutdown")
        thread = self._watchdog_thread
        if thread is not None:
            thread.join(timeout=1)
        speech_thread = self._speech_worker_thread
        if speech_thread is not None:
            speech_thread.join(timeout=1)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_payload_locked()

    def load_model(self) -> dict[str, Any]:
        wait_for_preload = False
        with self._lock:
            if self._shutdown_requested.is_set():
                raise RuntimeError("runtime is shutting down")
            if (
                self._preload_in_progress
                and self._preload_thread is not None
                and self._preload_thread.ident != threading.get_ident()
            ):
                wait_for_preload = True

        if wait_for_preload:
            self._preload_complete.wait()

        with self._lock:
            if self._model is not None:
                self._touch_locked()
                return {
                    "result": "success",
                    "loaded": True,
                    "info": "model already loaded",
                    **self._status_payload_locked(),
                }

            try:
                loaded_model = self._load_model_impl()
            except Exception as exc:
                self._last_error = str(exc)
                raise

            now = _utc_now()
            if self._shutdown_requested.is_set():
                self._release_model_resources(loaded_model, resolved_device=self._resolved_device)
                self._resolved_device = None
                self._last_unloaded_at = now
                return {
                    "result": "success",
                    "loaded": False,
                    "info": "model load aborted during shutdown",
                    **self._status_payload_locked(),
                }

            self._model = loaded_model
            self._last_error = None
            self._last_loaded_at = now
            self._touch_locked(now)
            return {
                "result": "success",
                "loaded": True,
                "info": "model loaded",
                **self._status_payload_locked(),
            }

    def unload_model(self, reason: str = "manual") -> dict[str, Any]:
        with self._lock:
            if self._model is None:
                return {
                    "result": "success",
                    "loaded": False,
                    "info": "model already unloaded",
                    **self._status_payload_locked(),
                }

            self._unload_locked(reason=reason)
            return {
                "result": "success",
                "loaded": False,
                "info": f"model unloaded ({reason})",
                **self._status_payload_locked(),
            }

    def set_idle_unload_timeout(self, seconds: int) -> dict[str, Any]:
        if seconds <= 0:
            raise ValueError("seconds must be greater than zero")
        with self._lock:
            self.idle_unload_seconds = seconds
            return self._status_payload_locked()

    def enqueue_speech(
        self,
        *,
        chunks: list[str],
        voice_description: str,
        language: str = "en",
    ) -> dict[str, Any]:
        normalized_chunks = self._normalize_chunks(chunks)
        normalized_description = voice_description.strip()
        if not normalized_description:
            raise ValueError("voice_description must not be empty")
        normalized_language = _normalize_language(language)

        self.start_speech_worker()

        with self._lock:
            self._speech_jobs_queued += 1
            job_id = self._speech_jobs_queued
            enqueued_at = _utc_now()
            self._speech_last_enqueued_at = enqueued_at

        self._speech_queue.put(
            {
                "job_id": job_id,
                "chunks": normalized_chunks,
                "voice_description": normalized_description,
                "language": normalized_language,
            }
        )

        with self._lock:
            return {
                "result": "success",
                "queued": True,
                "job_id": job_id,
                "chunked": len(normalized_chunks) > 1,
                "chunk_count": len(normalized_chunks),
                "language": normalized_language,
                "enqueued_at": enqueued_at.isoformat(),
                **self._speech_status_payload_locked(),
            }

    def generate_audio(
        self,
        *,
        text: str,
        voice_description: str,
        language: str = "en",
        output_format: str = "wav",
        filename_stem: str | None = None,
    ) -> dict[str, Any]:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("text must not be empty")

        normalized_description = voice_description.strip()
        if not normalized_description:
            raise ValueError("voice_description must not be empty")

        normalized_format = output_format.strip().lower()
        if normalized_format not in ALLOWED_OUTPUT_FORMATS:
            raise ValueError("output_format must be one of: wav, flac")

        normalized_language = _normalize_language(language)
        output_path = self._build_output_path(filename_stem, normalized_format)

        synthesis_result = self._synthesize_audio(
            text=normalized_text,
            voice_description=normalized_description,
            language=normalized_language,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, synthesis_result["waveform"], synthesis_result["sample_rate"])

        return {
            "result": "success",
            "path": str(output_path),
            "format": normalized_format,
            "sample_rate": synthesis_result["sample_rate"],
            "sample_count": synthesis_result["sample_count"],
            "duration_seconds": synthesis_result["duration_seconds"],
            "model_id": synthesis_result["model_id"],
            "device": synthesis_result["device"],
            "language": synthesis_result["language"],
        }

    def generate_speech_buffer(
        self,
        *,
        chunks: list[str],
        voice_description: str,
        language: str = "en",
    ) -> dict[str, Any]:
        if not chunks:
            raise ValueError("chunks must not be empty")

        normalized_description = voice_description.strip()
        if not normalized_description:
            raise ValueError("voice_description must not be empty")

        normalized_language = _normalize_language(language)
        normalized_chunks = self._normalize_chunks(chunks)
        batch_result = self._synthesize_audio_batch(
            texts=normalized_chunks,
            voice_description=normalized_description,
            language=normalized_language,
        )
        waveform = self._concatenate_waveforms(batch_result["waveforms"])

        return {
            "result": "success",
            "format": "wav",
            "chunked": len(normalized_chunks) > 1,
            "chunk_count": len(normalized_chunks),
            "sample_rate": batch_result["sample_rate"],
            "sample_count": len(waveform),
            "duration_seconds": round(len(waveform) / batch_result["sample_rate"], 3),
            "model_id": batch_result["model_id"],
            "device": batch_result["device"],
            "language": batch_result["language"],
            "waveform": waveform,
        }

    def play_audio_buffer(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> dict[str, Any]:
        sd.play(waveform, sample_rate)
        sd.wait()
        return {
            "result": "success",
            "played": True,
            "player": "sounddevice",
        }

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(timeout=5):
            with self._lock:
                if self._model is None or self._last_used_monotonic is None:
                    continue
                idle_seconds = time.monotonic() - self._last_used_monotonic
                if idle_seconds >= self.idle_unload_seconds:
                    self._unload_locked(reason="idle timeout exceeded")

    def _background_preload_worker(self) -> None:
        try:
            try:
                self.load_model()
            except Exception:
                # load_model() already records last_error for status reporting
                pass
        finally:
            with self._lock:
                self._preload_in_progress = False
                self._preload_completed_at = _utc_now()
                self._preload_complete.set()

    def _status_payload_locked(self) -> dict[str, Any]:
        return {
            "ready": self._model is not None and self._last_error is None,
            "model_loaded": self._model is not None,
            "preload_in_progress": self._preload_in_progress,
            "device": self._resolved_device,
            "model_id": self.model_id,
            "idle_unload_seconds": self.idle_unload_seconds,
            "last_used_at": _timestamp_value(self._last_used_at),
            "last_loaded_at": _timestamp_value(self._last_loaded_at),
            "last_unloaded_at": _timestamp_value(self._last_unloaded_at),
            "preload_started_at": _timestamp_value(self._preload_started_at),
            "preload_completed_at": _timestamp_value(self._preload_completed_at),
            "output_dir": str(self.output_dir),
            "last_error": self._last_error,
            **self._speech_status_payload_locked(),
        }

    def _speech_status_payload_locked(self) -> dict[str, Any]:
        return {
            "speech_in_progress": self._speech_in_progress,
            "speech_current_job_id": self._speech_current_job_id,
            "speech_queue_depth": self._speech_queue.qsize(),
            "speech_jobs_queued": self._speech_jobs_queued,
            "speech_jobs_completed": self._speech_jobs_completed,
            "speech_jobs_failed": self._speech_jobs_failed,
            "speech_last_enqueued_at": _timestamp_value(self._speech_last_enqueued_at),
            "speech_last_completed_at": _timestamp_value(self._speech_last_completed_at),
            "speech_last_error": self._speech_last_error,
        }

    def _speech_worker_loop(self) -> None:
        while True:
            job = self._speech_queue.get()
            if job is None:
                return

            job_id = int(job["job_id"])
            chunks = list(job["chunks"])
            voice_description = str(job["voice_description"])
            language = str(job["language"])

            with self._lock:
                self._speech_in_progress = True
                self._speech_current_job_id = job_id
                self._speech_last_error = None

            try:
                buffer_result = self.generate_speech_buffer(
                    chunks=chunks,
                    voice_description=voice_description,
                    language=language,
                )
                self.play_audio_buffer(
                    buffer_result["waveform"],
                    buffer_result["sample_rate"],
                )
            except Exception as exc:
                with self._lock:
                    self._speech_jobs_failed += 1
                    self._speech_last_error = str(exc)
                    self._last_error = str(exc)
            else:
                with self._lock:
                    self._speech_jobs_completed += 1
                    self._speech_last_completed_at = _utc_now()
                    self._speech_last_error = None
            finally:
                with self._lock:
                    self._speech_in_progress = False
                    self._speech_current_job_id = None

    def _touch_locked(self, value: dt.datetime | None = None) -> None:
        self._last_used_at = value or _utc_now()
        self._last_used_monotonic = time.monotonic()

    def _build_output_path(self, filename_stem: str | None, output_format: str) -> Path:
        return self.output_dir / f"{_normalize_filename_stem(filename_stem)}.{output_format}"

    def _normalize_chunks(self, chunks: list[str]) -> list[str]:
        normalized_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        if not normalized_chunks:
            raise ValueError("chunks must not be empty")
        return normalized_chunks

    def _synthesize_audio(
        self,
        *,
        text: str,
        voice_description: str,
        language: str,
    ) -> dict[str, Any]:
        batch_result = self._synthesize_audio_batch(
            texts=[text],
            voice_description=voice_description,
            language=language,
        )
        waveform = batch_result["waveforms"][0]
        sample_rate = batch_result["sample_rate"]
        sample_count = len(waveform)
        duration_seconds = round(sample_count / sample_rate, 3) if sample_rate else None
        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "sample_count": sample_count,
            "duration_seconds": duration_seconds,
            "model_id": batch_result["model_id"],
            "device": batch_result["device"],
            "language": batch_result["language"],
        }

    def _synthesize_audio_batch(
        self,
        *,
        texts: list[str],
        voice_description: str,
        language: str,
    ) -> dict[str, Any]:
        with self._lock:
            model_missing = self._model is None

        if model_missing:
            self.load_model()

        with self._lock:
            assert self._model is not None
            try:
                wavs, sample_rate = self._model.generate_voice_design(
                    text=texts,
                    language=[language] * len(texts),
                    instruct=[voice_description] * len(texts),
                )
            except Exception as exc:
                self._last_error = str(exc)
                raise

            waveforms = [self._coerce_waveform(waveform) for waveform in wavs]
            now = _utc_now()
            self._last_error = None
            self._touch_locked(now)

            return {
                "waveforms": waveforms,
                "sample_rate": sample_rate,
                "model_id": self.model_id,
                "device": self._resolved_device,
                "language": language,
            }

    def _concatenate_waveforms(self, waveforms: list[Any]) -> np.ndarray:
        if not waveforms:
            raise ValueError("waveforms must not be empty")
        return np.concatenate([np.asarray(waveform, dtype=np.float32) for waveform in waveforms])

    def _load_model_impl(self) -> Any:
        with _suppress_default_stdout_prints_for_current_thread():
            import torch
            from qwen_tts import Qwen3TTSModel  # type: ignore[import-untyped]

            resolved_device = self._resolve_device(torch)
            self._resolved_device = resolved_device
            return Qwen3TTSModel.from_pretrained(
                self.model_id,
                device_map=resolved_device,
            )

    def _resolve_device(self, torch_module: Any) -> str:
        if self.device_preference == "cpu":
            return "cpu"
        if self.device_preference == "mps":
            if not torch_module.backends.mps.is_available():
                raise RuntimeError("SPEAK_TO_USER_DEVICE=mps but torch MPS is unavailable")
            return "mps"

        if torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _unload_locked(self, *, reason: str) -> None:
        model = self._model
        resolved_device = self._resolved_device
        self._model = None
        self._last_unloaded_at = _utc_now()
        self._release_model_resources(model, resolved_device=resolved_device)

    def _release_model_resources(self, model: Any | None, *, resolved_device: str | None) -> None:
        del model
        gc.collect()

        try:
            import torch
        except ImportError:
            return

        if resolved_device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif (
            resolved_device is not None
            and resolved_device.startswith("cuda")
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()

    def _coerce_waveform(self, waveform: Any) -> Any:
        if hasattr(waveform, "detach"):
            waveform = waveform.detach()
        if hasattr(waveform, "cpu"):
            waveform = waveform.cpu()
        if hasattr(waveform, "numpy"):
            waveform = waveform.numpy()
        waveform_array = np.asarray(waveform, dtype=np.float32)
        if waveform_array.ndim > 1:
            waveform_array = np.squeeze(waveform_array)
        return waveform_array
