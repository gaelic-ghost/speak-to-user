from __future__ import annotations

import builtins
import contextlib
import datetime as dt
import os
import queue
import sys
import threading
from typing import Any

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]

# MARK: Constants

DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_PLAYBACK_PREROLL_SECONDS = 0.25
DEFAULT_SPEECH_QUEUE_MAXSIZE = 32
DEFAULT_SPEECH_PHASE = "idle"
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


# MARK: Runtime


class TTSRuntime:
    def __init__(
        self,
        *,
        model_id: str,
        device_preference: str,
        speech_queue_maxsize: int = DEFAULT_SPEECH_QUEUE_MAXSIZE,
    ) -> None:
        self.model_id = model_id
        self.device_preference = device_preference
        self.speech_queue_maxsize = speech_queue_maxsize

        self._lock = threading.RLock()
        self._shutdown_requested = threading.Event()
        self._speech_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(
            maxsize=speech_queue_maxsize
        )
        self._speech_worker_thread: threading.Thread | None = None
        self._speech_worker_started = False

        self._model: Any | None = None
        self._resolved_device: str | None = None
        self._last_error: str | None = None
        self._last_used_at: dt.datetime | None = None
        self._last_loaded_at: dt.datetime | None = None

        self._speech_in_progress = False
        self._speech_phase = DEFAULT_SPEECH_PHASE
        self._speech_current_job_id: int | None = None
        self._speech_current_chunk_index: int | None = None
        self._speech_current_chunk_count: int | None = None
        self._speech_jobs_queued = 0
        self._speech_jobs_completed = 0
        self._speech_jobs_failed = 0
        self._speech_last_enqueued_at: dt.datetime | None = None
        self._speech_last_completed_at: dt.datetime | None = None
        self._speech_last_error: str | None = None

    @classmethod
    def from_env(cls) -> TTSRuntime:
        return cls(
            model_id=os.getenv("SPEAK_TO_USER_MODEL_ID", DEFAULT_MODEL_ID),
            device_preference=_normalize_device(os.getenv("SPEAK_TO_USER_DEVICE")),
        )

    def preload(self) -> dict[str, Any]:
        self.start_speech_worker()
        return self.load_model()

    def start_speech_worker(self) -> bool:
        with self._lock:
            if self._shutdown_requested.is_set():
                return False
            if self._speech_worker_started:
                return False

            self._speech_worker_thread = threading.Thread(
                target=self._speech_worker_loop,
                name="speak-to-user-speech-worker",
                daemon=True,
            )
            self._speech_worker_started = True
            self._speech_worker_thread.start()
            return True

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown_requested.set()
            worker_thread = self._speech_worker_thread

        if worker_thread is not None:
            while True:
                try:
                    self._speech_queue.put_nowait(None)
                    break
                except queue.Full:
                    if not worker_thread.is_alive():
                        break
                    worker_thread.join(timeout=0.05)
            worker_thread.join(timeout=1)

        with self._lock:
            if self._model is not None:
                self._unload_locked()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_payload_locked()

    def load_model(self) -> dict[str, Any]:
        with self._lock:
            if self._shutdown_requested.is_set():
                raise RuntimeError("runtime is shutting down")
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
            with self._lock:
                self._last_error = str(exc)
            raise

        with self._lock:
            if self._shutdown_requested.is_set():
                self._release_model_resources(loaded_model, resolved_device=self._resolved_device)
                self._resolved_device = None
                return {
                    "result": "success",
                    "loaded": False,
                    "info": "model load aborted during shutdown",
                    **self._status_payload_locked(),
                }

            self._model = loaded_model
            self._last_error = None
            self._last_loaded_at = _utc_now()
            self._touch_locked(self._last_loaded_at)
            return {
                "result": "success",
                "loaded": True,
                "info": "model loaded",
                **self._status_payload_locked(),
            }

    def speak_text(
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

        with self._lock:
            if self._shutdown_requested.is_set():
                raise RuntimeError("runtime is shutting down and cannot accept speech jobs")
            job_id = self._speech_jobs_queued + 1
            job = {
                "job_id": job_id,
                "chunks": normalized_chunks,
                "voice_description": normalized_description,
                "language": normalized_language,
            }

        self.start_speech_worker()

        try:
            self._speech_queue.put_nowait(job)
        except queue.Full as exc:
            with self._lock:
                self._speech_last_error = "speech queue is full"
                self._last_error = self._speech_last_error
            raise RuntimeError("speech queue is full") from exc

        with self._lock:
            enqueued_at = _utc_now()
            self._speech_jobs_queued = job_id
            self._speech_last_enqueued_at = enqueued_at
            self._speech_last_error = None
            return {
                "result": "success",
                "queued": True,
                "job_id": job_id,
                "chunked": len(normalized_chunks) > 1,
                "chunk_count": len(normalized_chunks),
                "language": normalized_language,
                "enqueued_at": enqueued_at.isoformat(),
                "playback_mode": "in-process-queue",
                **self._speech_status_payload_locked(),
            }

    def play_speech_chunks(
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
        with self._lock:
            self._speech_phase = "synthesizing"

        batch_result = self._synthesize_audio_batch(
            texts=normalized_chunks,
            voice_description=normalized_description,
            language=normalized_language,
        )
        waveforms = [
            self._prepare_waveform_for_output_stream(waveform)
            for waveform in batch_result["waveforms"]
        ]
        if not waveforms:
            raise RuntimeError("no speech chunks were available for playback")

        sample_rate = int(batch_result["sample_rate"])
        channel_count = self._waveform_channel_count(waveforms[0])
        total_sample_count = 0
        prepared_waveforms: list[np.ndarray] = []

        for waveform in waveforms:
            current_channel_count = self._waveform_channel_count(waveform)
            if current_channel_count != channel_count:
                raise RuntimeError("streamed speech chunk channel count changed during playback")
            prepared_waveforms.append(waveform)
            total_sample_count += int(waveform.shape[0])

        with self._lock:
            self._speech_phase = "playing"
            self._speech_current_chunk_index = len(prepared_waveforms)
            self._speech_current_chunk_count = len(prepared_waveforms)

        combined_waveform = self._combine_waveforms_for_playback(
            waveforms=prepared_waveforms,
            channel_count=channel_count,
        )
        self._play_waveform_blocking(
            waveform=combined_waveform,
            sample_rate=sample_rate,
            channel_count=channel_count,
        )

        return {
            "played": True,
            "sample_rate": sample_rate,
            "sample_count": total_sample_count,
            "duration_seconds": round(total_sample_count / sample_rate, 3),
            "model_id": batch_result["model_id"],
            "device": batch_result["device"],
            "language": batch_result["language"],
            "player": "sounddevice-play",
        }

    def _speech_worker_loop(self) -> None:
        while True:
            job = self._speech_queue.get()
            if job is None:
                self._speech_queue.task_done()
                return

            job_id = int(job["job_id"])
            chunks = list(job["chunks"])
            voice_description = str(job["voice_description"])
            language = str(job["language"])

            with self._lock:
                self._speech_in_progress = True
                self._speech_phase = "synthesizing"
                self._speech_current_job_id = job_id
                self._speech_current_chunk_index = 0
                self._speech_current_chunk_count = len(chunks)
                self._speech_last_error = None

            try:
                self.play_speech_chunks(
                    chunks=chunks,
                    voice_description=voice_description,
                    language=language,
                )
            except Exception as exc:
                with self._lock:
                    self._speech_jobs_failed += 1
                    self._speech_last_error = str(exc)
                    self._last_error = str(exc)
            else:
                with self._lock:
                    completed_at = _utc_now()
                    self._speech_jobs_completed += 1
                    self._speech_last_completed_at = completed_at
                    self._speech_last_error = None
                    self._touch_locked(completed_at)
            finally:
                with self._lock:
                    self._speech_in_progress = False
                    self._speech_phase = DEFAULT_SPEECH_PHASE
                    self._speech_current_job_id = None
                    self._speech_current_chunk_index = None
                    self._speech_current_chunk_count = None
                self._speech_queue.task_done()

    def _status_payload_locked(self) -> dict[str, Any]:
        return {
            "ready": self._model is not None and self._last_error is None,
            "model_loaded": self._model is not None,
            "device": self._resolved_device,
            "model_id": self.model_id,
            "last_used_at": _timestamp_value(self._last_used_at),
            "last_loaded_at": _timestamp_value(self._last_loaded_at),
            "last_error": self._last_error,
            **self._speech_status_payload_locked(),
        }

    def _speech_status_payload_locked(self) -> dict[str, Any]:
        return {
            "speech_in_progress": self._speech_in_progress,
            "speech_phase": self._speech_phase,
            "speech_current_job_id": self._speech_current_job_id,
            "speech_current_chunk_index": self._speech_current_chunk_index,
            "speech_current_chunk_count": self._speech_current_chunk_count,
            "speech_queue_depth": self._speech_queue.qsize(),
            "speech_queue_maxsize": self.speech_queue_maxsize,
            "speech_jobs_queued": self._speech_jobs_queued,
            "speech_jobs_completed": self._speech_jobs_completed,
            "speech_jobs_failed": self._speech_jobs_failed,
            "speech_last_enqueued_at": _timestamp_value(self._speech_last_enqueued_at),
            "speech_last_completed_at": _timestamp_value(self._speech_last_completed_at),
            "speech_last_error": self._speech_last_error,
        }

    def _touch_locked(self, value: dt.datetime | None = None) -> None:
        self._last_used_at = value or _utc_now()

    def _normalize_chunks(self, chunks: list[str]) -> list[str]:
        normalized_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        if not normalized_chunks:
            raise ValueError("chunks must not be empty")
        return normalized_chunks

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

    def _unload_locked(self) -> None:
        model = self._model
        resolved_device = self._resolved_device
        self._model = None
        self._resolved_device = None
        self._release_model_resources(model, resolved_device=resolved_device)

    def _release_model_resources(self, model: Any | None, *, resolved_device: str | None) -> None:
        del model

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

    def _prepare_waveform_for_output_stream(self, waveform: Any) -> np.ndarray:
        waveform_array = np.asarray(waveform, dtype=np.float32)
        if waveform_array.ndim == 0:
            raise ValueError("waveform must contain at least one frame")
        if waveform_array.ndim == 1:
            return np.ascontiguousarray(waveform_array)
        if waveform_array.ndim == 2:
            return np.ascontiguousarray(waveform_array)
        raise ValueError("waveform must be mono or multi-channel frame data")

    def _waveform_channel_count(self, waveform: np.ndarray) -> int:
        if waveform.ndim == 1:
            return 1
        if waveform.ndim == 2:
            return int(waveform.shape[1])
        raise ValueError("waveform must be mono or multi-channel frame data")

    def _combine_waveforms_for_playback(
        self,
        *,
        waveforms: list[np.ndarray],
        channel_count: int,
    ) -> np.ndarray:
        if not waveforms:
            raise ValueError("waveforms must not be empty")
        if channel_count == 1:
            return np.ascontiguousarray(np.concatenate(waveforms, axis=0), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(waveforms, axis=0), dtype=np.float32)

    def _play_waveform_blocking(
        self,
        *,
        waveform: np.ndarray,
        sample_rate: int,
        channel_count: int,
    ) -> None:
        if channel_count == 1 and waveform.ndim != 1:
            raise ValueError("mono playback requires a one-dimensional waveform")
        if channel_count > 1 and waveform.ndim != 2:
            raise ValueError("multi-channel playback requires two-dimensional waveform data")
        sd.play(waveform, samplerate=sample_rate, blocking=True)
