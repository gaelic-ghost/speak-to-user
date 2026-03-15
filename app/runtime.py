from __future__ import annotations

import builtins
from collections import deque
import contextlib
import datetime as dt
import gc
import importlib.util
import json
import os
import queue
import shutil
import sys
import threading
from typing import Any, cast

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]

# MARK: Constants

DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_PLAYBACK_PREROLL_SECONDS = 3.0
DEFAULT_PLAYBACK_PREROLL_CHUNKS = 2
DEFAULT_PLAYBACK_WAVEFORM_QUEUE_MAXSIZE = 16
DEFAULT_OUTPUT_STREAM_LATENCY = "high"
DEFAULT_SPEECH_QUEUE_MAXSIZE = 32
DEFAULT_SPEECH_PHASE = "idle"
DEFAULT_LOG_LEVEL = "info"
DEFAULT_RECENT_EVENT_LIMIT = 100
LOG_LEVEL_PRIORITY = {"minimal": 0, "info": 1, "debug": 2}
LANGUAGE_ALIASES = {
    "auto": "Auto",
    "en": "English",
    "english": "English",
    "fr": "French",
    "french": "French",
    "de": "German",
    "german": "German",
    "it": "Italian",
    "italian": "Italian",
    "ja": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "ru": "Russian",
    "russian": "Russian",
    "es": "Spanish",
    "spanish": "Spanish",
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


def _normalize_torch_dtype(raw: str | None) -> str | None:
    value = (raw or "").strip().lower()
    if not value:
        return None
    if value not in {"float16", "bfloat16", "float32"}:
        raise ValueError(
            "SPEAK_TO_USER_TORCH_DTYPE must be one of: float16, bfloat16, float32"
        )
    return value


def _normalize_positive_float(raw: str | None, *, env_var: str, default: float) -> float:
    if raw is None:
        return default

    value = raw.strip()
    if not value:
        raise ValueError(f"{env_var} must not be empty")

    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be a number") from exc

    if parsed <= 0:
        raise ValueError(f"{env_var} must be greater than zero")
    return parsed


def _normalize_positive_int(raw: str | None, *, env_var: str, default: int) -> int:
    if raw is None:
        return default

    value = raw.strip()
    if not value:
        raise ValueError(f"{env_var} must not be empty")

    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be an integer") from exc

    if parsed <= 0:
        raise ValueError(f"{env_var} must be greater than zero")
    return parsed


def _normalize_output_stream_latency(raw: str | None) -> str | float:
    value = (raw or DEFAULT_OUTPUT_STREAM_LATENCY).strip().lower()
    if value in {"low", "high"}:
        return value

    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            "SPEAK_TO_USER_OUTPUT_STREAM_LATENCY must be low, high, or a number"
        ) from exc

    if parsed <= 0:
        raise ValueError("SPEAK_TO_USER_OUTPUT_STREAM_LATENCY must be greater than zero")
    return parsed


def _normalize_log_level(raw: str | None) -> str:
    value = (raw or DEFAULT_LOG_LEVEL).strip().lower()
    if value not in LOG_LEVEL_PRIORITY:
        raise ValueError("SPEAK_TO_USER_LOG_LEVEL must be one of: minimal, info, debug")
    return value


def _normalize_language(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("language must not be empty")

    lowered = normalized.lower()
    alias = LANGUAGE_ALIASES.get(lowered)
    if alias is not None:
        return alias

    # Accept common locale variants like en-US and pt_BR by folding to the base language.
    for separator in ("-", "_"):
        if separator in lowered:
            base_language = lowered.split(separator, 1)[0]
            alias = LANGUAGE_ALIASES.get(base_language)
            if alias is not None:
                return alias

    return normalized


def _runtime_dependency_status() -> dict[str, bool]:
    return {
        "sox_available": shutil.which("sox") is not None,
        "qwen_tts_available": importlib.util.find_spec("qwen_tts") is not None,
        "torch_available": importlib.util.find_spec("torch") is not None,
        "torchaudio_available": importlib.util.find_spec("torchaudio") is not None,
    }


def _runtime_dependencies_ready() -> bool:
    status = _runtime_dependency_status()
    return bool(status["sox_available"] and status["qwen_tts_available"])


def _meta_tensor_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "meta tensor" in message and "tensor.item()" in message


def _duration_ms(start: dt.datetime | None, end: dt.datetime | None) -> int | None:
    if start is None or end is None:
        return None
    return max(0, int((end - start).total_seconds() * 1000))


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
        torch_dtype_name: str | None = None,
        speech_queue_maxsize: int = DEFAULT_SPEECH_QUEUE_MAXSIZE,
        playback_preroll_seconds: float = DEFAULT_PLAYBACK_PREROLL_SECONDS,
        playback_preroll_chunks: int = DEFAULT_PLAYBACK_PREROLL_CHUNKS,
        playback_waveform_queue_maxsize: int = DEFAULT_PLAYBACK_WAVEFORM_QUEUE_MAXSIZE,
        output_stream_latency: str | float = DEFAULT_OUTPUT_STREAM_LATENCY,
        log_level: str = DEFAULT_LOG_LEVEL,
    ) -> None:
        self.model_id = model_id
        self.device_preference = device_preference
        self.torch_dtype_name = torch_dtype_name
        self.speech_queue_maxsize = speech_queue_maxsize
        self.playback_preroll_seconds = playback_preroll_seconds
        self.playback_preroll_chunks = playback_preroll_chunks
        self.playback_waveform_queue_maxsize = playback_waveform_queue_maxsize
        self.output_stream_latency = output_stream_latency
        self.log_level = log_level

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
        self._cpu_fallback_active = False

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
        self._current_job_started_at: dt.datetime | None = None
        self._current_chunk_started_at: dt.datetime | None = None
        self._current_phase_started_at: dt.datetime | None = None
        self._speech_last_event_at: dt.datetime | None = None
        self._speech_last_event: str | None = None
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=DEFAULT_RECENT_EVENT_LIMIT)

    @classmethod
    def from_env(cls) -> TTSRuntime:
        return cls(
            model_id=os.getenv("SPEAK_TO_USER_MODEL_ID", DEFAULT_MODEL_ID),
            device_preference=_normalize_device(os.getenv("SPEAK_TO_USER_DEVICE")),
            torch_dtype_name=_normalize_torch_dtype(os.getenv("SPEAK_TO_USER_TORCH_DTYPE")),
            playback_preroll_seconds=_normalize_positive_float(
                os.getenv("SPEAK_TO_USER_PLAYBACK_PREROLL_SECONDS"),
                env_var="SPEAK_TO_USER_PLAYBACK_PREROLL_SECONDS",
                default=DEFAULT_PLAYBACK_PREROLL_SECONDS,
            ),
            playback_preroll_chunks=_normalize_positive_int(
                os.getenv("SPEAK_TO_USER_PLAYBACK_PREROLL_CHUNKS"),
                env_var="SPEAK_TO_USER_PLAYBACK_PREROLL_CHUNKS",
                default=DEFAULT_PLAYBACK_PREROLL_CHUNKS,
            ),
            playback_waveform_queue_maxsize=_normalize_positive_int(
                os.getenv("SPEAK_TO_USER_PLAYBACK_WAVEFORM_QUEUE_MAXSIZE"),
                env_var="SPEAK_TO_USER_PLAYBACK_WAVEFORM_QUEUE_MAXSIZE",
                default=DEFAULT_PLAYBACK_WAVEFORM_QUEUE_MAXSIZE,
            ),
            output_stream_latency=_normalize_output_stream_latency(
                os.getenv("SPEAK_TO_USER_OUTPUT_STREAM_LATENCY")
            ),
            log_level=_normalize_log_level(os.getenv("SPEAK_TO_USER_LOG_LEVEL")),
        )

    def preload(self) -> dict[str, Any]:
        self._emit_runtime_event("runtime_preload_started", level="minimal")
        self.start_speech_worker()
        result = self.load_model()
        self._emit_runtime_event(
            "runtime_preload_completed",
            level="minimal",
            model_loaded=bool(result["loaded"]),
        )
        return result

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
            self._emit_runtime_event("speech_worker_started", level="minimal")
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
        self._emit_runtime_event("runtime_shutdown_completed", level="minimal")

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_payload_locked()

    def load_model(self) -> dict[str, Any]:
        if not _runtime_dependencies_ready():
            dependency_status = _runtime_dependency_status()
            missing: list[str] = []
            if not dependency_status["sox_available"]:
                missing.append("sox")
            if not dependency_status["qwen_tts_available"]:
                missing.append("qwen_tts")
            message = f"runtime dependencies are unavailable: {', '.join(missing)}"
            with self._lock:
                self._last_error = message
            self._emit_runtime_event("model_load_failed", level="minimal", error=message)
            raise RuntimeError(message)

        with self._lock:
            if self._shutdown_requested.is_set():
                raise RuntimeError("runtime is shutting down")
            if self._model is not None:
                self._touch_locked()
                self._emit_runtime_event(
                    "model_load_skipped_already_loaded",
                    level="debug",
                    device=self._resolved_device,
                )
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
            self._emit_runtime_event("model_load_failed", level="minimal", error=str(exc))
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
            self._emit_runtime_event(
                "model_load_completed",
                level="minimal",
                device=self._resolved_device,
                cpu_fallback_active=self._cpu_fallback_active,
            )
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

        self.start_speech_worker()

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

            try:
                self._speech_queue.put_nowait(job)
            except queue.Full as exc:
                self._speech_last_error = "speech queue is full"
                self._last_error = self._speech_last_error
                raise RuntimeError("speech queue is full") from exc

            enqueued_at = _utc_now()
            self._speech_jobs_queued = job_id
            self._speech_last_enqueued_at = enqueued_at
            self._speech_last_error = None
            self._emit_runtime_event(
                "speech_job_queued",
                level="minimal",
                job_id=job_id,
                chunk_count=len(normalized_chunks),
                chunk_char_count=sum(len(chunk) for chunk in normalized_chunks),
                language=normalized_language,
                queue_depth=self._speech_queue.qsize(),
            )
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
        total_sample_count = 0
        buffered_sample_count = 0
        buffered_waveforms: list[np.ndarray] = []
        buffered_chunk_count = 0
        sample_rate: int | None = None
        channel_count: int | None = None
        model_id: str | None = None
        device: str | None = None
        preroll_sample_target: int | None = None
        producer_error: list[BaseException] = []
        synthesis_done = threading.Event()
        waveform_queue: queue.Queue[dict[str, Any] | object] = queue.Queue(
            maxsize=self.playback_waveform_queue_maxsize
        )
        queue_sentinel = object()

        stream: sd.OutputStream | None = None

        def synthesize_into_queue() -> None:
            try:
                for chunk_index, text_chunk in enumerate(normalized_chunks, start=1):
                    synth_started_at = _utc_now()
                    self._emit_runtime_event(
                        "speech_chunk_synthesis_started",
                        level="info",
                        job_id=self._speech_current_job_id,
                        chunk_index=chunk_index,
                        chunk_count=len(normalized_chunks),
                        chunk_char_count=len(text_chunk),
                    )
                    chunk_result = self._synthesize_audio_chunk(
                        text=text_chunk,
                        voice_description=normalized_description,
                        language=normalized_language,
                    )
                    waveform = self._prepare_waveform_for_output_stream(chunk_result["waveform"])
                    self._emit_runtime_event(
                        "speech_chunk_synthesis_completed",
                        level="info",
                        job_id=self._speech_current_job_id,
                        chunk_index=chunk_index,
                        chunk_count=len(normalized_chunks),
                        synth_duration_ms=_duration_ms(synth_started_at, _utc_now()),
                        sample_count=int(waveform.shape[0]),
                    )
                    waveform_queue.put(
                        {
                            "waveform": waveform,
                            "sample_rate": int(chunk_result["sample_rate"]),
                            "model_id": str(chunk_result["model_id"]),
                            "device": str(chunk_result["device"]),
                            "language": str(chunk_result["language"]),
                        }
                    )
            except BaseException as exc:
                producer_error.append(exc)
                self._emit_runtime_event(
                    "speech_chunk_producer_failed",
                    level="minimal",
                    job_id=self._speech_current_job_id,
                    error=str(exc),
                )
            finally:
                synthesis_done.set()
                waveform_queue.put(queue_sentinel)

        producer_thread = threading.Thread(
            target=synthesize_into_queue,
            name="speak-to-user-waveform-producer",
            daemon=True,
        )
        producer_thread.start()

        try:
            next_chunk_index = 1
            playback_started_at: dt.datetime | None = None
            while True:
                with self._lock:
                    self._set_speech_phase_locked("synthesizing")

                queue_item = waveform_queue.get()
                if queue_item is queue_sentinel:
                    break

                chunk_result = cast(dict[str, Any], queue_item)
                waveform = chunk_result["waveform"]

                current_sample_rate = int(chunk_result["sample_rate"])
                current_model_id = str(chunk_result["model_id"])
                current_device = str(chunk_result["device"])
                current_language = str(chunk_result["language"])

                if sample_rate is None:
                    sample_rate = current_sample_rate
                    channel_count = self._waveform_channel_count(waveform)
                    preroll_sample_target = max(
                        1,
                        int(sample_rate * self.playback_preroll_seconds),
                    )
                    model_id = current_model_id
                    device = current_device
                else:
                    if current_sample_rate != sample_rate:
                        raise RuntimeError(
                            "streamed speech chunk sample rate changed during playback"
                        )
                    current_channel_count = self._waveform_channel_count(waveform)
                    if channel_count is None or current_channel_count != channel_count:
                        raise RuntimeError(
                            "streamed speech chunk channel count changed during playback"
                        )
                    if model_id != current_model_id:
                        raise RuntimeError("streamed speech chunk model id changed during playback")
                    if device != current_device:
                        raise RuntimeError("streamed speech chunk device changed during playback")
                    if current_language != normalized_language:
                        raise RuntimeError(
                            "streamed speech chunk language changed during playback"
                        )

                if stream is None:
                    buffered_waveforms.append(waveform)
                    buffered_sample_count += int(waveform.shape[0])
                    buffered_chunk_count += 1
                    assert sample_rate is not None
                    assert channel_count is not None
                    assert preroll_sample_target is not None

                    have_enough_preroll_audio = buffered_sample_count >= preroll_sample_target
                    have_enough_preroll_chunks = (
                        buffered_chunk_count >= min(
                            self.playback_preroll_chunks,
                            len(normalized_chunks),
                        )
                    )
                    if (
                        not synthesis_done.is_set()
                        and not have_enough_preroll_audio
                        and not have_enough_preroll_chunks
                    ):
                        self._emit_runtime_event(
                            "speech_preroll_waiting",
                            level="debug",
                            job_id=self._speech_current_job_id,
                            buffered_chunk_count=buffered_chunk_count,
                            buffered_sample_count=buffered_sample_count,
                            preroll_sample_target=preroll_sample_target,
                        )
                        continue

                    with self._lock:
                        self._set_speech_phase_locked("opening_output")
                    self._emit_runtime_event(
                        "speech_output_stream_opening",
                        level="info",
                        job_id=self._speech_current_job_id,
                        sample_rate=sample_rate,
                        channel_count=channel_count,
                        buffered_chunk_count=buffered_chunk_count,
                        buffered_sample_count=buffered_sample_count,
                    )
                    stream = self._open_output_stream(
                        sample_rate=sample_rate,
                        channel_count=channel_count,
                    )
                    playback_started_at = _utc_now()
                    self._emit_runtime_event(
                        "speech_output_stream_opened",
                        level="info",
                        job_id=self._speech_current_job_id,
                        sample_rate=sample_rate,
                        channel_count=channel_count,
                    )

                    for buffered_waveform in buffered_waveforms:
                        with self._lock:
                            self._set_speech_phase_locked("playing")
                            self._speech_current_chunk_index = next_chunk_index
                            self._speech_current_chunk_count = len(normalized_chunks)
                            self._current_chunk_started_at = _utc_now()
                        self._emit_runtime_event(
                            "speech_chunk_playback_started",
                            level="info",
                            job_id=self._speech_current_job_id,
                            chunk_index=next_chunk_index,
                            chunk_count=len(normalized_chunks),
                            sample_count=int(buffered_waveform.shape[0]),
                        )
                        self._write_output_stream_chunk(stream, buffered_waveform)
                        total_sample_count += int(buffered_waveform.shape[0])
                        self._emit_runtime_event(
                            "speech_chunk_playback_completed",
                            level="info",
                            job_id=self._speech_current_job_id,
                            chunk_index=next_chunk_index,
                            chunk_count=len(normalized_chunks),
                            playback_duration_ms=_duration_ms(
                                self._current_chunk_started_at or playback_started_at,
                                _utc_now(),
                            ),
                        )
                        next_chunk_index += 1
                    buffered_waveforms.clear()
                    continue

                assert stream is not None
                with self._lock:
                    self._set_speech_phase_locked("playing")
                    self._speech_current_chunk_index = next_chunk_index
                    self._speech_current_chunk_count = len(normalized_chunks)
                    self._current_chunk_started_at = _utc_now()
                self._emit_runtime_event(
                    "speech_chunk_playback_started",
                    level="info",
                    job_id=self._speech_current_job_id,
                    chunk_index=next_chunk_index,
                    chunk_count=len(normalized_chunks),
                    sample_count=int(waveform.shape[0]),
                )
                self._write_output_stream_chunk(stream, waveform)
                total_sample_count += int(waveform.shape[0])
                self._emit_runtime_event(
                    "speech_chunk_playback_completed",
                    level="info",
                    job_id=self._speech_current_job_id,
                    chunk_index=next_chunk_index,
                    chunk_count=len(normalized_chunks),
                    playback_duration_ms=_duration_ms(
                        self._current_chunk_started_at or playback_started_at,
                        _utc_now(),
                    ),
                )
                next_chunk_index += 1

            if producer_error:
                raise RuntimeError(str(producer_error[0])) from producer_error[0]

            if stream is None:
                raise RuntimeError("no speech chunks were available for playback")

            assert sample_rate is not None
            assert model_id is not None
            assert device is not None
            return {
                "played": True,
                "sample_rate": sample_rate,
                "sample_count": total_sample_count,
                "duration_seconds": round(total_sample_count / sample_rate, 3),
                "model_id": model_id,
                "device": device,
                "language": normalized_language,
                "player": "sounddevice-stream",
            }
        finally:
            producer_thread.join(timeout=1)
            if stream is not None:
                self._emit_runtime_event(
                    "speech_output_stream_closing",
                    level="debug",
                    job_id=self._speech_current_job_id,
                )
                stream.stop()
                stream.close()
                self._emit_runtime_event(
                    "speech_output_stream_closed",
                    level="debug",
                    job_id=self._speech_current_job_id,
                )

    def _speech_worker_loop(self) -> None:
        while True:
            job = self._speech_queue.get()
            if job is None:
                self._speech_queue.task_done()
                self._emit_runtime_event("speech_worker_stopped", level="minimal")
                return

            job_id = int(job["job_id"])
            chunks = list(job["chunks"])
            voice_description = str(job["voice_description"])
            language = str(job["language"])

            with self._lock:
                self._speech_in_progress = True
                self._current_job_started_at = _utc_now()
                self._set_speech_phase_locked("synthesizing")
                self._speech_current_job_id = job_id
                self._speech_current_chunk_index = 0
                self._speech_current_chunk_count = len(chunks)
                self._speech_last_error = None
            self._emit_runtime_event(
                "speech_job_started",
                level="minimal",
                job_id=job_id,
                chunk_count=len(chunks),
                queue_depth=self._speech_queue.qsize(),
                enqueue_to_start_ms=_duration_ms(
                    self._speech_last_enqueued_at,
                    self._current_job_started_at,
                ),
            )

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
                self._emit_runtime_event(
                    "speech_job_failed",
                    level="minimal",
                    job_id=job_id,
                    phase=self._speech_phase,
                    error=str(exc),
                    total_job_duration_ms=_duration_ms(self._current_job_started_at, _utc_now()),
                )
            else:
                with self._lock:
                    completed_at = _utc_now()
                    self._speech_jobs_completed += 1
                    self._speech_last_completed_at = completed_at
                    self._speech_last_error = None
                    self._touch_locked(completed_at)
                self._emit_runtime_event(
                    "speech_job_completed",
                    level="minimal",
                    job_id=job_id,
                    total_job_duration_ms=_duration_ms(self._current_job_started_at, completed_at),
                )
            finally:
                with self._lock:
                    self._speech_in_progress = False
                    self._set_speech_phase_locked(DEFAULT_SPEECH_PHASE)
                    self._speech_current_job_id = None
                    self._speech_current_chunk_index = None
                    self._speech_current_chunk_count = None
                    self._current_job_started_at = None
                    self._current_chunk_started_at = None
                self._speech_queue.task_done()

    def _status_payload_locked(self) -> dict[str, Any]:
        dependency_status = _runtime_dependency_status()
        return {
            "ready": (
                _runtime_dependencies_ready()
                and self._model is not None
                and self._last_error is None
            ),
            "model_loaded": self._model is not None,
            "device_preference": self.device_preference,
            "device": self._resolved_device,
            "model_id": self.model_id,
            "torch_dtype": self.torch_dtype_name,
            "cpu_fallback_active": self._cpu_fallback_active,
            "playback_preroll_seconds": self.playback_preroll_seconds,
            "playback_preroll_chunks": self.playback_preroll_chunks,
            "playback_waveform_queue_maxsize": self.playback_waveform_queue_maxsize,
            "output_stream_latency": self.output_stream_latency,
            "log_level": self.log_level,
            "last_used_at": _timestamp_value(self._last_used_at),
            "last_loaded_at": _timestamp_value(self._last_loaded_at),
            "last_error": self._last_error,
            "current_job_started_at": _timestamp_value(self._current_job_started_at),
            "current_chunk_started_at": _timestamp_value(self._current_chunk_started_at),
            "current_phase_started_at": _timestamp_value(self._current_phase_started_at),
            "speech_last_event_at": _timestamp_value(self._speech_last_event_at),
            "speech_last_event": self._speech_last_event,
            "recent_events": list(self._recent_events),
            **dependency_status,
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
            "speech_last_event_at": _timestamp_value(self._speech_last_event_at),
            "speech_last_event": self._speech_last_event,
        }

    def _touch_locked(self, value: dt.datetime | None = None) -> None:
        self._last_used_at = value or _utc_now()

    def _set_speech_phase_locked(self, phase: str) -> None:
        if self._speech_phase != phase or self._current_phase_started_at is None:
            self._current_phase_started_at = _utc_now()
        self._speech_phase = phase

    def _emit_runtime_event(self, event: str, *, level: str, **details: Any) -> None:
        if LOG_LEVEL_PRIORITY[level] > LOG_LEVEL_PRIORITY[self.log_level]:
            return

        timestamp = _utc_now()
        payload: dict[str, Any] = {"timestamp": timestamp.isoformat(), "event": event}
        payload.update(details)

        with self._lock:
            self._speech_last_event_at = timestamp
            self._speech_last_event = event
            self._recent_events.append(dict(payload))

        _ORIGINAL_PRINT(json.dumps(payload, sort_keys=True), file=sys.stderr)

    def _normalize_chunks(self, chunks: list[str]) -> list[str]:
        normalized_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        if not normalized_chunks:
            raise ValueError("chunks must not be empty")
        return normalized_chunks

    def _resolve_torch_dtype(self, torch_module: Any) -> Any | None:
        if self.torch_dtype_name is None:
            return None

        if self.torch_dtype_name == "float16":
            return torch_module.float16
        if self.torch_dtype_name == "bfloat16":
            return torch_module.bfloat16
        if self.torch_dtype_name == "float32":
            return torch_module.float32
        raise RuntimeError("unsupported torch dtype configuration")

    def _load_model_kwargs(self, torch_module: Any) -> dict[str, Any]:
        load_kwargs: dict[str, Any] = {}
        torch_dtype = self._resolve_torch_dtype(torch_module)

        if self._cpu_fallback_active and self.device_preference == "auto":
            load_kwargs["device_map"] = "cpu"
            load_kwargs["torch_dtype"] = torch_dtype or torch_module.float32
            return load_kwargs

        load_kwargs["device_map"] = self._resolve_device(torch_module)
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        return load_kwargs

    def _reload_model_with_cpu_fallback(self) -> None:
        with self._lock:
            if self.device_preference != "auto":
                raise RuntimeError(
                    "CPU fallback is only supported when SPEAK_TO_USER_DEVICE=auto"
                )
            previous_model = self._model
            previous_device = self._resolved_device
            self._model = None
            self._resolved_device = None
            self._cpu_fallback_active = True
        self._emit_runtime_event("model_cpu_fallback_activated", level="minimal")

        self._release_model_resources(previous_model, resolved_device=previous_device)
        gc.collect()

        loaded_model = self._load_model_impl()

        with self._lock:
            if self._shutdown_requested.is_set():
                self._release_model_resources(loaded_model, resolved_device=self._resolved_device)
                self._resolved_device = None
                raise RuntimeError("model reload aborted during shutdown")

            self._model = loaded_model
            self._last_error = None
            self._last_loaded_at = _utc_now()
            self._touch_locked(self._last_loaded_at)

    def _synthesize_audio_chunk(
        self,
        *,
        text: str,
        voice_description: str,
        language: str,
    ) -> dict[str, Any]:
        with self._lock:
            model_missing = self._model is None

        if model_missing:
            self.load_model()

        with self._lock:
            model = self._model

        assert model is not None

        def generate(active_model: Any) -> tuple[list[Any], int]:
            wavs, sample_rate = active_model.generate_voice_design(
                text=text,
                language=language,
                instruct=voice_description,
            )
            return cast(list[Any], wavs), int(sample_rate)

        try:
            wavs, sample_rate = generate(model)
        except Exception as exc:
            if _meta_tensor_runtime_error(exc):
                self._emit_runtime_event(
                    "speech_chunk_synthesis_retrying_with_cpu_fallback",
                    level="minimal",
                    job_id=self._speech_current_job_id,
                    error=str(exc),
                )
                try:
                    self._reload_model_with_cpu_fallback()
                    with self._lock:
                        reloaded_model = self._model
                    if reloaded_model is None:
                        raise RuntimeError("CPU fallback reload completed without a loaded model")
                    wavs, sample_rate = generate(reloaded_model)
                except Exception as retry_exc:
                    with self._lock:
                        self._last_error = f"synthesis failed after CPU fallback retry: {retry_exc}"
                    self._emit_runtime_event(
                        "speech_chunk_synthesis_retry_failed",
                        level="minimal",
                        job_id=self._speech_current_job_id,
                        error=str(retry_exc),
                    )
                    raise RuntimeError(
                        f"synthesis failed after CPU fallback retry: {retry_exc}"
                    ) from retry_exc
            else:
                with self._lock:
                    self._last_error = str(exc)
                self._emit_runtime_event(
                    "speech_chunk_synthesis_failed",
                    level="minimal",
                    job_id=self._speech_current_job_id,
                    error=str(exc),
                )
                raise

        waveforms = [self._coerce_waveform(waveform) for waveform in wavs]
        if not waveforms:
            raise RuntimeError("model returned no waveform for synthesized chunk")

        with self._lock:
            now = _utc_now()
            self._last_error = None
            self._touch_locked(now)
            return {
                "waveform": waveforms[0],
                "sample_rate": sample_rate,
                "model_id": self.model_id,
                "device": self._resolved_device,
                "language": language,
            }

    def _load_model_impl(self) -> Any:
        with _suppress_default_stdout_prints_for_current_thread():
            import torch
            from qwen_tts import Qwen3TTSModel  # type: ignore[import-untyped]

            load_kwargs = self._load_model_kwargs(torch)
            resolved_device = str(load_kwargs["device_map"])
            self._resolved_device = resolved_device
            return Qwen3TTSModel.from_pretrained(self.model_id, **load_kwargs)

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

    def _open_output_stream(
        self,
        *,
        sample_rate: int,
        channel_count: int,
    ) -> sd.OutputStream:
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=channel_count,
            dtype="float32",
            latency=self.output_stream_latency,
        )
        stream.start()
        return stream

    def _write_output_stream_chunk(
        self,
        stream: sd.OutputStream,
        waveform: np.ndarray,
    ) -> None:
        underflowed = stream.write(waveform)
        if underflowed:
            raise RuntimeError("audio output underflowed during streamed playback")
