from __future__ import annotations

import builtins
from collections import deque
import contextlib
import datetime as dt
import gc
import importlib.util
import json
import os
from pathlib import Path
import queue
import shutil
import sys
import threading
from typing import Any, cast

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]
from pydantic import BaseModel

from fastmcp.server.server import StateValue

# MARK: Constants

VOICE_DESIGN_MODEL_KIND = "voice_design"
CLONE_MODEL_KIND = "clone"
DEFAULT_VOICE_DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_CLONE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_ENABLE_VOICE_DESIGN_MODEL = True
DEFAULT_ENABLE_CLONE_MODEL = True
DEFAULT_PLAYBACK_PREROLL_SECONDS = 3.0
DEFAULT_PLAYBACK_PREROLL_CHUNKS = 2
DEFAULT_PLAYBACK_WAVEFORM_QUEUE_MAXSIZE = 16
DEFAULT_PLAYBACK_UNDERFLOW_RETRIES = 2
DEFAULT_OUTPUT_STREAM_LATENCY = "high"
DEFAULT_SPEECH_QUEUE_MAXSIZE = 32
DEFAULT_SPEECH_PHASE = "idle"
DEFAULT_LOG_LEVEL = "info"
DEFAULT_RECENT_EVENT_LIMIT = 100
SPEECH_PROFILE_COLLECTION = "speak_to_user_profiles"
SPEECH_PROFILE_INDEX_KEY = "index"
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


class SerializedTensor(BaseModel):
    dtype: str
    data: list[Any]


class StoredSpeechProfilePromptItem(BaseModel):
    ref_code: SerializedTensor | None = None
    ref_spk_embedding: SerializedTensor
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str | None = None


class StoredSpeechProfile(BaseModel):
    name: str
    clone_model_id: str
    clone_mode: str
    created_at: str
    updated_at: str
    prompt_items: list[StoredSpeechProfilePromptItem]


class StoredSpeechProfileIndex(BaseModel):
    names: list[str]


# MARK: Module Helpers

def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _timestamp_value(value: dt.datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _normalize_enabled_flag(
    raw: str | None,
    *,
    env_var: str,
    default: bool,
) -> bool:
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{env_var} must be one of: true, false, 1, 0, yes, no, on, off")


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


def _normalize_nonnegative_int(raw: str | None, *, env_var: str, default: int) -> int:
    if raw is None:
        return default

    value = raw.strip()
    if not value:
        raise ValueError(f"{env_var} must not be empty")

    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be an integer") from exc

    if parsed < 0:
        raise ValueError(f"{env_var} must be zero or greater")
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
        "soundfile_available": importlib.util.find_spec("soundfile") is not None,
    }


def _runtime_dependencies_ready() -> bool:
    status = _runtime_dependency_status()
    return bool(
        status["sox_available"]
        and status["qwen_tts_available"]
        and status["soundfile_available"]
    )


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
        voice_design_model_id: str = DEFAULT_VOICE_DESIGN_MODEL_ID,
        clone_model_id: str = DEFAULT_CLONE_MODEL_ID,
        enable_voice_design_model: bool = DEFAULT_ENABLE_VOICE_DESIGN_MODEL,
        enable_clone_model: bool = DEFAULT_ENABLE_CLONE_MODEL,
        device_preference: str,
        torch_dtype_name: str | None = None,
        speech_queue_maxsize: int = DEFAULT_SPEECH_QUEUE_MAXSIZE,
        playback_preroll_seconds: float = DEFAULT_PLAYBACK_PREROLL_SECONDS,
        playback_preroll_chunks: int = DEFAULT_PLAYBACK_PREROLL_CHUNKS,
        playback_waveform_queue_maxsize: int = DEFAULT_PLAYBACK_WAVEFORM_QUEUE_MAXSIZE,
        playback_underflow_retries: int = DEFAULT_PLAYBACK_UNDERFLOW_RETRIES,
        output_stream_latency: str | float = DEFAULT_OUTPUT_STREAM_LATENCY,
        log_level: str = DEFAULT_LOG_LEVEL,
    ) -> None:
        self.voice_design_model_id = voice_design_model_id
        self.clone_model_id = clone_model_id
        self.enable_voice_design_model = enable_voice_design_model
        self.enable_clone_model = enable_clone_model
        self.device_preference = device_preference
        self.torch_dtype_name = torch_dtype_name
        self.speech_queue_maxsize = speech_queue_maxsize
        self.playback_preroll_seconds = playback_preroll_seconds
        self.playback_preroll_chunks = playback_preroll_chunks
        self.playback_waveform_queue_maxsize = playback_waveform_queue_maxsize
        self.playback_underflow_retries = playback_underflow_retries
        self.output_stream_latency = output_stream_latency
        self.log_level = log_level

        self._lock = threading.RLock()
        self._shutdown_requested = threading.Event()
        self._speech_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(
            maxsize=speech_queue_maxsize
        )
        self._speech_worker_thread: threading.Thread | None = None
        self._speech_worker_started = False

        self._model_slots: dict[str, dict[str, Any]] = {
            VOICE_DESIGN_MODEL_KIND: self._new_model_slot(
                model_id=voice_design_model_id,
                enabled=enable_voice_design_model,
            ),
            CLONE_MODEL_KIND: self._new_model_slot(
                model_id=clone_model_id,
                enabled=enable_clone_model,
            ),
        }
        self._last_error: str | None = None

        self._speech_in_progress = False
        self._speech_phase = DEFAULT_SPEECH_PHASE
        self._speech_current_job_id: int | None = None
        self._speech_current_chunk_index: int | None = None
        self._speech_current_chunk_count: int | None = None
        self._speech_current_mode: str | None = None
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
        self._speech_profile_count: int | None = None
        self._speech_profile_last_error: str | None = None

    @classmethod
    def from_env(cls) -> TTSRuntime:
        return cls(
            voice_design_model_id=os.getenv(
                "SPEAK_TO_USER_VOICE_DESIGN_MODEL_ID",
                DEFAULT_VOICE_DESIGN_MODEL_ID,
            ),
            clone_model_id=os.getenv(
                "SPEAK_TO_USER_CLONE_MODEL_ID",
                DEFAULT_CLONE_MODEL_ID,
            ),
            enable_voice_design_model=_normalize_enabled_flag(
                os.getenv("SPEAK_TO_USER_ENABLE_VOICE_DESIGN_MODEL"),
                env_var="SPEAK_TO_USER_ENABLE_VOICE_DESIGN_MODEL",
                default=DEFAULT_ENABLE_VOICE_DESIGN_MODEL,
            ),
            enable_clone_model=_normalize_enabled_flag(
                os.getenv("SPEAK_TO_USER_ENABLE_CLONE_MODEL"),
                env_var="SPEAK_TO_USER_ENABLE_CLONE_MODEL",
                default=DEFAULT_ENABLE_CLONE_MODEL,
            ),
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
            playback_underflow_retries=_normalize_nonnegative_int(
                os.getenv("SPEAK_TO_USER_PLAYBACK_UNDERFLOW_RETRIES"),
                env_var="SPEAK_TO_USER_PLAYBACK_UNDERFLOW_RETRIES",
                default=DEFAULT_PLAYBACK_UNDERFLOW_RETRIES,
            ),
            output_stream_latency=_normalize_output_stream_latency(
                os.getenv("SPEAK_TO_USER_OUTPUT_STREAM_LATENCY")
            ),
            log_level=_normalize_log_level(os.getenv("SPEAK_TO_USER_LOG_LEVEL")),
        )

    def preload(self) -> dict[str, Any]:
        self._emit_runtime_event("runtime_preload_started", level="minimal")
        self.start_speech_worker()

        enabled_model_kinds = self._enabled_model_kinds()
        for model_kind in enabled_model_kinds:
            self.load_model(model_kind=model_kind)

        result = {
            "result": "success",
            "loaded": self._all_enabled_models_loaded_locked(),
            "loaded_model_kinds": enabled_model_kinds,
            **self.status(),
        }
        self._emit_runtime_event(
            "runtime_preload_completed",
            level="minimal",
            enabled_model_kinds=enabled_model_kinds,
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
            for model_kind in list(self._model_slots):
                self._unload_model_locked(model_kind)
        self._emit_runtime_event("runtime_shutdown_completed", level="minimal")

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_payload_locked()

    def load_model(self, *, model_kind: str = VOICE_DESIGN_MODEL_KIND) -> dict[str, Any]:
        if not _runtime_dependencies_ready():
            dependency_status = _runtime_dependency_status()
            missing = [
                name.removesuffix("_available")
                for name, available in dependency_status.items()
                if not available and name
            ]
            message = f"runtime dependencies are unavailable: {', '.join(sorted(missing))}"
            with self._lock:
                self._last_error = message
                self._model_state(model_kind)["last_error"] = message
            self._emit_runtime_event(
                "model_load_failed",
                level="minimal",
                model_kind=model_kind,
                error=message,
            )
            raise RuntimeError(message)

        with self._lock:
            slot = self._model_state(model_kind)
            if self._shutdown_requested.is_set():
                raise RuntimeError("runtime is shutting down")
            if not slot["enabled"]:
                raise RuntimeError(f"{model_kind} model is disabled")
            if slot["model"] is not None:
                self._touch_model_locked(model_kind)
                self._emit_runtime_event(
                    "model_load_skipped_already_loaded",
                    level="debug",
                    model_kind=model_kind,
                    device=slot["resolved_device"],
                )
                return {
                    "result": "success",
                    "loaded": True,
                    "info": f"{model_kind} model already loaded",
                    **self._status_payload_locked(),
                }

        try:
            loaded_model = self._load_model_impl(model_kind)
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
                self._model_state(model_kind)["last_error"] = str(exc)
            self._emit_runtime_event(
                "model_load_failed",
                level="minimal",
                model_kind=model_kind,
                error=str(exc),
            )
            raise

        with self._lock:
            slot = self._model_state(model_kind)
            if self._shutdown_requested.is_set():
                self._release_model_resources(
                    loaded_model,
                    resolved_device=slot["resolved_device"],
                )
                slot["resolved_device"] = None
                return {
                    "result": "success",
                    "loaded": False,
                    "info": f"{model_kind} model load aborted during shutdown",
                    **self._status_payload_locked(),
                }

            slot["model"] = loaded_model
            slot["last_error"] = None
            slot["last_loaded_at"] = _utc_now()
            self._touch_model_locked(model_kind, slot["last_loaded_at"])
            self._last_error = None
            self._emit_runtime_event(
                "model_load_completed",
                level="minimal",
                model_kind=model_kind,
                device=slot["resolved_device"],
                cpu_fallback_active=slot["cpu_fallback_active"],
            )
            return {
                "result": "success",
                "loaded": True,
                "info": f"{model_kind} model loaded",
                **self._status_payload_locked(),
            }

    def speak_text(
        self,
        *,
        chunks: list[str],
        voice_description: str,
        language: str = "en",
    ) -> dict[str, Any]:
        normalized_description = voice_description.strip()
        if not normalized_description:
            raise ValueError("voice_description must not be empty")

        return self._enqueue_speech_job(
            mode=VOICE_DESIGN_MODEL_KIND,
            chunks=self._normalize_chunks(chunks),
            language=_normalize_language(language),
            voice_description=normalized_description,
        )

    def speak_text_as_clone(
        self,
        *,
        chunks: list[str],
        reference_audio_path: str,
        reference_text: str | None = None,
        language: str = "en",
    ) -> dict[str, Any]:
        normalized_reference_audio_path = reference_audio_path.strip()
        if not normalized_reference_audio_path:
            raise ValueError("reference_audio_path must not be empty")

        normalized_reference_text = reference_text.strip() if reference_text is not None else None
        if normalized_reference_text == "":
            normalized_reference_text = None

        reference_audio = self._decode_reference_audio_file(normalized_reference_audio_path)
        result = self._enqueue_speech_job(
            mode=CLONE_MODEL_KIND,
            chunks=self._normalize_chunks(chunks),
            language=_normalize_language(language),
            reference_audio=reference_audio,
            reference_audio_path=normalized_reference_audio_path,
            reference_text=normalized_reference_text,
        )
        result["clone_mode"] = (
            "reference_text" if normalized_reference_text is not None else "x_vector_only"
        )
        result["reference_audio_path"] = normalized_reference_audio_path
        result["reference_text_included"] = normalized_reference_text is not None
        result["clone_model_id"] = self.clone_model_id
        return result

    async def generate_speech_profile(
        self,
        *,
        state_store: Any,
        name: str,
        reference_audio_path: str,
        reference_text: str | None = None,
    ) -> dict[str, Any]:
        normalized_name = self._normalize_profile_name(name)
        normalized_reference_text = reference_text.strip() if reference_text is not None else None
        if normalized_reference_text == "":
            normalized_reference_text = None

        existing_profile = await self._load_speech_profile(
            state_store=state_store,
            name=normalized_name,
        )
        if existing_profile is not None:
            message = f"speech profile `{normalized_name}` already exists"
            self._speech_profile_last_error = message
            raise RuntimeError(message)

        reference_audio = self._decode_reference_audio_file(reference_audio_path)
        prompt_items = self._create_voice_clone_prompt_items(
            reference_audio=reference_audio,
            reference_text=normalized_reference_text,
        )
        now = _utc_now()
        profile = StoredSpeechProfile(
            name=normalized_name,
            clone_model_id=self.clone_model_id,
            clone_mode=(
                "reference_text" if normalized_reference_text is not None else "x_vector_only"
            ),
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
            prompt_items=[
                self._serialize_voice_clone_prompt_item(prompt_item)
                for prompt_item in prompt_items
            ],
        )

        names = await self._load_speech_profile_names(state_store)
        names.append(normalized_name)
        names = sorted(set(names))
        await self._save_speech_profile(
            state_store=state_store,
            profile=profile,
        )
        await self._save_speech_profile_names(state_store, names)

        self._speech_profile_count = len(names)
        self._speech_profile_last_error = None
        self._emit_runtime_event(
            "speech_profile_created",
            level="minimal",
            profile_name=normalized_name,
            clone_model_id=self.clone_model_id,
            clone_mode=profile.clone_mode,
        )
        return {
            "result": "success",
            **self._speech_profile_metadata(profile),
        }

    async def list_speech_profiles(self, *, state_store: Any) -> dict[str, Any]:
        names = await self._load_speech_profile_names(state_store)
        profiles: list[dict[str, Any]] = []
        for name in names:
            profile = await self._load_speech_profile(state_store=state_store, name=name)
            if profile is None:
                continue
            profiles.append(self._speech_profile_metadata(profile))

        self._speech_profile_count = len(profiles)
        self._speech_profile_last_error = None
        return {
            "result": "success",
            "profiles": profiles,
            "profile_count": len(profiles),
        }

    async def delete_speech_profile(self, *, state_store: Any, name: str) -> dict[str, Any]:
        normalized_name = self._normalize_profile_name(name)
        existing_profile = await self._load_speech_profile(
            state_store=state_store,
            name=normalized_name,
        )
        if existing_profile is None:
            message = f"speech profile `{normalized_name}` does not exist"
            self._speech_profile_last_error = message
            raise RuntimeError(message)

        await state_store.delete(
            key=self._speech_profile_key(normalized_name),
            collection=SPEECH_PROFILE_COLLECTION,
        )
        names = [
            candidate
            for candidate in await self._load_speech_profile_names(state_store)
            if candidate != normalized_name
        ]
        await self._save_speech_profile_names(state_store, names)
        self._speech_profile_count = len(names)
        self._speech_profile_last_error = None
        self._emit_runtime_event(
            "speech_profile_deleted",
            level="minimal",
            profile_name=normalized_name,
        )
        return {
            "result": "success",
            "deleted": True,
            "name": normalized_name,
            "profile_count": len(names),
        }

    async def speak_with_profile(
        self,
        *,
        state_store: Any,
        chunks: list[str],
        name: str,
        language: str = "en",
    ) -> dict[str, Any]:
        normalized_name = self._normalize_profile_name(name)
        profile = await self._load_speech_profile(
            state_store=state_store,
            name=normalized_name,
        )
        if profile is None:
            message = f"speech profile `{normalized_name}` does not exist"
            self._speech_profile_last_error = message
            raise RuntimeError(message)
        if profile.clone_model_id != self.clone_model_id:
            message = (
                f"speech profile `{normalized_name}` is bound to clone model "
                f"`{profile.clone_model_id}`, but the active clone model is `{self.clone_model_id}`"
            )
            self._speech_profile_last_error = message
            raise RuntimeError(message)

        prompt_items = self._deserialize_voice_clone_prompt_items(profile.prompt_items)
        result = self._enqueue_speech_job(
            mode=CLONE_MODEL_KIND,
            chunks=self._normalize_chunks(chunks),
            language=_normalize_language(language),
            voice_clone_prompt_items=prompt_items,
            profile_name=normalized_name,
        )
        result["profile_name"] = normalized_name
        result["clone_mode"] = profile.clone_mode
        result["clone_model_id"] = profile.clone_model_id
        self._speech_profile_last_error = None
        self._emit_runtime_event(
            "speech_profile_enqueued",
            level="minimal",
            profile_name=normalized_name,
            clone_model_id=profile.clone_model_id,
            clone_mode=profile.clone_mode,
        )
        return result

    def play_speech_chunks(
        self,
        *,
        mode: str,
        chunks: list[str],
        language: str = "en",
        voice_description: str | None = None,
        reference_audio: tuple[np.ndarray, int] | None = None,
        reference_text: str | None = None,
        voice_clone_prompt_items: list[Any] | None = None,
    ) -> dict[str, Any]:
        if not chunks:
            raise ValueError("chunks must not be empty")

        normalized_chunks = self._normalize_chunks(chunks)
        normalized_language = _normalize_language(language)
        if mode == VOICE_DESIGN_MODEL_KIND:
            normalized_description = (voice_description or "").strip()
            if not normalized_description:
                raise ValueError("voice_description must not be empty")
        else:
            normalized_description = None
            if reference_audio is None and voice_clone_prompt_items is None:
                raise ValueError(
                    "reference_audio or voice_clone_prompt_items must be provided "
                    "for clone playback"
                )

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

        stream_holder: list[sd.OutputStream | None] = [None]

        def close_output_stream(current_stream: sd.OutputStream) -> None:
            self._emit_runtime_event(
                "speech_output_stream_closing",
                level="debug",
                job_id=self._speech_current_job_id,
                mode=mode,
            )
            current_stream.stop()
            current_stream.close()
            stream_holder[0] = None
            self._emit_runtime_event(
                "speech_output_stream_closed",
                level="debug",
                job_id=self._speech_current_job_id,
                mode=mode,
            )

        def open_output_stream_for_playback() -> sd.OutputStream:
            assert sample_rate is not None
            assert channel_count is not None
            with self._lock:
                self._set_speech_phase_locked("opening_output")
            self._emit_runtime_event(
                "speech_output_stream_opening",
                level="info",
                job_id=self._speech_current_job_id,
                mode=mode,
                sample_rate=sample_rate,
                channel_count=channel_count,
                buffered_chunk_count=buffered_chunk_count,
                buffered_sample_count=buffered_sample_count,
            )
            opened_stream = self._open_output_stream(
                sample_rate=sample_rate,
                channel_count=channel_count,
            )
            stream_holder[0] = opened_stream
            self._emit_runtime_event(
                "speech_output_stream_opened",
                level="info",
                job_id=self._speech_current_job_id,
                mode=mode,
                sample_rate=sample_rate,
                channel_count=channel_count,
            )
            return opened_stream

        def write_chunk_with_recovery(
            current_stream: sd.OutputStream,
            *,
            chunk_index: int,
            waveform_to_write: np.ndarray,
            chunk_started_at: dt.datetime,
        ) -> sd.OutputStream:
            attempt = 0
            stream_for_write = current_stream
            while True:
                underflowed = self._write_output_stream_chunk(stream_for_write, waveform_to_write)
                if not underflowed:
                    return stream_for_write

                self._emit_runtime_event(
                    "speech_chunk_playback_underflow",
                    level="minimal",
                    job_id=self._speech_current_job_id,
                    mode=mode,
                    chunk_index=chunk_index,
                    chunk_count=len(normalized_chunks),
                    retry_attempt=attempt + 1,
                    max_retries=self.playback_underflow_retries,
                )
                if attempt >= self.playback_underflow_retries:
                    raise RuntimeError(
                        "audio output underflowed during streamed playback "
                        f"after {attempt + 1} attempt(s)"
                    )

                close_output_stream(stream_for_write)
                stream_for_write = open_output_stream_for_playback()
                playback_recovery_duration_ms = _duration_ms(chunk_started_at, _utc_now())
                self._emit_runtime_event(
                    "speech_chunk_playback_retrying",
                    level="minimal",
                    job_id=self._speech_current_job_id,
                    mode=mode,
                    chunk_index=chunk_index,
                    chunk_count=len(normalized_chunks),
                    retry_attempt=attempt + 1,
                    max_retries=self.playback_underflow_retries,
                    recovery_duration_ms=playback_recovery_duration_ms,
                )
                attempt += 1

        def synthesize_into_queue() -> None:
            try:
                for chunk_index, text_chunk in enumerate(normalized_chunks, start=1):
                    synth_started_at = _utc_now()
                    self._emit_runtime_event(
                        "speech_chunk_synthesis_started",
                        level="info",
                        job_id=self._speech_current_job_id,
                        mode=mode,
                        chunk_index=chunk_index,
                        chunk_count=len(normalized_chunks),
                        chunk_char_count=len(text_chunk),
                    )
                    chunk_result = self._synthesize_audio_chunk(
                        mode=mode,
                        text=text_chunk,
                        voice_description=normalized_description,
                        language=normalized_language,
                        reference_audio=reference_audio,
                        reference_text=reference_text,
                        voice_clone_prompt_items=voice_clone_prompt_items,
                    )
                    waveform = self._prepare_waveform_for_output_stream(chunk_result["waveform"])
                    self._emit_runtime_event(
                        "speech_chunk_synthesis_completed",
                        level="info",
                        job_id=self._speech_current_job_id,
                        mode=mode,
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
                    mode=mode,
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

                if stream_holder[0] is None:
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
                            mode=mode,
                            buffered_chunk_count=buffered_chunk_count,
                            buffered_sample_count=buffered_sample_count,
                            preroll_sample_target=preroll_sample_target,
                        )
                        continue

                    stream_holder[0] = open_output_stream_for_playback()
                    playback_started_at = _utc_now()

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
                            mode=mode,
                            chunk_index=next_chunk_index,
                            chunk_count=len(normalized_chunks),
                            sample_count=int(buffered_waveform.shape[0]),
                        )
                        assert self._current_chunk_started_at is not None
                        stream_holder[0] = write_chunk_with_recovery(
                            stream_holder[0],
                            chunk_index=next_chunk_index,
                            waveform_to_write=buffered_waveform,
                            chunk_started_at=self._current_chunk_started_at,
                        )
                        total_sample_count += int(buffered_waveform.shape[0])
                        self._emit_runtime_event(
                            "speech_chunk_playback_completed",
                            level="info",
                            job_id=self._speech_current_job_id,
                            mode=mode,
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

                assert stream_holder[0] is not None
                with self._lock:
                    self._set_speech_phase_locked("playing")
                    self._speech_current_chunk_index = next_chunk_index
                    self._speech_current_chunk_count = len(normalized_chunks)
                    self._current_chunk_started_at = _utc_now()
                self._emit_runtime_event(
                    "speech_chunk_playback_started",
                    level="info",
                    job_id=self._speech_current_job_id,
                    mode=mode,
                    chunk_index=next_chunk_index,
                    chunk_count=len(normalized_chunks),
                    sample_count=int(waveform.shape[0]),
                )
                assert self._current_chunk_started_at is not None
                stream_holder[0] = write_chunk_with_recovery(
                    stream_holder[0],
                    chunk_index=next_chunk_index,
                    waveform_to_write=waveform,
                    chunk_started_at=self._current_chunk_started_at,
                )
                total_sample_count += int(waveform.shape[0])
                self._emit_runtime_event(
                    "speech_chunk_playback_completed",
                    level="info",
                    job_id=self._speech_current_job_id,
                    mode=mode,
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

            if stream_holder[0] is None:
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
                "mode": mode,
            }
        finally:
            producer_thread.join(timeout=1)
            if stream_holder[0] is not None:
                close_output_stream(stream_holder[0])

    def _speech_worker_loop(self) -> None:
        while True:
            job = self._speech_queue.get()
            if job is None:
                self._speech_queue.task_done()
                self._emit_runtime_event("speech_worker_stopped", level="minimal")
                return

            job_id = int(job["job_id"])
            mode = str(job["mode"])
            chunks = list(job["chunks"])
            language = str(job["language"])

            with self._lock:
                self._speech_in_progress = True
                self._current_job_started_at = _utc_now()
                self._set_speech_phase_locked("synthesizing")
                self._speech_current_job_id = job_id
                self._speech_current_chunk_index = 0
                self._speech_current_chunk_count = len(chunks)
                self._speech_current_mode = mode
                self._speech_last_error = None
            self._emit_runtime_event(
                "speech_job_started",
                level="minimal",
                job_id=job_id,
                mode=mode,
                chunk_count=len(chunks),
                queue_depth=self._speech_queue.qsize(),
                enqueue_to_start_ms=_duration_ms(
                    self._speech_last_enqueued_at,
                    self._current_job_started_at,
                ),
            )

            try:
                if mode == VOICE_DESIGN_MODEL_KIND:
                    self.play_speech_chunks(
                        mode=mode,
                        chunks=chunks,
                        voice_description=str(job["voice_description"]),
                        language=language,
                    )
                elif mode == CLONE_MODEL_KIND:
                    self.play_speech_chunks(
                        mode=mode,
                        chunks=chunks,
                        language=language,
                        reference_audio=cast(
                            tuple[np.ndarray, int] | None,
                            job.get("reference_audio"),
                        ),
                        reference_text=cast(str | None, job.get("reference_text")),
                        voice_clone_prompt_items=cast(
                            list[Any] | None,
                            job.get("voice_clone_prompt_items"),
                        ),
                    )
                else:
                    raise RuntimeError(f"unsupported speech mode `{mode}`")
            except Exception as exc:
                with self._lock:
                    self._speech_jobs_failed += 1
                    self._speech_last_error = str(exc)
                    self._last_error = str(exc)
                self._emit_runtime_event(
                    "speech_job_failed",
                    level="minimal",
                    job_id=job_id,
                    mode=mode,
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
                self._emit_runtime_event(
                    "speech_job_completed",
                    level="minimal",
                    job_id=job_id,
                    mode=mode,
                    total_job_duration_ms=_duration_ms(self._current_job_started_at, completed_at),
                )
            finally:
                with self._lock:
                    self._speech_in_progress = False
                    self._set_speech_phase_locked(DEFAULT_SPEECH_PHASE)
                    self._speech_current_job_id = None
                    self._speech_current_chunk_index = None
                    self._speech_current_chunk_count = None
                    self._speech_current_mode = None
                    self._current_job_started_at = None
                    self._current_chunk_started_at = None
                self._speech_queue.task_done()

    def _status_payload_locked(self) -> dict[str, Any]:
        dependency_status = _runtime_dependency_status()
        voice_design_state = self._model_state(VOICE_DESIGN_MODEL_KIND)
        clone_state = self._model_state(CLONE_MODEL_KIND)
        overall_ready = (
            _runtime_dependencies_ready()
            and self._last_error is None
            and self._all_enabled_models_loaded_locked()
        )
        return {
            "ready": overall_ready,
            "model_loaded": voice_design_state["model"] is not None,
            "device_preference": self.device_preference,
            "device": voice_design_state["resolved_device"],
            "model_id": self.voice_design_model_id,
            "torch_dtype": self.torch_dtype_name,
            "cpu_fallback_active": voice_design_state["cpu_fallback_active"],
            "playback_preroll_seconds": self.playback_preroll_seconds,
            "playback_preroll_chunks": self.playback_preroll_chunks,
            "playback_waveform_queue_maxsize": self.playback_waveform_queue_maxsize,
            "playback_underflow_retries": self.playback_underflow_retries,
            "output_stream_latency": self.output_stream_latency,
            "log_level": self.log_level,
            "last_used_at": _timestamp_value(voice_design_state["last_used_at"]),
            "last_loaded_at": _timestamp_value(voice_design_state["last_loaded_at"]),
            "last_error": self._last_error,
            "voice_design_model_enabled": voice_design_state["enabled"],
            "voice_design_model_loaded": voice_design_state["model"] is not None,
            "voice_design_model_id": voice_design_state["model_id"],
            "voice_design_device": voice_design_state["resolved_device"],
            "voice_design_last_used_at": _timestamp_value(voice_design_state["last_used_at"]),
            "voice_design_last_loaded_at": _timestamp_value(voice_design_state["last_loaded_at"]),
            "voice_design_last_error": voice_design_state["last_error"],
            "voice_design_cpu_fallback_active": voice_design_state["cpu_fallback_active"],
            "clone_model_enabled": clone_state["enabled"],
            "clone_model_loaded": clone_state["model"] is not None,
            "clone_model_id": clone_state["model_id"],
            "clone_device": clone_state["resolved_device"],
            "clone_last_used_at": _timestamp_value(clone_state["last_used_at"]),
            "clone_last_loaded_at": _timestamp_value(clone_state["last_loaded_at"]),
            "clone_last_error": clone_state["last_error"],
            "clone_cpu_fallback_active": clone_state["cpu_fallback_active"],
            "clone_in_progress": (
                self._speech_in_progress
                and self._speech_current_mode == CLONE_MODEL_KIND
            ),
            "speech_profile_storage_ready": True,
            "speech_profile_count": self._speech_profile_count,
            "speech_profile_last_error": self._speech_profile_last_error,
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
            "speech_current_mode": self._speech_current_mode,
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

    def _enqueue_speech_job(
        self,
        *,
        mode: str,
        chunks: list[str],
        language: str,
        voice_description: str | None = None,
        reference_audio: tuple[np.ndarray, int] | None = None,
        reference_audio_path: str | None = None,
        reference_text: str | None = None,
        voice_clone_prompt_items: list[Any] | None = None,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        self.start_speech_worker()

        with self._lock:
            slot = self._model_state(mode)
            if self._shutdown_requested.is_set():
                raise RuntimeError("runtime is shutting down and cannot accept speech jobs")
            if not slot["enabled"]:
                raise RuntimeError(f"{mode} model is disabled and cannot accept speech jobs")

            job_id = self._speech_jobs_queued + 1
            job: dict[str, Any] = {
                "job_id": job_id,
                "mode": mode,
                "chunks": chunks,
                "language": language,
            }
            if voice_description is not None:
                job["voice_description"] = voice_description
            if reference_audio is not None:
                job["reference_audio"] = reference_audio
            if reference_audio_path is not None:
                job["reference_audio_path"] = reference_audio_path
            if reference_text is not None:
                job["reference_text"] = reference_text
            if voice_clone_prompt_items is not None:
                job["voice_clone_prompt_items"] = voice_clone_prompt_items
            if profile_name is not None:
                job["profile_name"] = profile_name

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
                mode=mode,
                chunk_count=len(chunks),
                chunk_char_count=sum(len(chunk) for chunk in chunks),
                language=language,
                queue_depth=self._speech_queue.qsize(),
                reference_text_included=reference_text is not None,
                profile_name=profile_name,
            )
            return {
                "result": "success",
                "queued": True,
                "job_id": job_id,
                "chunked": len(chunks) > 1,
                "chunk_count": len(chunks),
                "language": language,
                "enqueued_at": enqueued_at.isoformat(),
                "playback_mode": "in-process-queue",
                "mode": mode,
                **self._speech_status_payload_locked(),
            }

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

    def _load_model_kwargs(self, model_kind: str, torch_module: Any) -> dict[str, Any]:
        load_kwargs: dict[str, Any] = {}
        torch_dtype = self._resolve_torch_dtype(torch_module)
        slot = self._model_state(model_kind)

        if slot["cpu_fallback_active"] and self.device_preference == "auto":
            load_kwargs["device_map"] = "cpu"
            load_kwargs["torch_dtype"] = torch_dtype or torch_module.float32
            return load_kwargs

        load_kwargs["device_map"] = self._resolve_device(torch_module)
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        return load_kwargs

    def _reload_model_with_cpu_fallback(self, model_kind: str) -> None:
        with self._lock:
            if self.device_preference != "auto":
                raise RuntimeError(
                    "CPU fallback is only supported when SPEAK_TO_USER_DEVICE=auto"
                )
            slot = self._model_state(model_kind)
            previous_model = slot["model"]
            previous_device = slot["resolved_device"]
            slot["model"] = None
            slot["resolved_device"] = None
            slot["cpu_fallback_active"] = True
        self._emit_runtime_event(
            "model_cpu_fallback_activated",
            level="minimal",
            model_kind=model_kind,
        )

        self._release_model_resources(previous_model, resolved_device=previous_device)
        gc.collect()

        loaded_model = self._load_model_impl(model_kind)

        with self._lock:
            slot = self._model_state(model_kind)
            if self._shutdown_requested.is_set():
                self._release_model_resources(loaded_model, resolved_device=slot["resolved_device"])
                slot["resolved_device"] = None
                raise RuntimeError("model reload aborted during shutdown")

            slot["model"] = loaded_model
            slot["last_error"] = None
            slot["last_loaded_at"] = _utc_now()
            self._touch_model_locked(model_kind, slot["last_loaded_at"])
            self._last_error = None

    def _synthesize_audio_chunk(
        self,
        *,
        mode: str,
        text: str,
        language: str,
        voice_description: str | None = None,
        reference_audio: tuple[np.ndarray, int] | None = None,
        reference_text: str | None = None,
        voice_clone_prompt_items: list[Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            model_missing = self._model_state(mode)["model"] is None

        if model_missing:
            self.load_model(model_kind=mode)

        with self._lock:
            slot = self._model_state(mode)
            model = slot["model"]

        assert model is not None

        def generate(active_model: Any) -> tuple[list[Any], int]:
            if mode == VOICE_DESIGN_MODEL_KIND:
                wavs, sample_rate = active_model.generate_voice_design(
                    text=text,
                    language=language,
                    instruct=voice_description,
                    non_streaming_mode=True,
                )
                return cast(list[Any], wavs), int(sample_rate)

            generate_kwargs: dict[str, Any] = {
                "text": text,
                "language": language,
                "non_streaming_mode": True,
            }
            if voice_clone_prompt_items is not None:
                generate_kwargs["voice_clone_prompt"] = voice_clone_prompt_items
            else:
                assert reference_audio is not None
                generate_kwargs["ref_audio"] = reference_audio
                generate_kwargs["x_vector_only_mode"] = reference_text is None
                if reference_text is not None:
                    generate_kwargs["ref_text"] = reference_text
            wavs, sample_rate = active_model.generate_voice_clone(**generate_kwargs)
            return cast(list[Any], wavs), int(sample_rate)

        try:
            wavs, sample_rate = generate(model)
        except Exception as exc:
            if _meta_tensor_runtime_error(exc):
                self._emit_runtime_event(
                    "speech_chunk_synthesis_retrying_with_cpu_fallback",
                    level="minimal",
                    job_id=self._speech_current_job_id,
                    mode=mode,
                    error=str(exc),
                )
                try:
                    self._reload_model_with_cpu_fallback(mode)
                    with self._lock:
                        reloaded_model = self._model_state(mode)["model"]
                    if reloaded_model is None:
                        raise RuntimeError("CPU fallback reload completed without a loaded model")
                    wavs, sample_rate = generate(reloaded_model)
                except Exception as retry_exc:
                    with self._lock:
                        self._last_error = f"synthesis failed after CPU fallback retry: {retry_exc}"
                        self._model_state(mode)["last_error"] = self._last_error
                    self._emit_runtime_event(
                        "speech_chunk_synthesis_retry_failed",
                        level="minimal",
                        job_id=self._speech_current_job_id,
                        mode=mode,
                        error=str(retry_exc),
                    )
                    raise RuntimeError(
                        f"synthesis failed after CPU fallback retry: {retry_exc}"
                    ) from retry_exc
            else:
                with self._lock:
                    self._last_error = str(exc)
                    self._model_state(mode)["last_error"] = str(exc)
                self._emit_runtime_event(
                    "speech_chunk_synthesis_failed",
                    level="minimal",
                    job_id=self._speech_current_job_id,
                    mode=mode,
                    error=str(exc),
                )
                raise

        waveforms = [self._coerce_waveform(waveform) for waveform in wavs]
        if not waveforms:
            raise RuntimeError("model returned no waveform for synthesized chunk")

        with self._lock:
            slot = self._model_state(mode)
            now = _utc_now()
            slot["last_error"] = None
            self._last_error = None
            self._touch_model_locked(mode, now)
            return {
                "waveform": waveforms[0],
                "sample_rate": sample_rate,
                "model_id": slot["model_id"],
                "device": slot["resolved_device"],
                "language": language,
            }

    def _load_model_impl(self, model_kind: str) -> Any:
        with _suppress_default_stdout_prints_for_current_thread():
            import torch
            from qwen_tts import Qwen3TTSModel  # type: ignore[import-untyped]

            load_kwargs = self._load_model_kwargs(model_kind, torch)
            resolved_device = str(load_kwargs["device_map"])
            self._model_state(model_kind)["resolved_device"] = resolved_device
            return Qwen3TTSModel.from_pretrained(
                self._model_state(model_kind)["model_id"],
                **load_kwargs,
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

    def _new_model_slot(self, *, model_id: str, enabled: bool) -> dict[str, Any]:
        return {
            "enabled": enabled,
            "model_id": model_id,
            "model": None,
            "resolved_device": None,
            "last_error": None,
            "last_used_at": None,
            "last_loaded_at": None,
            "cpu_fallback_active": False,
        }

    def _enabled_model_kinds(self) -> list[str]:
        with self._lock:
            return [
                model_kind
                for model_kind, slot in self._model_slots.items()
                if cast(bool, slot["enabled"])
            ]

    def _all_enabled_models_loaded_locked(self) -> bool:
        enabled_slots = [
            slot for slot in self._model_slots.values() if cast(bool, slot["enabled"])
        ]
        if not enabled_slots:
            return False
        return all(slot["model"] is not None for slot in enabled_slots)

    def _model_state(self, model_kind: str) -> dict[str, Any]:
        return self._model_slots[model_kind]

    def _touch_model_locked(self, model_kind: str, value: dt.datetime | None = None) -> None:
        self._model_state(model_kind)["last_used_at"] = value or _utc_now()

    def _set_speech_phase_locked(self, phase: str) -> None:
        if self._speech_phase != phase or self._current_phase_started_at is None:
            self._current_phase_started_at = _utc_now()
        self._speech_phase = phase

    def _unload_model_locked(self, model_kind: str) -> None:
        slot = self._model_state(model_kind)
        model = slot["model"]
        resolved_device = slot["resolved_device"]
        slot["model"] = None
        slot["resolved_device"] = None
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

    def _decode_reference_audio_file(self, reference_audio_path: str) -> tuple[np.ndarray, int]:
        expanded_path = Path(reference_audio_path).expanduser()
        if not expanded_path.is_file():
            raise ValueError(f"reference audio file does not exist: {expanded_path}")

        try:
            import soundfile as sf  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "soundfile is unavailable for clone reference audio decoding"
            ) from exc

        try:
            waveform, sample_rate = sf.read(expanded_path, dtype="float32")
        except Exception as exc:
            raise ValueError(f"failed to read reference audio file: {expanded_path}") from exc

        waveform_array = np.asarray(waveform, dtype=np.float32)
        if waveform_array.size == 0:
            raise ValueError(f"reference audio file is empty: {expanded_path}")
        return waveform_array, int(sample_rate)

    def _normalize_profile_name(self, name: str) -> str:
        normalized = name.strip()
        if not normalized:
            raise ValueError("name must not be empty")
        return normalized

    def _speech_profile_key(self, name: str) -> str:
        return f"profile:{name}"

    async def _load_speech_profile_names(self, state_store: Any) -> list[str]:
        result = await state_store.get(
            key=SPEECH_PROFILE_INDEX_KEY,
            collection=SPEECH_PROFILE_COLLECTION,
            default=StateValue(value={"names": []}),
        )
        if result is None:
            return []
        index = StoredSpeechProfileIndex.model_validate(result.value)
        return list(index.names)

    async def _save_speech_profile_names(self, state_store: Any, names: list[str]) -> None:
        index = StoredSpeechProfileIndex(names=sorted(names))
        await state_store.put(
            key=SPEECH_PROFILE_INDEX_KEY,
            value=StateValue(value=index.model_dump(mode="json")),
            collection=SPEECH_PROFILE_COLLECTION,
        )

    async def _save_speech_profile(self, *, state_store: Any, profile: StoredSpeechProfile) -> None:
        await state_store.put(
            key=self._speech_profile_key(profile.name),
            value=StateValue(value=profile.model_dump(mode="json")),
            collection=SPEECH_PROFILE_COLLECTION,
        )

    async def _load_speech_profile(
        self,
        *,
        state_store: Any,
        name: str,
    ) -> StoredSpeechProfile | None:
        result = await state_store.get(
            key=self._speech_profile_key(name),
            collection=SPEECH_PROFILE_COLLECTION,
        )
        if result is None:
            return None
        return StoredSpeechProfile.model_validate(result.value)

    def _speech_profile_metadata(self, profile: StoredSpeechProfile) -> dict[str, Any]:
        return {
            "name": profile.name,
            "clone_model_id": profile.clone_model_id,
            "clone_mode": profile.clone_mode,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
            "reference_text_included": profile.clone_mode == "reference_text",
        }

    def _serialize_voice_clone_prompt_item(self, prompt_item: Any) -> StoredSpeechProfilePromptItem:
        return StoredSpeechProfilePromptItem(
            ref_code=self._serialize_tensor(prompt_item.ref_code)
            if prompt_item.ref_code is not None
            else None,
            ref_spk_embedding=self._serialize_tensor(prompt_item.ref_spk_embedding),
            x_vector_only_mode=bool(prompt_item.x_vector_only_mode),
            icl_mode=bool(prompt_item.icl_mode),
            ref_text=cast(str | None, prompt_item.ref_text),
        )

    def _deserialize_voice_clone_prompt_items(
        self,
        prompt_items: list[StoredSpeechProfilePromptItem],
    ) -> list[Any]:
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem  # type: ignore[import-untyped]

        return [
            VoiceClonePromptItem(
                ref_code=self._deserialize_tensor(prompt_item.ref_code)
                if prompt_item.ref_code is not None
                else None,
                ref_spk_embedding=cast(
                    Any,
                    self._deserialize_tensor(prompt_item.ref_spk_embedding),
                ),
                x_vector_only_mode=prompt_item.x_vector_only_mode,
                icl_mode=prompt_item.icl_mode,
                ref_text=prompt_item.ref_text,
            )
            for prompt_item in prompt_items
        ]

    def _serialize_tensor(self, tensor: Any) -> SerializedTensor:
        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        if hasattr(tensor, "tolist"):
            data = tensor.tolist()
        else:
            data = np.asarray(tensor).tolist()
        dtype_name = str(getattr(tensor, "dtype", np.asarray(tensor).dtype))
        return SerializedTensor(dtype=dtype_name, data=cast(list[Any], data))

    def _deserialize_tensor(self, serialized: SerializedTensor) -> Any:
        import torch

        return torch.tensor(
            serialized.data,
            dtype=self._torch_dtype_from_name(serialized.dtype, torch),
        )

    def _torch_dtype_from_name(self, dtype_name: str, torch_module: Any) -> Any:
        name = dtype_name.removeprefix("torch.")
        if hasattr(torch_module, name):
            return getattr(torch_module, name)
        if name == "float32":
            return torch_module.float32
        if name == "float16":
            return torch_module.float16
        if name == "bfloat16":
            return torch_module.bfloat16
        if name == "int64":
            return torch_module.int64
        if name == "int32":
            return torch_module.int32
        if name == "int16":
            return torch_module.int16
        if name == "int8":
            return torch_module.int8
        if name == "uint8":
            return torch_module.uint8
        raise RuntimeError(f"unsupported serialized tensor dtype `{dtype_name}`")

    def _create_voice_clone_prompt_items(
        self,
        *,
        reference_audio: tuple[np.ndarray, int],
        reference_text: str | None = None,
    ) -> list[Any]:
        with self._lock:
            model_missing = self._model_state(CLONE_MODEL_KIND)["model"] is None

        if model_missing:
            self.load_model(model_kind=CLONE_MODEL_KIND)

        with self._lock:
            model = self._model_state(CLONE_MODEL_KIND)["model"]

        assert model is not None

        def build_prompt(active_model: Any) -> list[Any]:
            return cast(
                list[Any],
                active_model.create_voice_clone_prompt(
                    ref_audio=reference_audio,
                    ref_text=reference_text,
                    x_vector_only_mode=reference_text is None,
                ),
            )

        try:
            prompt_items = build_prompt(model)
        except Exception as exc:
            if _meta_tensor_runtime_error(exc):
                self._emit_runtime_event(
                    "speech_profile_prompt_retrying_with_cpu_fallback",
                    level="minimal",
                    error=str(exc),
                )
                self._reload_model_with_cpu_fallback(CLONE_MODEL_KIND)
                with self._lock:
                    reloaded_model = self._model_state(CLONE_MODEL_KIND)["model"]
                if reloaded_model is None:
                    raise RuntimeError(
                        "CPU fallback reload completed without a clone model"
                    ) from exc
                prompt_items = build_prompt(reloaded_model)
            else:
                self._speech_profile_last_error = str(exc)
                self._emit_runtime_event(
                    "speech_profile_prompt_build_failed",
                    level="minimal",
                    error=str(exc),
                )
                raise

        with self._lock:
            self._touch_model_locked(CLONE_MODEL_KIND)
        return prompt_items

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
    ) -> bool:
        underflowed = stream.write(waveform)
        return bool(underflowed)
