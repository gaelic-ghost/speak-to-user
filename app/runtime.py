from __future__ import annotations

# MARK: Imports

from collections import deque
from dataclasses import asdict, dataclass
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import threading
from typing import Any, Literal, cast
import uuid

import numpy as np
import sounddevice as sd
import soundfile as sf


# MARK: Constants

VOICE_DESIGN_MODEL_KIND = "voice_design"
CLONE_MODEL_KIND = "clone"
PROFILE_COLLECTION = "speech_profiles"
STARTUP_MODEL_COLLECTION = "runtime_config"
STARTUP_MODEL_KEY = "startup_model_option"
DEFAULT_STATE_DIR = Path.home() / ".local" / "gaelic-ghost" / "speak-to-user" / "profiles"
DEFAULT_VOICE_DESIGN_MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
DEFAULT_CLONE_MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-6bit"
RECENT_EVENT_LIMIT = 25


# MARK: Data Structures

PlaybackBackend = Literal["sounddevice", "null"]


@dataclass(slots=True)
class SpeechProfile:
    name: str
    model_id: str
    reference_audio_path: str
    reference_text: str | None
    language: str
    source_kind: str
    created_at: str
    seed_text: str | None = None
    voice_description: str | None = None


# MARK: Runtime

class TTSRuntime:
    def __init__(
        self,
        *,
        voice_design_model_id: str,
        clone_model_id: str,
        enable_voice_design_model: bool,
        enable_clone_model: bool,
        playback_backend: PlaybackBackend,
        state_dir: Path,
        tts_chunk_max_chars: int = 160,
    ) -> None:
        self.voice_design_model_id = voice_design_model_id
        self.clone_model_id = clone_model_id
        self.enable_voice_design_model = enable_voice_design_model
        self.enable_clone_model = enable_clone_model
        self.playback_backend = playback_backend
        self.state_dir = state_dir
        self.tts_chunk_max_chars = tts_chunk_max_chars
        self.profiles_dir = self.state_dir / "profile_audio"

        self._models: dict[str, Any] = {}
        self._model_lock = threading.RLock()
        self._speech_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._recent_events: deque[dict[str, object]] = deque(maxlen=RECENT_EVENT_LIMIT)

        self._startup_model_option = self._default_startup_model_option()
        self._speech_in_progress = False
        self._speech_phase = "idle"
        self._speech_current_job_id: str | None = None
        self._speech_jobs_completed = 0
        self._speech_jobs_failed = 0
        self._speech_last_error: str | None = None
        self._speech_last_event: str | None = None
        self._speech_last_event_at: str | None = None
        self._last_loaded_at: dict[str, str | None] = {
            VOICE_DESIGN_MODEL_KIND: None,
            CLONE_MODEL_KIND: None,
        }
        self._last_used_at: dict[str, str | None] = {
            VOICE_DESIGN_MODEL_KIND: None,
            CLONE_MODEL_KIND: None,
        }
        self._last_model_error: dict[str, str | None] = {
            VOICE_DESIGN_MODEL_KIND: None,
            CLONE_MODEL_KIND: None,
        }

    @classmethod
    def from_env(cls) -> TTSRuntime:
        return cls(
            voice_design_model_id=os.getenv(
                "SPEAK_TO_USER_VOICE_DESIGN_MODEL_ID",
                DEFAULT_VOICE_DESIGN_MODEL_ID,
            ).strip()
            or DEFAULT_VOICE_DESIGN_MODEL_ID,
            clone_model_id=os.getenv(
                "SPEAK_TO_USER_CLONE_MODEL_ID",
                DEFAULT_CLONE_MODEL_ID,
            ).strip()
            or DEFAULT_CLONE_MODEL_ID,
            enable_voice_design_model=_env_bool(
                "SPEAK_TO_USER_ENABLE_VOICE_DESIGN_MODEL",
                default=True,
            ),
            enable_clone_model=_env_bool(
                "SPEAK_TO_USER_ENABLE_CLONE_MODEL",
                default=True,
            ),
            playback_backend=_env_playback_backend(
                os.getenv("SPEAK_TO_USER_PLAYBACK_BACKEND", "sounddevice")
            ),
            state_dir=_env_state_dir(os.getenv("SPEAK_TO_USER_STATE_DIR")),
            tts_chunk_max_chars=_env_positive_int(
                "SPEAK_TO_USER_TTS_CHUNK_MAX_CHARS",
                default=160,
            ),
        )

    # MARK: Lifecycle

    async def preload(self, *, state_store: Any) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        stored_option = await state_store.get(
            STARTUP_MODEL_KEY,
            collection=STARTUP_MODEL_COLLECTION,
            default=self._default_startup_model_option(),
        )
        option = (
            stored_option
            if isinstance(stored_option, str)
            else self._default_startup_model_option()
        )
        self._startup_model_option = option

        for model_kind in self._model_kinds_for_option(option):
            self.load_model(model_kind=model_kind)

    def shutdown(self) -> None:
        with self._model_lock:
            self._models.clear()

    # MARK: Model Management

    def available_model_ids(self) -> list[str]:
        model_ids: list[str] = []
        if self.enable_voice_design_model:
            model_ids.append(self.voice_design_model_id)
        if self.enable_clone_model:
            model_ids.append(self.clone_model_id)
        return model_ids

    def model_kind_for_id(self, model_id: str) -> str:
        if model_id == self.voice_design_model_id:
            return VOICE_DESIGN_MODEL_KIND
        if model_id == self.clone_model_id:
            return CLONE_MODEL_KIND
        raise ValueError(
            "Unknown model id "
            f"`{model_id}`. Expected one of: {', '.join(self.available_model_ids())}"
        )

    def missing_required_model_ids(self, *, model_kinds: list[str]) -> list[str]:
        missing: list[str] = []
        for model_kind in model_kinds:
            if not self._is_model_loaded(model_kind):
                missing.append(self._model_id_for_kind(model_kind))
        return missing

    def required_models_not_loaded_message(self, *, model_kinds: list[str]) -> str:
        missing = self.missing_required_model_ids(model_kinds=model_kinds)
        return "Required TTS models are not loaded yet: " + ", ".join(missing)

    def load_model(self, *, model_kind: str) -> dict[str, object]:
        self._require_enabled_model_kind(model_kind)
        with self._model_lock:
            if model_kind in self._models:
                return {
                    "result": "success",
                    "loaded": True,
                    "model_id": self._model_id_for_kind(model_kind),
                    **self.status(),
                }

            try:
                self._models[model_kind] = self._load_model_impl(model_kind)
            except Exception as exc:
                message = (
                    f"Failed to load `{self._model_id_for_kind(model_kind)}` with mlx-audio. "
                    "The most likely causes are a missing `mlx-audio` install, "
                    "an invalid model id, "
                    f"or insufficient unified memory. Original error: {exc}"
                )
                self._last_model_error[model_kind] = message
                raise RuntimeError(message) from exc

            self._last_loaded_at[model_kind] = _utc_now()
            self._last_model_error[model_kind] = None
            self._record_event("model_loaded", model_kind=model_kind)
            return {
                "result": "success",
                "loaded": True,
                "model_id": self._model_id_for_kind(model_kind),
                **self.status(),
            }

    def unload_model(self, *, model_kind: str) -> dict[str, object]:
        with self._model_lock:
            self._models.pop(model_kind, None)
        self._record_event("model_unloaded", model_kind=model_kind)
        return {
            "result": "success",
            "unloaded": True,
            "model_id": self._model_id_for_kind(model_kind),
            **self.status(),
        }

    async def set_startup_model(self, *, state_store: Any, option: str) -> dict[str, object]:
        normalized_option = option.strip()
        if normalized_option not in self.startup_model_options():
            raise ValueError(
                "Invalid startup model option. Expected one of: "
                f"{', '.join(self.startup_model_options())}"
            )
        await state_store.put(
            STARTUP_MODEL_KEY,
            normalized_option,
            collection=STARTUP_MODEL_COLLECTION,
        )
        self._startup_model_option = normalized_option
        self._record_event("startup_model_option_changed", option=normalized_option)
        return {
            "result": "success",
            "startup_model_option": normalized_option,
            **self.status(),
        }

    def startup_model_options(self) -> list[str]:
        return ["none", "all", *self.available_model_ids()]

    # MARK: Speech

    def speak_text(
        self,
        *,
        chunks: list[str],
        voice_description: str,
        language: str,
    ) -> dict[str, object]:
        if not voice_description.strip():
            raise ValueError("voice_description must not be empty")
        return self._run_speech_job(
            mode="voice_design",
            model_kind=VOICE_DESIGN_MODEL_KIND,
            chunks=chunks,
            language=language,
            instruct=voice_description.strip(),
        )

    def speak_text_as_clone(
        self,
        *,
        chunks: list[str],
        reference_audio_path: str,
        reference_text: str | None,
        language: str,
    ) -> dict[str, object]:
        reference_path = self._validated_reference_audio_path(reference_audio_path)
        return self._run_speech_job(
            mode="clone",
            model_kind=CLONE_MODEL_KIND,
            chunks=chunks,
            language=language,
            ref_audio=str(reference_path),
            ref_text=reference_text.strip() if reference_text else None,
        )

    async def generate_speech_profile(
        self,
        *,
        state_store: Any,
        name: str,
        reference_audio_path: str,
        reference_text: str | None,
    ) -> dict[str, object]:
        profile_name = _normalize_profile_name(name)
        existing = await self._get_profile(state_store, profile_name)
        if existing is not None:
            raise ValueError(f"Speech profile `{profile_name}` already exists")

        source_path = self._validated_reference_audio_path(reference_audio_path)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        stored_path = self.profiles_dir / f"{profile_name}{source_path.suffix.lower() or '.wav'}"
        shutil.copy2(source_path, stored_path)

        profile = SpeechProfile(
            name=profile_name,
            model_id=self.clone_model_id,
            reference_audio_path=str(stored_path),
            reference_text=reference_text.strip() if reference_text else None,
            language="en",
            source_kind="reference_audio",
            created_at=_utc_now(),
        )
        await self._put_profile(state_store, profile)
        self._record_event("speech_profile_created", name=profile_name)
        return {
            "result": "success",
            "profile": asdict(profile),
        }

    async def generate_speech_profile_from_voice_design(
        self,
        *,
        state_store: Any,
        name: str,
        text: str,
        voice_description: str,
        language: str,
    ) -> dict[str, object]:
        profile_name = _normalize_profile_name(name)
        existing = await self._get_profile(state_store, profile_name)
        if existing is not None:
            raise ValueError(f"Speech profile `{profile_name}` already exists")

        waveform, sample_rate = self._synthesize_chunks(
            model_kind=VOICE_DESIGN_MODEL_KIND,
            chunks=[text.strip()],
            language=language,
            instruct=voice_description.strip(),
        )

        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        stored_path = self.profiles_dir / f"{profile_name}.wav"
        sf.write(stored_path, waveform, sample_rate)

        profile = SpeechProfile(
            name=profile_name,
            model_id=self.clone_model_id,
            reference_audio_path=str(stored_path),
            reference_text=text.strip(),
            language=language,
            source_kind="voice_design_seed",
            created_at=_utc_now(),
            seed_text=text.strip(),
            voice_description=voice_description.strip(),
        )
        await self._put_profile(state_store, profile)
        self._record_event("speech_profile_created_from_voice_design", name=profile_name)
        return {
            "result": "success",
            "profile": asdict(profile),
        }

    async def list_speech_profiles(self, *, state_store: Any) -> dict[str, object]:
        profiles = await self._list_profiles(state_store)
        return {
            "result": "success",
            "profiles": [asdict(profile) for profile in profiles],
            "profile_count": len(profiles),
        }

    async def delete_speech_profile(self, *, state_store: Any, name: str) -> dict[str, object]:
        profile_name = _normalize_profile_name(name)
        profile = await self._get_profile(state_store, profile_name)
        if profile is None:
            raise ValueError(f"Speech profile `{profile_name}` does not exist")

        await state_store.delete(profile_name, collection=PROFILE_COLLECTION)
        profile_path = Path(profile.reference_audio_path)
        if profile_path.exists():
            profile_path.unlink()
        self._record_event("speech_profile_deleted", name=profile_name)
        return {
            "result": "success",
            "deleted": True,
            "name": profile_name,
        }

    async def resolve_profile(self, *, state_store: Any, name: str) -> SpeechProfile:
        profile_name = _normalize_profile_name(name)
        profile = await self._get_profile(state_store, profile_name)
        if profile is None:
            raise ValueError(f"Speech profile `{profile_name}` does not exist")
        if profile.model_id != self.clone_model_id:
            raise ValueError(
                f"Speech profile `{profile_name}` targets `{profile.model_id}`, "
                f"but the current clone model is `{self.clone_model_id}`"
            )
        return profile

    def speak_with_profile(
        self,
        *,
        name: str,
        profile: SpeechProfile,
        chunks: list[str],
        language: str,
    ) -> dict[str, object]:
        del name
        return self._run_speech_job(
            mode="profile",
            model_kind=CLONE_MODEL_KIND,
            chunks=chunks,
            language=language,
            ref_audio=profile.reference_audio_path,
            ref_text=profile.reference_text,
            profile_name=profile.name,
        )

    # MARK: Status

    def status(self) -> dict[str, object]:
        with self._status_lock:
            return {
                "ready": True,
                "playback_backend": self.playback_backend,
                "state_dir": str(self.state_dir),
                "profiles_dir": str(self.profiles_dir),
                "startup_model_option": self._startup_model_option,
                "startup_model_options": self.startup_model_options(),
                "loaded_model_ids": self._loaded_model_ids(),
                "voice_design_model_enabled": self.enable_voice_design_model,
                "voice_design_model_loaded": self._is_model_loaded(VOICE_DESIGN_MODEL_KIND),
                "voice_design_model_id": self.voice_design_model_id,
                "voice_design_last_loaded_at": self._last_loaded_at[VOICE_DESIGN_MODEL_KIND],
                "voice_design_last_used_at": self._last_used_at[VOICE_DESIGN_MODEL_KIND],
                "voice_design_last_error": self._last_model_error[VOICE_DESIGN_MODEL_KIND],
                "clone_model_enabled": self.enable_clone_model,
                "clone_model_loaded": self._is_model_loaded(CLONE_MODEL_KIND),
                "clone_model_id": self.clone_model_id,
                "clone_last_loaded_at": self._last_loaded_at[CLONE_MODEL_KIND],
                "clone_last_used_at": self._last_used_at[CLONE_MODEL_KIND],
                "clone_last_error": self._last_model_error[CLONE_MODEL_KIND],
                "speech_in_progress": self._speech_in_progress,
                "speech_phase": self._speech_phase,
                "speech_current_job_id": self._speech_current_job_id,
                "speech_jobs_completed": self._speech_jobs_completed,
                "speech_jobs_failed": self._speech_jobs_failed,
                "speech_last_error": self._speech_last_error,
                "speech_last_event": self._speech_last_event,
                "speech_last_event_at": self._speech_last_event_at,
                "tts_chunk_max_chars": self.tts_chunk_max_chars,
                "recent_events": list(self._recent_events),
            }

    # MARK: Internal Helpers

    def _default_startup_model_option(self) -> str:
        enabled_model_ids = self.available_model_ids()
        if not enabled_model_ids:
            return "none"
        if len(enabled_model_ids) > 1:
            return "all"
        return enabled_model_ids[0]

    def _model_kinds_for_option(self, option: str) -> list[str]:
        if option == "none":
            return []
        if option == "all":
            return [self.model_kind_for_id(model_id) for model_id in self.available_model_ids()]
        return [self.model_kind_for_id(option)]

    def _require_enabled_model_kind(self, model_kind: str) -> None:
        if model_kind == VOICE_DESIGN_MODEL_KIND and not self.enable_voice_design_model:
            raise ValueError("The voice-design model is disabled in this runtime configuration")
        if model_kind == CLONE_MODEL_KIND and not self.enable_clone_model:
            raise ValueError("The clone model is disabled in this runtime configuration")

    def _model_id_for_kind(self, model_kind: str) -> str:
        if model_kind == VOICE_DESIGN_MODEL_KIND:
            return self.voice_design_model_id
        if model_kind == CLONE_MODEL_KIND:
            return self.clone_model_id
        raise ValueError(f"Unknown model kind `{model_kind}`")

    def _loaded_model_ids(self) -> list[str]:
        with self._model_lock:
            return [self._model_id_for_kind(kind) for kind in self._models]

    def _is_model_loaded(self, model_kind: str) -> bool:
        with self._model_lock:
            return model_kind in self._models

    def _load_model_impl(self, model_kind: str) -> Any:
        from mlx_audio.tts.utils import load

        return load(self._model_id_for_kind(model_kind))

    def _run_speech_job(
        self,
        *,
        mode: str,
        model_kind: str,
        chunks: list[str],
        language: str,
        instruct: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        profile_name: str | None = None,
    ) -> dict[str, object]:
        if not chunks:
            raise ValueError("Speech request did not contain any playable text chunks")
        if not self._is_model_loaded(model_kind):
            raise RuntimeError(self.required_models_not_loaded_message(model_kinds=[model_kind]))

        job_id = str(uuid.uuid4())
        self._begin_speech_job(job_id)
        try:
            waveform, sample_rate = self._synthesize_chunks(
                model_kind=model_kind,
                chunks=chunks,
                language=language,
                instruct=instruct,
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
            self._play_audio(waveform, sample_rate)
        except Exception as exc:
            self._finish_speech_job(
                success=False,
                event_name="speech_job_failed",
                error_message=(
                    "Speech job "
                    f"`{job_id}` failed while using `{self._model_id_for_kind(model_kind)}`. "
                    f"Original error: {exc}"
                ),
            )
            raise

        completed_at = _utc_now()
        self._finish_speech_job(
            success=True,
            event_name="speech_job_completed",
            error_message=None,
        )
        return {
            "result": "success",
            "completed": True,
            "job_id": job_id,
            "mode": mode,
            "profile_name": profile_name,
            "chunk_count": len(chunks),
            "language": language,
            "playback_backend": self.playback_backend,
            "completed_at": completed_at,
            **self.status(),
        }

    def _synthesize_chunks(
        self,
        *,
        model_kind: str,
        chunks: list[str],
        language: str,
        instruct: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> tuple[np.ndarray, int]:
        with self._speech_lock:
            model = self._models.get(model_kind)
            if model is None:
                raise RuntimeError(
                    self.required_models_not_loaded_message(model_kinds=[model_kind])
                )

            audio_chunks: list[np.ndarray] = []
            sample_rate: int | None = None

            for chunk in chunks:
                results = list(
                    model.generate(
                        text=chunk,
                        lang_code=language,
                        instruct=instruct,
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        verbose=False,
                    )
                )
                if not results:
                    raise RuntimeError(
                        f"mlx-audio returned no waveform for chunk `{chunk[:40]}`. "
                        "The model likely rejected the prompt or terminated generation early."
                    )

                for result in results:
                    chunk_audio = np.asarray(result.audio, dtype=np.float32)
                    if chunk_audio.ndim != 1:
                        chunk_audio = chunk_audio.reshape(-1)
                    audio_chunks.append(chunk_audio)
                    result_sample_rate = int(getattr(result, "sample_rate", 24000))
                    sample_rate = sample_rate or result_sample_rate

            self._last_used_at[model_kind] = _utc_now()
            if sample_rate is None:
                raise RuntimeError(
                    "mlx-audio did not report a sample rate for the generated audio"
                )
            return np.concatenate(audio_chunks), sample_rate

    def _play_audio(self, waveform: np.ndarray, sample_rate: int) -> None:
        if self.playback_backend == "null":
            return

        try:
            sd.play(waveform, samplerate=sample_rate, blocking=True)
        except Exception as exc:
            raise RuntimeError(
                "Local audio playback failed while writing the generated waveform "
                "to the output device. "
                f"Backend=`{self.playback_backend}`, sample_rate={sample_rate}. "
                f"Original error: {exc}"
            ) from exc

    def _begin_speech_job(self, job_id: str) -> None:
        with self._status_lock:
            self._speech_in_progress = True
            self._speech_phase = "synthesizing"
            self._speech_current_job_id = job_id
            self._speech_last_error = None
        self._record_event("speech_job_started", job_id=job_id)

    def _finish_speech_job(
        self,
        *,
        success: bool,
        event_name: str,
        error_message: str | None,
    ) -> None:
        with self._status_lock:
            self._speech_in_progress = False
            self._speech_phase = "idle"
            self._speech_current_job_id = None
            self._speech_last_error = error_message
            if success:
                self._speech_jobs_completed += 1
            else:
                self._speech_jobs_failed += 1
        self._record_event(event_name, error=error_message)

    def _record_event(self, event_name: str, **fields: object) -> None:
        timestamp = _utc_now()
        event = {"event": event_name, "timestamp": timestamp, **fields}
        with self._status_lock:
            self._speech_last_event = event_name
            self._speech_last_event_at = timestamp
            self._recent_events.append(event)

    def _validated_reference_audio_path(self, raw_path: str) -> Path:
        path = Path(raw_path.strip()).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Reference audio file `{path}` does not exist on disk")
        if not path.is_file():
            raise ValueError(f"Reference audio path `{path}` is not a regular file")
        return path

    async def _get_profile(self, state_store: Any, name: str) -> SpeechProfile | None:
        payload = await state_store.get(name, collection=PROFILE_COLLECTION, default=None)
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Speech profile `{name}` has unreadable state-store data and cannot be loaded"
            )
        return SpeechProfile(**payload)

    async def _put_profile(self, state_store: Any, profile: SpeechProfile) -> None:
        await state_store.put(profile.name, asdict(profile), collection=PROFILE_COLLECTION)

    async def _list_profiles(self, state_store: Any) -> list[SpeechProfile]:
        profiles: list[SpeechProfile] = []
        metadata_dir = self.state_dir / "metadata"
        if not metadata_dir.exists():
            values = getattr(state_store, "values", None)
            if isinstance(values, dict):
                for (key, collection), payload in values.items():
                    if collection != PROFILE_COLLECTION or not isinstance(key, str):
                        continue
                    if isinstance(payload, dict):
                        profiles.append(SpeechProfile(**payload))
            return profiles

        for metadata_path in sorted(metadata_dir.glob("*.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            if metadata.get("collection") != PROFILE_COLLECTION:
                continue

            key = metadata.get("key")
            if not isinstance(key, str):
                continue

            profile = await self._get_profile(state_store, key)
            if profile is not None:
                profiles.append(profile)

        return profiles


# MARK: Environment Helpers

def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value such as true or false")


def _env_positive_int(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _env_state_dir(raw: str | None) -> Path:
    value = (raw or str(DEFAULT_STATE_DIR)).strip()
    if not value:
        raise ValueError("SPEAK_TO_USER_STATE_DIR must not be empty")
    return Path(value).expanduser().resolve()


def _env_playback_backend(raw: str) -> PlaybackBackend:
    normalized = raw.strip().lower()
    if normalized not in {"sounddevice", "null"}:
        raise ValueError("SPEAK_TO_USER_PLAYBACK_BACKEND must be `sounddevice` or `null`")
    return cast(PlaybackBackend, normalized)


def _normalize_profile_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise ValueError("Profile name must not be empty")
    if "/" in normalized or "\\" in normalized:
        raise ValueError("Profile name must not contain path separators")
    return normalized


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()
