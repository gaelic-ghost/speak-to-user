from __future__ import annotations

# MARK: Imports

from collections import deque
from dataclasses import asdict, dataclass
import datetime as dt
import json
from pathlib import Path
import tempfile
import threading
from typing import Any, Literal, cast
import uuid

import numpy as np
import sounddevice as sd
import soundfile as sf

from app.text_chunking import chunk_text_for_tts


# MARK: Constants

VOICE_DESIGN_MODEL_KIND = "voice_design"
CLONE_MODEL_KIND = "clone"
PROFILE_COLLECTION = "speech_profiles"
STARTUP_MODEL_COLLECTION = "runtime_config"
STARTUP_MODEL_KEY = "startup_model_option"
DEFAULT_STARTUP_MODEL_OPTION = "all"
DEFAULT_TTS_CHUNK_MAX_CHARS = 160
DEFAULT_TTS_MAX_NEW_TOKENS = 4096
DEFAULT_TTS_TEMPERATURE = 0.9
DEFAULT_TTS_TOP_K = 50
DEFAULT_TTS_TOP_P = 1.0
DEFAULT_VOICE_DESIGN_MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
DEFAULT_CLONE_MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"
RECENT_EVENT_LIMIT = 25
VOICE_DESIGN_PROFILE_SEED_MAX_CHARS = 240


# MARK: Data Structures

PlaybackBackend = Literal["sounddevice", "null"]


@dataclass(slots=True)
class SerializedArray:
    dtype: str
    data: list[Any]


@dataclass(slots=True)
class StoredSpeechProfilePromptItem:
    ref_code: SerializedArray | None
    ref_spk_embedding: SerializedArray
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str | None = None


@dataclass(slots=True)
class SpeechProfile:
    name: str
    clone_model_id: str
    clone_mode: str
    created_at: str
    updated_at: str
    prompt_items: list[StoredSpeechProfilePromptItem]
    profile_source: str | None = None
    seed_text: str | None = None
    voice_description: str | None = None


# MARK: Runtime


class TTSRuntime:
    def __init__(
        self,
        *,
        voice_design_model_id: str,
        clone_model_id: str,
        profiles_dir: Path,
        playback_backend: PlaybackBackend = "sounddevice",
    ) -> None:
        self.voice_design_model_id = voice_design_model_id
        self.clone_model_id = clone_model_id
        self.profiles_dir = profiles_dir
        self.playback_backend = playback_backend

        self._models: dict[str, Any] = {}
        self._model_lock = threading.RLock()
        self._speech_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._recent_events: deque[dict[str, object]] = deque(maxlen=RECENT_EVENT_LIMIT)

        self._startup_model_option = DEFAULT_STARTUP_MODEL_OPTION
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

    # MARK: Lifecycle

    async def preload(self, *, state_store: Any) -> None:
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        stored_option = await state_store.get(
            STARTUP_MODEL_KEY,
            collection=STARTUP_MODEL_COLLECTION,
            default=DEFAULT_STARTUP_MODEL_OPTION,
        )
        option = stored_option if isinstance(stored_option, str) else DEFAULT_STARTUP_MODEL_OPTION
        self._startup_model_option = option

        for model_kind in self._model_kinds_for_option(option):
            self.load_model(model_kind=model_kind)

    def shutdown(self) -> None:
        with self._model_lock:
            self._models.clear()

    # MARK: Model Management

    def available_model_ids(self) -> list[str]:
        return [self.voice_design_model_id, self.clone_model_id]

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
        return [
            self._model_id_for_kind(model_kind)
            for model_kind in model_kinds
            if not self._is_model_loaded(model_kind)
        ]

    def required_models_not_loaded_message(self, *, model_kinds: list[str]) -> str:
        missing = self.missing_required_model_ids(model_kinds=model_kinds)
        return "Required TTS models are not loaded yet: " + ", ".join(missing)

    def load_model(self, *, model_kind: str) -> dict[str, object]:
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
        text: str,
        voice_description: str,
        language: str,
    ) -> dict[str, object]:
        normalized_voice_description = voice_description.strip()
        if not normalized_voice_description:
            raise ValueError("voice_description must not be empty")
        return self._run_speech_job(
            mode="voice_design",
            model_kind=VOICE_DESIGN_MODEL_KIND,
            text=text,
            language=language,
            instruct=normalized_voice_description,
        )

    def speak_text_as_clone(
        self,
        *,
        text: str,
        reference_audio_path: str,
        reference_text: str | None,
        language: str,
    ) -> dict[str, object]:
        reference_path = self._validated_reference_audio_path(reference_audio_path)
        return self._run_speech_job(
            mode="clone",
            model_kind=CLONE_MODEL_KIND,
            text=text,
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

        prompt_items = self._create_voice_clone_prompt_items(
            reference_audio_path=reference_audio_path,
            reference_text=reference_text,
        )
        now = _utc_now()
        normalized_reference_text = reference_text.strip() if reference_text else None
        profile = SpeechProfile(
            name=profile_name,
            clone_model_id=self.clone_model_id,
            clone_mode="reference_text" if normalized_reference_text else "x_vector_only",
            created_at=now,
            updated_at=now,
            prompt_items=prompt_items,
        )
        await self._put_profile(state_store, profile)
        self._record_event("speech_profile_created", name=profile_name)
        return {
            "result": "success",
            **self._speech_profile_metadata(profile),
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

        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("text must not be empty")
        if len(normalized_text) > VOICE_DESIGN_PROFILE_SEED_MAX_CHARS:
            raise ValueError(
                "text must be 240 characters or fewer for voice-designed profile seeds"
            )
        normalized_voice_description = voice_description.strip()
        if not normalized_voice_description:
            raise ValueError("voice_description must not be empty")

        waveform, sample_rate = self._synthesize_text(
            model_kind=VOICE_DESIGN_MODEL_KIND,
            text=normalized_text,
            language=language,
            instruct=normalized_voice_description,
        )

        temp_reference_audio_path = self._write_temp_reference_audio_file(
            waveform=waveform,
            sample_rate=sample_rate,
        )
        try:
            prompt_items = self._create_voice_clone_prompt_items(
                reference_audio_path=temp_reference_audio_path,
                reference_text=normalized_text,
            )
        finally:
            temp_path = Path(temp_reference_audio_path)
            if temp_path.exists():
                temp_path.unlink()

        now = _utc_now()
        profile = SpeechProfile(
            name=profile_name,
            clone_model_id=self.clone_model_id,
            clone_mode="reference_text",
            created_at=now,
            updated_at=now,
            prompt_items=prompt_items,
            profile_source="voice_design",
            seed_text=normalized_text,
            voice_description=normalized_voice_description,
        )
        await self._put_profile(state_store, profile)
        self._record_event("speech_profile_created_from_voice_design", name=profile_name)
        return {
            "result": "success",
            **self._speech_profile_metadata(profile),
            "seed_text_stored": True,
            "voice_description_stored": True,
        }

    async def list_speech_profiles(self, *, state_store: Any) -> dict[str, object]:
        profiles = await self._list_profiles(state_store)
        return {
            "result": "success",
            "profiles": [self._speech_profile_metadata(profile) for profile in profiles],
            "profile_count": len(profiles),
        }

    async def delete_speech_profile(self, *, state_store: Any, name: str) -> dict[str, object]:
        profile_name = _normalize_profile_name(name)
        profile = await self._get_profile(state_store, profile_name)
        if profile is None:
            raise ValueError(f"Speech profile `{profile_name}` does not exist")

        await state_store.delete(profile_name, collection=PROFILE_COLLECTION)
        self._record_event("speech_profile_deleted", name=profile_name)
        return {
            "result": "success",
            "deleted": True,
            "name": profile_name,
        }

    async def speak_with_profile(
        self,
        *,
        state_store: Any,
        name: str,
        text: str,
        language: str,
    ) -> dict[str, object]:
        profile_name = _normalize_profile_name(name)
        profile = await self._get_profile(state_store, profile_name)
        if profile is None:
            raise ValueError(f"Speech profile `{profile_name}` does not exist")
        if profile.clone_model_id != self.clone_model_id:
            raise ValueError(
                f"Speech profile `{profile_name}` targets `{profile.clone_model_id}`, "
                f"but the current clone model is `{self.clone_model_id}`"
            )
        return self._run_speech_job(
            mode="profile",
            model_kind=CLONE_MODEL_KIND,
            text=text,
            language=language,
            prompt_items=profile.prompt_items,
            profile_name=profile.name,
        )

    # MARK: Status

    def status(self) -> dict[str, object]:
        with self._status_lock:
            return {
                "ready": True,
                "profiles_dir": str(self.profiles_dir),
                "startup_model_option": self._startup_model_option,
                "startup_model_options": self.startup_model_options(),
                "loaded_model_ids": self._loaded_model_ids(),
                "voice_design_model_loaded": self._is_model_loaded(VOICE_DESIGN_MODEL_KIND),
                "voice_design_model_id": self.voice_design_model_id,
                "voice_design_last_loaded_at": self._last_loaded_at[VOICE_DESIGN_MODEL_KIND],
                "voice_design_last_used_at": self._last_used_at[VOICE_DESIGN_MODEL_KIND],
                "voice_design_last_error": self._last_model_error[VOICE_DESIGN_MODEL_KIND],
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
                "recent_events": list(self._recent_events),
            }

    # MARK: Internal Helpers

    def _model_kinds_for_option(self, option: str) -> list[str]:
        if option == "none":
            return []
        if option == "all":
            return [VOICE_DESIGN_MODEL_KIND, CLONE_MODEL_KIND]
        return [self.model_kind_for_id(option)]

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
        text: str,
        language: str,
        instruct: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        prompt_items: list[StoredSpeechProfilePromptItem] | None = None,
        profile_name: str | None = None,
    ) -> dict[str, object]:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("text must not be empty")
        if not self._is_model_loaded(model_kind):
            raise RuntimeError(self.required_models_not_loaded_message(model_kinds=[model_kind]))

        job_id = str(uuid.uuid4())
        self._begin_speech_job(job_id)
        try:
            waveform, sample_rate = self._synthesize_text(
                model_kind=model_kind,
                text=normalized_text,
                language=language,
                instruct=instruct,
                ref_audio=ref_audio,
                ref_text=ref_text,
                prompt_items=prompt_items,
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
            "language": language,
            "completed_at": completed_at,
            **self.status(),
        }

    def _synthesize_text(
        self,
        *,
        model_kind: str,
        text: str,
        language: str,
        instruct: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        prompt_items: list[StoredSpeechProfilePromptItem] | None = None,
    ) -> tuple[np.ndarray, int]:
        return self._synthesize_chunks(
            model_kind=model_kind,
            chunks=chunk_text_for_tts(text, max_chars=DEFAULT_TTS_CHUNK_MAX_CHARS),
            language=language,
            instruct=instruct,
            ref_audio=ref_audio,
            ref_text=ref_text,
            prompt_items=prompt_items,
        )

    def _synthesize_chunks(
        self,
        *,
        model_kind: str,
        chunks: list[str],
        language: str,
        instruct: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        prompt_items: list[StoredSpeechProfilePromptItem] | None = None,
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
                if prompt_items is None:
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
                else:
                    chunk_audio, result_sample_rate = self._synthesize_with_prompt_items(
                        model=model,
                        text=chunk,
                        language=language,
                        prompt_items=prompt_items,
                    )
                    audio_chunks.append(chunk_audio)
                    sample_rate = sample_rate or result_sample_rate

            self._last_used_at[model_kind] = _utc_now()
            if sample_rate is None:
                raise RuntimeError(
                    "mlx-audio did not report a sample rate for the generated audio"
                )
            return np.concatenate(audio_chunks), sample_rate

    def _synthesize_with_prompt_items(
        self,
        *,
        model: Any,
        text: str,
        language: str,
        prompt_items: list[StoredSpeechProfilePromptItem],
    ) -> tuple[np.ndarray, int]:
        if not prompt_items:
            raise RuntimeError("speech profile does not contain any reusable clone artifacts")

        prompt_item = prompt_items[0]
        if prompt_item.icl_mode:
            return self._synthesize_with_icl_prompt_item(
                model=model,
                text=text,
                language=language,
                prompt_item=prompt_item,
            )
        return self._synthesize_with_x_vector_prompt_item(
            model=model,
            text=text,
            language=language,
            prompt_item=prompt_item,
        )

    def _synthesize_with_x_vector_prompt_item(
        self,
        *,
        model: Any,
        text: str,
        language: str,
        prompt_item: StoredSpeechProfilePromptItem,
    ) -> tuple[np.ndarray, int]:
        import mlx.core as mx

        speaker_embedding = self._deserialize_array(prompt_item.ref_spk_embedding)
        input_embeds, trailing_text_hidden, tts_pad_embed = (
            self._prepare_generation_inputs_with_speaker_embedding(
                model=model,
                text=text,
                language=language,
                speaker_embedding=speaker_embedding,
            )
        )
        waveform = self._run_prefilled_generation_loop(
            model=model,
            input_embeds=input_embeds,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            repetition_penalty=1.05,
        )
        waveform_np = np.asarray(waveform, dtype=np.float32).reshape(-1)
        mx.clear_cache()
        return waveform_np, int(model.sample_rate)

    def _synthesize_with_icl_prompt_item(
        self,
        *,
        model: Any,
        text: str,
        language: str,
        prompt_item: StoredSpeechProfilePromptItem,
    ) -> tuple[np.ndarray, int]:
        import mlx.core as mx

        if prompt_item.ref_code is None or not prompt_item.ref_text:
            raise RuntimeError(
                "speech profile is missing the stored reference-text clone artifact payload"
            )

        speaker_embedding = self._deserialize_array(prompt_item.ref_spk_embedding)
        ref_codes = self._deserialize_array(prompt_item.ref_code)
        input_embeds, trailing_text_hidden, tts_pad_embed, prepared_ref_codes = (
            self._prepare_icl_generation_inputs_from_prompt_item(
                model=model,
                text=text,
                language=language,
                ref_text=prompt_item.ref_text,
                ref_codes=ref_codes,
                speaker_embedding=speaker_embedding,
            )
        )
        waveform = self._run_prefilled_generation_loop(
            model=model,
            input_embeds=input_embeds,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            repetition_penalty=1.5,
            ref_codes=prepared_ref_codes,
        )
        waveform_np = np.asarray(waveform, dtype=np.float32).reshape(-1)
        mx.clear_cache()
        return waveform_np, int(model.sample_rate)

    def _run_prefilled_generation_loop(
        self,
        *,
        model: Any,
        input_embeds: Any,
        trailing_text_hidden: Any,
        tts_pad_embed: Any,
        repetition_penalty: float,
        ref_codes: Any | None = None,
    ) -> Any:
        import mlx.core as mx

        if model.tokenizer is None:
            raise RuntimeError("mlx-audio clone model tokenizer is unavailable")
        if model.speech_tokenizer is None:
            raise RuntimeError("mlx-audio clone model speech tokenizer is unavailable")

        cache = model.talker.make_cache()
        code_cache = model.talker.code_predictor.make_cache()
        generated_codes: list[Any] = []
        generated_token_ids: list[int] = []
        config = model.config.talker_config
        eos_token_id = config.codec_eos_token_id
        suppress_tokens = [
            token_id
            for token_id in range(config.vocab_size - 1024, config.vocab_size)
            if token_id != eos_token_id
        ]
        trailing_idx = 0

        for step in range(DEFAULT_TTS_MAX_NEW_TOKENS):
            logits, hidden = model.talker(input_embeds, cache=cache)
            next_token = model._sample_token(
                logits,
                temperature=DEFAULT_TTS_TEMPERATURE,
                top_k=DEFAULT_TTS_TOP_K,
                top_p=DEFAULT_TTS_TOP_P,
                repetition_penalty=repetition_penalty,
                generated_tokens=(generated_token_ids if generated_token_ids else None),
                suppress_tokens=suppress_tokens,
                eos_token_id=eos_token_id,
            )
            is_eos = next_token[0, 0] == eos_token_id

            code_tokens = [next_token]
            code_hidden = hidden[:, -1:, :]

            for code_cache_entry in code_cache:
                code_cache_entry.keys = None
                code_cache_entry.values = None
                code_cache_entry.offset = 0

            for code_idx in range(config.num_code_groups - 1):
                if code_idx == 0:
                    code_0_embed = model.talker.get_input_embeddings()(next_token)
                    code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
                else:
                    code_embed = model.talker.code_predictor.codec_embedding[code_idx - 1](
                        code_tokens[-1]
                    )
                    code_input = code_embed

                code_logits, code_cache, _ = model.talker.code_predictor(
                    code_input,
                    cache=code_cache,
                    generation_step=code_idx,
                )
                next_code = model._sample_token(
                    code_logits,
                    temperature=DEFAULT_TTS_TEMPERATURE,
                    top_k=DEFAULT_TTS_TOP_K,
                    top_p=DEFAULT_TTS_TOP_P,
                )
                code_tokens.append(next_code)

            all_codes = mx.concatenate(code_tokens, axis=1)

            if trailing_idx < trailing_text_hidden.shape[1]:
                text_embed = trailing_text_hidden[:, trailing_idx : trailing_idx + 1, :]
                trailing_idx += 1
            else:
                text_embed = tts_pad_embed

            codec_embed = model.talker.get_input_embeddings()(next_token)
            for code_index, code in enumerate(code_tokens[1:]):
                codec_embed = codec_embed + model.talker.code_predictor.codec_embedding[
                    code_index
                ](code)

            input_embeds = text_embed + codec_embed
            mx.eval(input_embeds, is_eos)
            if is_eos.item():
                break

            generated_token_ids.append(int(next_token[0, 0]))
            generated_codes.append(all_codes)
            if step > 0 and step % 50 == 0:
                mx.clear_cache()

        if not generated_codes:
            raise RuntimeError(
                "mlx-audio generated no codec frames from the stored clone artifacts"
            )

        codes = mx.stack(generated_codes, axis=1)
        if ref_codes is None:
            audio, audio_lengths = model.speech_tokenizer.decode(codes)
            audio = audio[0]
            valid_len = int(audio_lengths[0])
            if valid_len > 0 and valid_len < audio.shape[0]:
                audio = audio[:valid_len]
            mx.eval(audio)
            return audio

        ref_codes_t = mx.transpose(ref_codes, (0, 2, 1))
        full_codes = mx.concatenate([ref_codes_t, codes], axis=1)
        ref_len = ref_codes.shape[2]
        total_len = full_codes.shape[1]

        audio, audio_lengths = model.speech_tokenizer.decode(full_codes)
        audio = audio[0]
        valid_len = int(audio_lengths[0])
        if valid_len > 0 and valid_len < audio.shape[0]:
            audio = audio[:valid_len]

        cut = int(ref_len / max(total_len, 1) * audio.shape[0])
        if cut > 0 and cut < audio.shape[0]:
            audio = audio[cut:]

        mx.eval(audio)
        return audio

    def _prepare_generation_inputs_with_speaker_embedding(
        self,
        *,
        model: Any,
        text: str,
        language: str,
        speaker_embedding: Any,
    ) -> tuple[Any, Any, Any]:
        import mlx.core as mx

        if model.tokenizer is None:
            raise RuntimeError("mlx-audio clone model tokenizer is unavailable")

        config = model.config.talker_config
        chat_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = mx.array(model.tokenizer.encode(chat_text))[None, :]
        text_embed = model.talker.text_projection(model.talker.get_text_embeddings()(input_ids))

        tts_tokens = mx.array(
            [
                [
                    model.config.tts_bos_token_id,
                    model.config.tts_eos_token_id,
                    model.config.tts_pad_token_id,
                ]
            ]
        )
        tts_embeds = model.talker.text_projection(model.talker.get_text_embeddings()(tts_tokens))
        tts_bos_embed = tts_embeds[:, 0:1, :]
        tts_eos_embed = tts_embeds[:, 1:2, :]
        tts_pad_embed = tts_embeds[:, 2:3, :]

        language_id = None
        if language.lower() != "auto" and config.codec_language_id:
            if language.lower() in config.codec_language_id:
                language_id = config.codec_language_id[language.lower()]

        if language_id is None:
            codec_prefill = [
                config.codec_nothink_id,
                config.codec_think_bos_id,
                config.codec_think_eos_id,
            ]
        else:
            codec_prefill = [
                config.codec_think_id,
                config.codec_think_bos_id,
                language_id,
                config.codec_think_eos_id,
            ]

        codec_embed = model.talker.get_input_embeddings()(mx.array([codec_prefill]))
        codec_embed_suffix = model.talker.get_input_embeddings()(
            mx.array([[config.codec_pad_id, config.codec_bos_id]])
        )
        normalized_speaker_embedding = speaker_embedding.reshape(1, 1, -1)
        codec_embed = mx.concatenate(
            [codec_embed, normalized_speaker_embedding, codec_embed_suffix],
            axis=1,
        )

        role_embed = text_embed[:, :3, :]
        pad_count = codec_embed.shape[1] - 2
        pad_embeds = mx.broadcast_to(tts_pad_embed, (1, pad_count, tts_pad_embed.shape[-1]))
        combined_embed = mx.concatenate([pad_embeds, tts_bos_embed], axis=1)
        combined_embed = combined_embed + codec_embed[:, :-1, :]

        input_embeds = mx.concatenate([role_embed, combined_embed], axis=1)
        first_text_embed = text_embed[:, 3:4, :] + codec_embed[:, -1:, :]
        input_embeds = mx.concatenate([input_embeds, first_text_embed], axis=1)
        trailing_text_hidden = mx.concatenate([text_embed[:, 4:-5, :], tts_eos_embed], axis=1)
        return input_embeds, trailing_text_hidden, tts_pad_embed

    def _prepare_icl_generation_inputs_from_prompt_item(
        self,
        *,
        model: Any,
        text: str,
        language: str,
        ref_text: str,
        ref_codes: Any,
        speaker_embedding: Any,
    ) -> tuple[Any, Any, Any, Any]:
        import mlx.core as mx

        if model.tokenizer is None:
            raise RuntimeError("mlx-audio clone model tokenizer is unavailable")

        config = model.config.talker_config
        normalized_ref_codes = ref_codes
        if normalized_ref_codes.ndim == 2:
            normalized_ref_codes = normalized_ref_codes[None, :, :]
        if normalized_ref_codes.ndim != 3:
            raise RuntimeError("stored reference codes have an unexpected shape")
        ref_chat = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
        ref_ids = mx.array(model.tokenizer.encode(ref_chat))[None, :]
        ref_text_ids = ref_ids[:, 3:-2]

        target_chat = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        target_ids = mx.array(model.tokenizer.encode(target_chat))[None, :]
        text_ids = target_ids[:, 3:-5]

        tts_tokens = mx.array(
            [
                [
                    model.config.tts_bos_token_id,
                    model.config.tts_eos_token_id,
                    model.config.tts_pad_token_id,
                ]
            ]
        )
        tts_embeds = model.talker.text_projection(model.talker.get_text_embeddings()(tts_tokens))
        tts_bos_embed = tts_embeds[:, 0:1, :]
        tts_eos_embed = tts_embeds[:, 1:2, :]
        tts_pad_embed = tts_embeds[:, 2:3, :]

        combined_text_ids = mx.concatenate([ref_text_ids, text_ids], axis=1)
        text_embed = model.talker.text_projection(
            model.talker.get_text_embeddings()(combined_text_ids)
        )
        text_embed = mx.concatenate([text_embed, tts_eos_embed], axis=1)
        text_lens = text_embed.shape[1]

        first_cb_codes = normalized_ref_codes[:, 0, :]
        ref_codec_embed = model.talker.get_input_embeddings()(first_cb_codes)
        for index in range(config.num_code_groups - 1):
            cb_codes = normalized_ref_codes[:, index + 1, :]
            ref_codec_embed = (
                ref_codec_embed
                + model.talker.code_predictor.codec_embedding[index](cb_codes)
            )

        codec_bos_embed = model.talker.get_input_embeddings()(mx.array([[config.codec_bos_id]]))
        codec_embed_icl = mx.concatenate([codec_bos_embed, ref_codec_embed], axis=1)
        codec_lens = codec_embed_icl.shape[1]

        codec_pad_embed = model.talker.get_input_embeddings()(mx.array([[config.codec_pad_id]]))
        text_with_codec_pad = text_embed + mx.broadcast_to(
            codec_pad_embed, (1, text_lens, codec_pad_embed.shape[-1])
        )
        codec_with_text_pad = codec_embed_icl + mx.broadcast_to(
            tts_pad_embed, (1, codec_lens, tts_pad_embed.shape[-1])
        )
        icl_input_embed = mx.concatenate([text_with_codec_pad, codec_with_text_pad], axis=1)
        trailing_text_hidden = tts_pad_embed

        language_id = None
        if language.lower() != "auto" and config.codec_language_id:
            if language.lower() in config.codec_language_id:
                language_id = config.codec_language_id[language.lower()]

        if language_id is None:
            codec_prefill = [
                config.codec_nothink_id,
                config.codec_think_bos_id,
                config.codec_think_eos_id,
            ]
        else:
            codec_prefill = [
                config.codec_think_id,
                config.codec_think_bos_id,
                language_id,
                config.codec_think_eos_id,
            ]

        codec_prefix_embed = model.talker.get_input_embeddings()(mx.array([codec_prefill]))
        codec_prefix_suffix = model.talker.get_input_embeddings()(
            mx.array([[config.codec_pad_id, config.codec_bos_id]])
        )
        normalized_speaker_embedding = speaker_embedding.reshape(1, 1, -1)
        codec_prefix_embed = mx.concatenate(
            [codec_prefix_embed, normalized_speaker_embedding, codec_prefix_suffix],
            axis=1,
        )

        role_embed = model.talker.text_projection(
            model.talker.get_text_embeddings()(target_ids[:, :3])
        )
        pad_count = codec_prefix_embed.shape[1] - 2
        pad_embeds = mx.broadcast_to(tts_pad_embed, (1, pad_count, tts_pad_embed.shape[-1]))
        combined_prefix = mx.concatenate([pad_embeds, tts_bos_embed], axis=1)
        combined_prefix = combined_prefix + codec_prefix_embed[:, :-1, :]

        input_embeds = mx.concatenate([role_embed, combined_prefix, icl_input_embed], axis=1)
        return input_embeds, trailing_text_hidden, tts_pad_embed, normalized_ref_codes

    def _create_voice_clone_prompt_items(
        self,
        *,
        reference_audio_path: str,
        reference_text: str | None,
    ) -> list[StoredSpeechProfilePromptItem]:
        import mlx.core as mx
        from mlx_audio.utils import load_audio

        reference_path = self._validated_reference_audio_path(reference_audio_path)
        clone_model = self._require_loaded_model(CLONE_MODEL_KIND)
        if clone_model.tokenizer is None:
            raise RuntimeError("mlx-audio clone model tokenizer is unavailable")
        if clone_model.speech_tokenizer is None:
            raise RuntimeError("mlx-audio clone model speech tokenizer is unavailable")

        reference_audio = load_audio(str(reference_path), sample_rate=24000)
        speaker_embedding = clone_model.extract_speaker_embedding(reference_audio)
        mx.eval(speaker_embedding)

        normalized_reference_text = reference_text.strip() if reference_text else None
        if normalized_reference_text:
            audio_for_encoder = reference_audio
            if audio_for_encoder.ndim == 1:
                audio_for_encoder = audio_for_encoder[None, None, :]
            elif audio_for_encoder.ndim == 2:
                audio_for_encoder = audio_for_encoder[None, :]
            ref_code = clone_model.speech_tokenizer.encode(audio_for_encoder)
            mx.eval(ref_code)
            return [
                StoredSpeechProfilePromptItem(
                    ref_code=self._serialize_array(ref_code),
                    ref_spk_embedding=self._serialize_array(speaker_embedding),
                    x_vector_only_mode=False,
                    icl_mode=True,
                    ref_text=normalized_reference_text,
                )
            ]

        return [
            StoredSpeechProfilePromptItem(
                ref_code=None,
                ref_spk_embedding=self._serialize_array(speaker_embedding),
                x_vector_only_mode=True,
                icl_mode=False,
                ref_text=None,
            )
        ]

    def _write_temp_reference_audio_file(
        self,
        *,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        sf.write(temp_path, waveform, sample_rate)
        return temp_path

    def _require_loaded_model(self, model_kind: str) -> Any:
        model = self._models.get(model_kind)
        if model is None:
            raise RuntimeError(self.required_models_not_loaded_message(model_kinds=[model_kind]))
        return model

    def _serialize_array(self, value: Any) -> SerializedArray:
        array = np.asarray(value)
        return SerializedArray(dtype=str(array.dtype), data=array.tolist())

    def _deserialize_array(self, value: SerializedArray) -> Any:
        import mlx.core as mx

        return mx.array(np.asarray(value.data, dtype=np.dtype(value.dtype)))

    def _speech_profile_metadata(self, profile: SpeechProfile) -> dict[str, object]:
        return {
            "name": profile.name,
            "clone_model_id": profile.clone_model_id,
            "clone_mode": profile.clone_mode,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
            "reference_text_included": profile.clone_mode == "reference_text",
            "profile_source": profile.profile_source,
        }

    def _play_audio(self, waveform: np.ndarray, sample_rate: int) -> None:
        if self.playback_backend == "null":
            return

        try:
            sd.play(waveform, samplerate=sample_rate, blocking=True)
        except Exception as exc:
            raise RuntimeError(
                "Local audio playback failed while writing the generated waveform "
                "to the output device. "
                f"Sample_rate={sample_rate}. Original error: {exc}"
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

    def _metadata_dir(self) -> Path:
        return self.profiles_dir.parent / "metadata"

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
        return _speech_profile_from_payload(payload)

    async def _put_profile(self, state_store: Any, profile: SpeechProfile) -> None:
        await state_store.put(profile.name, asdict(profile), collection=PROFILE_COLLECTION)

    async def _list_profiles(self, state_store: Any) -> list[SpeechProfile]:
        profiles: list[SpeechProfile] = []
        metadata_dir = self._metadata_dir()
        if not metadata_dir.exists():
            values = getattr(state_store, "values", None)
            if isinstance(values, dict):
                for (key, collection), payload in values.items():
                    if collection != PROFILE_COLLECTION or not isinstance(key, str):
                        continue
                    if isinstance(payload, dict):
                        profiles.append(_speech_profile_from_payload(payload))
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


# MARK: Helpers


def _normalize_profile_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise ValueError("Profile name must not be empty")
    if "/" in normalized or "\\" in normalized:
        raise ValueError("Profile name must not contain path separators")
    return normalized


def _speech_profile_from_payload(payload: dict[str, Any]) -> SpeechProfile:
    prompt_items_payload = payload.get("prompt_items")
    if not isinstance(prompt_items_payload, list):
        raise RuntimeError("Speech profile payload is missing reusable clone artifacts")

    prompt_items = [_prompt_item_from_payload(item) for item in prompt_items_payload]
    name = payload.get("name")
    clone_model_id = payload.get("clone_model_id")
    clone_mode = payload.get("clone_mode")
    created_at = payload.get("created_at")
    updated_at = payload.get("updated_at")
    if not all(
        isinstance(value, str)
        for value in [name, clone_model_id, clone_mode, created_at, updated_at]
    ):
        raise RuntimeError("Speech profile payload is missing required metadata fields")

    profile_source = payload.get("profile_source")
    seed_text = payload.get("seed_text")
    voice_description = payload.get("voice_description")
    if profile_source is not None and not isinstance(profile_source, str):
        raise RuntimeError("Speech profile payload has an invalid `profile_source` field")
    if seed_text is not None and not isinstance(seed_text, str):
        raise RuntimeError("Speech profile payload has an invalid `seed_text` field")
    if voice_description is not None and not isinstance(voice_description, str):
        raise RuntimeError("Speech profile payload has an invalid `voice_description` field")

    return SpeechProfile(
        name=cast(str, name),
        clone_model_id=cast(str, clone_model_id),
        clone_mode=cast(str, clone_mode),
        created_at=cast(str, created_at),
        updated_at=cast(str, updated_at),
        prompt_items=prompt_items,
        profile_source=profile_source,
        seed_text=seed_text,
        voice_description=voice_description,
    )


def _prompt_item_from_payload(payload: Any) -> StoredSpeechProfilePromptItem:
    if not isinstance(payload, dict):
        raise RuntimeError("Speech profile prompt artifact payload is unreadable")

    ref_spk_embedding_payload = payload.get("ref_spk_embedding")
    if not isinstance(ref_spk_embedding_payload, dict):
        raise RuntimeError("Speech profile prompt artifact is missing `ref_spk_embedding`")

    ref_code_payload = payload.get("ref_code")
    ref_code = (
        _serialized_array_from_payload(ref_code_payload)
        if isinstance(ref_code_payload, dict)
        else None
    )
    ref_text = payload.get("ref_text")
    if ref_text is not None and not isinstance(ref_text, str):
        raise RuntimeError("Speech profile prompt artifact has an invalid `ref_text` field")

    return StoredSpeechProfilePromptItem(
        ref_code=ref_code,
        ref_spk_embedding=_serialized_array_from_payload(ref_spk_embedding_payload),
        x_vector_only_mode=bool(payload.get("x_vector_only_mode")),
        icl_mode=bool(payload.get("icl_mode")),
        ref_text=cast(str | None, ref_text),
    )


def _serialized_array_from_payload(payload: dict[str, Any]) -> SerializedArray:
    dtype = payload.get("dtype")
    data = payload.get("data")
    if not isinstance(dtype, str) or not isinstance(data, list):
        raise RuntimeError("Stored clone artifact tensor payload is unreadable")
    return SerializedArray(dtype=dtype, data=data)


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()
