from __future__ import annotations

# MARK: Static Usage Text

USAGE_GUIDE_TEXT = """speak-to-user usage guide

This server uses mlx-audio for synthesis.
Use speak_text for normal narrated replies.
Use speak_text_as_clone for one-off cloning from a real local reference clip.
Use generate_speech_profile for a reusable profile from real reference audio.
Use generate_speech_profile_from_voice_design for a reusable profile
from a short synthetic seed clip.
Use speak_with_profile for repeated playback once a saved profile exists.
Use load_model and unload_model to manage resident models explicitly.
Use set_startup_model to persist which model or models preload on server startup.

Agent defaults:
- prefer language=\"en\" unless the caller needs another language
- check tts_status before assuming speech is broken
- use the status and profiles state resources when you need a read-only snapshot
- load required models before retrying a model-gated tool
- remember playback is global and serial across clients
- this service is tuned for reply playback, not long-form narration
- prefer accurate reference_text when cloning from real audio
- prefer short seed text for generate_speech_profile_from_voice_design
  because it is not a long narration path
"""


# MARK: Prompt Implementations

def choose_speak_to_user_workflow_prompt() -> str:
    return (
        "Choose the smallest speak-to-user workflow that fits the job. "
        "Use speak_text for ordinary narration, "
        "speak_text_as_clone for one-off real-audio cloning, "
        "generate_speech_profile for reusable profiles from real reference audio, "
        "generate_speech_profile_from_voice_design for reusable profiles "
        "from a short synthetic seed, "
        "and speak_with_profile once a profile already exists. "
        "Use load_model or unload_model when you need explicit model residency control, "
        "and set_startup_model when you want startup preload behavior to persist. "
        "Check tts_status first if readiness or "
        "busy state is unclear. "
        "Use the read-only status and profiles resources when you need current service "
        "state without mutating anything. "
        "Treat this server as reply playback infrastructure built on mlx-audio, "
        "not a long-form narration path."
    )


def guide_speech_profile_workflow_prompt() -> str:
    return (
        "For speech profiles, prefer real reference audio plus accurate "
        "reference_text when available. "
        "Use generate_speech_profile_from_voice_design only when you need "
        "the server to synthesize a short seed clip for profile creation. "
        "Keep seed text short and focused, then use speak_with_profile for "
        "reply-sized playback after the profile exists. Recreate weak profiles "
        "from better source material instead "
        "of reusing poor seeds."
    )
