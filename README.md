# speak-to-user

[![Coverage](https://img.shields.io/badge/coverage-65%25-yellow)](./.coverage)

`speak-to-user` is a local [FastMCP](https://gofastmcp.com/) server that turns agent replies into spoken audio on Gale's Mac.

This rewrite uses [`mlx-audio`](https://github.com/Blaizzy/mlx-audio) as the synthesis runtime instead of the older `qwen-tts` stack. The repo stays intentionally small: two resident model lanes, direct playback, and reusable profile artifacts stored in FastMCP state.

## Runtime Model Roles

- `speak_text` uses a voice-design model for normal narrated replies driven by `voice_description`.
- `speak_text_as_clone` uses a clone-capable model for one-off reference-audio cloning.
- `generate_speech_profile` stores a reusable clone-conditioning artifact for repeated playback.
- `generate_speech_profile_from_voice_design` creates a short synthetic seed clip first, then stores a reusable clone-conditioning artifact derived from that clip.
- `speak_with_profile` reuses a saved profile without rebuilding clone conditioning from the original clip on every request.

Default model ids:

- voice design: `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`
- clone: `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit`

Both are configurable through environment variables.

## MCP Surface

Tools:

- `health`
- `tts_status`
- `load_model`
- `unload_model`
- `set_startup_model`
- `speak_text`
- `speak_text_as_clone`
- `generate_speech_profile`
- `generate_speech_profile_from_voice_design`
- `list_speech_profiles`
- `delete_speech_profile`
- `speak_with_profile`

Prompts:

- `choose_speak_to_user_workflow`
- `guide_speech_profile_workflow`

Resources:

- `guide://speak-to-user/usage`
- `state://speak-to-user/status`
- `state://speak-to-user/profiles`

## Behavior Notes

- Playback is global and serial across connected clients.
- `speak_text`, `speak_text_as_clone`, and `speak_with_profile` chunk long text before synthesis.
- Profiles are persisted as reusable clone-conditioning artifacts in the configured state store.
- The runtime keeps model loading explicit. If a required model is missing, the MCP tool can ask whether it should be loaded before retrying.
- Local playback uses `sounddevice`.

## Requirements

- macOS on Apple Silicon
- Python `3.12`
- [`uv`](https://docs.astral.sh/uv/)
- enough unified memory for the configured `mlx-audio` models

## Install

```bash
uv sync
```

## Run

Stable service:

```bash
uv run python app/server.py
```

Dev service on a different port:

```bash
SPEAK_TO_USER_PORT=8766 uv run python app/server.py
```

By default the MCP endpoint is served at `http://127.0.0.1:8765/mcp`.

## Configuration

- `SPEAK_TO_USER_VOICE_DESIGN_MODEL_ID`
  Default: `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`
- `SPEAK_TO_USER_CLONE_MODEL_ID`
  Default: `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit`
- `SPEAK_TO_USER_HOST`
  Default: `127.0.0.1`
- `SPEAK_TO_USER_PORT`
  Default: `8765`
- `SPEAK_TO_USER_MCP_PATH`
  Default: `/mcp`
- `SPEAK_TO_USER_STATE_DIR`
  Default: `~/.local/gaelic-ghost/speak-to-user/profiles`

## Validation

```bash
uv run pytest
uv run pytest --cov=app --cov-report=term-missing
uv run ruff check .
uv run mypy .
```
