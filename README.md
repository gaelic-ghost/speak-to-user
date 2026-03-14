# speak-to-gale

`speak-to-gale` is a local [FastMCP](https://gofastmcp.com/) server that gives coding agents a reliable text-to-speech path on Gale's Mac. It wraps the [`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) model, manages model lifecycle in-process, writes generated audio to disk, and can optionally play the result locally.

The server is designed for agent workflows where spoken feedback should be available by default, but where runtime state still needs to be inspectable and controllable through MCP tools.

## Features

- FastMCP tool server with a small, focused TTS tool surface
- Background model preload on startup instead of blocking server boot
- Automatic idle unloading to reclaim memory after inactivity
- Runtime status introspection, including preload timestamps and the last error
- Local audio generation in `wav` or `flac`
- Local playback through the host machine for spoken replies

## Requirements

- macOS
- [uv](https://docs.astral.sh/uv/)
- Python 3.12
- Enough local RAM / VRAM for the Qwen TTS model

## Install

```bash
uv sync
```

## Run The Server

```bash
uv run python app/server.py
```

When the server starts, it creates a `TTSRuntime`, starts an idle-unload watchdog, and kicks off model preload in a background thread. Startup does not fail just because preload is still running; use the `tts_status` tool to confirm readiness and inspect any preload error.

## MCP Tools

The server exposes the following tools:

- `health`: lightweight smoke check with a UTC timestamp
- `tts_status`: reports runtime readiness, device, model state, preload timestamps, output directory, and `last_error`
- `load_model`: loads the TTS model into memory on demand
- `unload_model`: unloads the model manually
- `set_idle_unload_timeout`: updates the automatic idle-unload threshold in seconds
- `generate_audio`: synthesizes one local audio file and returns metadata
- `speak_text`: synthesizes audio and plays it locally on the machine

## Configuration

Environment variables:

- `SPEAK_TO_GALE_MODEL_ID`: model ID to load
  Default: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `SPEAK_TO_GALE_IDLE_UNLOAD_SECONDS`: idle timeout before automatic unload
  Default: `1200`
- `SPEAK_TO_GALE_OUTPUT_DIR`: directory for generated audio files
  Default: `generated-audio`
- `SPEAK_TO_GALE_DEVICE`: device preference
  Allowed values: `auto`, `mps`, `cpu`
  Default: `auto`

## Development

Project layout:

- `app/server.py`: FastMCP entrypoint and tool registration
- `app/runtime.py`: model lifecycle, preload, idle unload, generation, and playback
- `app/tools.py`: tool adapters that translate runtime operations into MCP payloads
- `tests/test_server.py`: server and tool behavior tests
- `tests/test_runtime.py`: runtime lifecycle and generation tests

Validation commands:

```bash
uv run pytest
uv run ruff check .
uv run mypy .
```

## Notes

- Generated audio is written under the configured output directory, which defaults to `generated-audio/` in the repository root.
- `speak_text` is intended for local use on Gale's machine; it plays audio on the host instead of returning a remote-streaming session.
- If preload fails, the error is recorded in runtime state and surfaced through `tts_status` rather than crashing the server process.
