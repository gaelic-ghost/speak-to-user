# speak-to-user

`speak-to-user` is a local [FastMCP](https://gofastmcp.com/) server with one job: take text jobs and speak them through the host machine.

It uses a resident voice-design model for normal TTS and can also keep a separate clone model resident for reference-audio voice cloning:

- [`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
- [`Qwen/Qwen3-TTS-12Hz-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)

The intended deployment mode is one long-lived local Streamable HTTP MCP service so multiple Codex clients can share the same resident TTS model instead of launching separate stdio subprocesses.

## Tools

- `health`: simple smoke-test response.
- `tts_status`: reports queue status, observability events, and separate status for the voice-design and clone models.
- `speak_text`: enqueues one full text job for the voice-design model.
- `speak_text_as_clone`: enqueues one full text job for the clone model using a local reference audio file and optional reference text.

There is no manual load tool, no unload tool, no idle auto-unload, no detached helper process, and no file-generation path.

## Requirements

- macOS
- [uv](https://docs.astral.sh/uv/)
- Python 3.12
- enough RAM for the Qwen TTS model

## Install

```bash
uv sync
```

## Run

```bash
uv run python app/server.py
```

By default, the entrypoint serves MCP over HTTP at `http://127.0.0.1:8765/mcp`.

Server startup blocks until all enabled models are loaded. After that, both `speak_text` and `speak_text_as_clone` push one full text job into one in-process FIFO queue. Every request is chunked before playback, the worker synthesizes ahead into a bounded waveform queue, opens playback only after preroll, and then keeps generating and writing later chunks in order while the same output stream stays open. During active work, `tts_status` exposes whether the runtime is still synthesizing audio or has reached device playback, plus which synthesis mode is active.

For the dev checkout, prefer a different port so it does not collide with the stable service:

```bash
SPEAK_TO_USER_PORT=8766 uv run python app/server.py
```

## Clone Inputs

`speak_text_as_clone` accepts:

- `text`
- `reference_audio_path`
- optional `reference_text`
- optional `language`

Clone inference uses the installed `qwen_tts` clone API in two modes:

- without `reference_text`: `x_vector_only_mode=True`
- with `reference_text`: `x_vector_only_mode=False`

The reference audio must be a readable local file. WAV or FLAC is the safest choice.

## Configuration

- `SPEAK_TO_USER_ENABLE_VOICE_DESIGN_MODEL`
  Default: `true`
- `SPEAK_TO_USER_ENABLE_CLONE_MODEL`
  Default: `true`
- `SPEAK_TO_USER_VOICE_DESIGN_MODEL_ID`
  Default: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `SPEAK_TO_USER_CLONE_MODEL_ID`
  Default: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- `SPEAK_TO_USER_DEVICE`
  Allowed values: `auto`, `mps`, `cpu`
  Default: `auto`
- `SPEAK_TO_USER_TORCH_DTYPE`
  Optional values: `float16`, `bfloat16`, `float32`
  Default: unset
- `SPEAK_TO_USER_HOST`
  Default: `127.0.0.1`
- `SPEAK_TO_USER_PORT`
  Default: `8765`
- `SPEAK_TO_USER_MCP_PATH`
  Default: `/mcp`
- `SPEAK_TO_USER_PLAYBACK_PREROLL_SECONDS`
  Default: `3.0`
- `SPEAK_TO_USER_PLAYBACK_PREROLL_CHUNKS`
  Default: `2`
- `SPEAK_TO_USER_PLAYBACK_WAVEFORM_QUEUE_MAXSIZE`
  Default: `16`
- `SPEAK_TO_USER_OUTPUT_STREAM_LATENCY`
  Allowed values: `low`, `high`, or a positive number
  Default: `high`
- `SPEAK_TO_USER_LOG_LEVEL`
  Allowed values: `minimal`, `info`, `debug`
  Default: `info`

Runtime language inputs accept either full language names understood by the model, short codes like `en`, or common locale variants like `en-US` and `pt_BR`.

## LaunchAgents

This repo includes LaunchAgent plists in [launchd/com.galew.speak-to-user.stable.plist](/Users/galew/Workspace/speak-to-user/launchd/com.galew.speak-to-user.stable.plist) and [launchd/com.galew.speak-to-user.dev.plist](/Users/galew/Workspace/speak-to-user/launchd/com.galew.speak-to-user.dev.plist).

They call [scripts/run_service.sh](/Users/galew/Workspace/speak-to-user/scripts/run_service.sh), which sets a Homebrew-friendly `PATH` so tools like `sox` resolve correctly under `launchd`.

- Stable service: `http://127.0.0.1:8765/mcp`
- Dev service: `http://127.0.0.1:8766/mcp`

Runtime observability is split between the LaunchAgent stderr logs and `tts_status`.
At the default `info` log level, the runtime emits structured JSON events for job queueing, synthesis, preroll, stream open/close, chunk playback, completion, and failure. `tts_status` also includes a bounded in-memory `recent_events` history plus the latest event name and timestamps for the current job, chunk, and phase.

Install or refresh the LaunchAgents:

```bash
cp launchd/com.galew.speak-to-user.stable.plist ~/Library/LaunchAgents/
cp launchd/com.galew.speak-to-user.dev.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist 2>/dev/null || true
launchctl unload ~/Library/LaunchAgents/com.galew.speak-to-user.dev.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist
launchctl load ~/Library/LaunchAgents/com.galew.speak-to-user.dev.plist
```

Logs:

- `~/Library/Logs/speak-to-user.stable.stdout.log`
- `~/Library/Logs/speak-to-user.stable.stderr.log`
- `~/Library/Logs/speak-to-user.dev.stdout.log`
- `~/Library/Logs/speak-to-user.dev.stderr.log`

## Codex Config

Point Codex at the shared HTTP services instead of a stdio launcher:

```toml
[mcp_servers.speak_to_user]
url = "http://127.0.0.1:8765/mcp"

[mcp_servers.speak_to_user_dev]
url = "http://127.0.0.1:8766/mcp"
```

## Development

- [app/server.py](/Users/galew/Workspace/speak-to-user/app/server.py)
- [app/tools.py](/Users/galew/Workspace/speak-to-user/app/tools.py)
- [app/runtime.py](/Users/galew/Workspace/speak-to-user/app/runtime.py)

Checks:

```bash
uv run pytest
uv run ruff check .
uv run mypy .
```
