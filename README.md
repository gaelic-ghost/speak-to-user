# speak-to-user

`speak-to-user` is a local [FastMCP](https://gofastmcp.com/) server with one job: take text jobs and speak them through the host machine.

It uses [`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign), loads the model when the server starts, keeps it resident for the life of the server process, and exposes one playback path: `speak_text`.

The intended deployment mode is one long-lived local Streamable HTTP MCP service so multiple Codex clients can share the same resident TTS model instead of launching separate stdio subprocesses.

## Tools

- `health`: simple smoke-test response.
- `tts_status`: reports whether the resident model is ready and shows queue status plus the current speech phase (`idle`, `synthesizing`, `opening_output`, or `playing`).
- `speak_text`: enqueues one full text job and always chunks text sentence-by-sentence before playback, with word fallback only for overlong single sentences.

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

Server startup blocks until the model is loaded. After that, `speak_text` pushes one full text job into one in-process FIFO queue. Every request is chunked sentence-by-sentence inside that job, the worker synthesizes ahead into a bounded waveform queue, opens playback only after a larger preroll, and then keeps generating and writing later chunks in order while the same output stream stays open. During active work, `tts_status` exposes whether the runtime is still synthesizing audio or has reached device playback.

For the dev checkout, prefer a different port so it does not collide with the stable service:

```bash
SPEAK_TO_USER_PORT=8766 uv run python app/server.py
```

## Configuration

- `SPEAK_TO_USER_MODEL_ID`
  Default: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
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

Runtime language inputs accept either full language names understood by the model, short codes like `en`, or common locale variants like `en-US` and `pt_BR`.

## LaunchAgents

This repo includes LaunchAgent plists in [launchd/com.galew.speak-to-user.stable.plist](/Users/galew/Workspace/speak-to-user/launchd/com.galew.speak-to-user.stable.plist) and [launchd/com.galew.speak-to-user.dev.plist](/Users/galew/Workspace/speak-to-user/launchd/com.galew.speak-to-user.dev.plist).

They call [scripts/run_service.sh](/Users/galew/Workspace/speak-to-user/scripts/run_service.sh), which sets a Homebrew-friendly `PATH` so tools like `sox` resolve correctly under `launchd`.

- Stable service: `http://127.0.0.1:8765/mcp`
- Dev service: `http://127.0.0.1:8766/mcp`

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
