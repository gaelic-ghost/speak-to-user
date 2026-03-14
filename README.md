# speak-to-user

`speak-to-user` is a local [FastMCP](https://gofastmcp.com/) server with one job: take text jobs and speak them through the host machine.

It uses [`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign), loads the model when the server starts, keeps it resident for the life of the server process, and exposes one playback path: `speak_text`.

## Tools

- `health`: simple smoke-test response.
- `tts_status`: reports whether the resident model is ready and shows queue and playback status.
- `speak_text`: enqueues one full text job exactly as sent by the agent.

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

Server startup blocks until the model is loaded. After that, `speak_text` pushes one full text job into one in-process FIFO queue, and the server's playback worker speaks jobs in order while the Codex MCP session stays alive.

## Configuration

- `SPEAK_TO_USER_MODEL_ID`
  Default: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `SPEAK_TO_USER_DEVICE`
  Allowed values: `auto`, `mps`, `cpu`
  Default: `auto`

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
