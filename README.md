# speak-to-user

`speak-to-user` is a local [FastMCP](https://gofastmcp.com/) server with one job: take text jobs and speak them through the host machine.

It uses [`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign), loads the model when the server starts, keeps it resident for the life of the server process, and exposes one playback path: `speak_text`.

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

Server startup blocks until the model is loaded. After that, `speak_text` pushes one full text job into one in-process FIFO queue. Every request is chunked sentence-by-sentence inside that job, the worker synthesizes chunk 1, opens playback after a tiny preroll, and then keeps generating and writing later chunks in order while the same output stream stays open. During active work, `tts_status` exposes whether the runtime is still synthesizing audio or has reached device playback.

## Configuration

- `SPEAK_TO_USER_MODEL_ID`
  Default: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `SPEAK_TO_USER_DEVICE`
  Allowed values: `auto`, `mps`, `cpu`
  Default: `auto`

Runtime language inputs accept either full language names understood by the model, short codes like `en`, or common locale variants like `en-US` and `pt_BR`.

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
