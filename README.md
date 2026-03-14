# speak-to-user

`speak-to-user` is a local [FastMCP](https://gofastmcp.com/) server for coding agents that need a dependable, host-local text-to-speech path for user-facing replies.

It wraps [`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign), keeps model lifecycle management inside the server process, writes generated audio to disk only when explicitly requested, and can speak directly through the host machine without persisting playback output.

The implementation is intentionally small: one runtime per server process for direct tools, one detached playback helper for `speak_text`, one model batch call per speech request, and one live audio stream per playback job.

## Why This Exists

Agent workflows often end with text output even when spoken output would be faster to consume, more natural, or more accessible. This server gives agents a compact MCP tool surface for:

- checking runtime readiness
- loading and unloading the TTS model
- generating local audio files
- speaking a response out loud on the host machine

## Current Capabilities

- FastMCP server with a focused TTS-oriented tool surface
- Background model preload on startup
- Automatic idle unloading to reclaim memory after inactivity
- Runtime status inspection, including preload timing and last-error visibility
- Bounded speech queue with request-level backpressure
- Local file generation in `wav` and `flac`
- Local playback on macOS through the host machine
- Host-local configuration through environment variables

## Requirements

- macOS
- [uv](https://docs.astral.sh/uv/)
- Python 3.12
- Enough local memory for the Qwen TTS runtime

## Installation

```bash
uv sync
```

## Running The Server

```bash
uv run python app/server.py
```

The default FastMCP transport is stdio. The server starts on demand, creates one `TTSRuntime` for that process, and shuts the runtime down when the stdio session ends.

On startup, the server:

1. builds a `TTSRuntime` from environment configuration
2. starts the idle-unload watchdog
3. begins background preload of the TTS model

Startup is intentionally non-blocking. If preload is still in progress, the server still comes up, and agents can inspect readiness through `tts_status`. The `ready` field flips to `true` only once the model is actually loaded and no runtime error is recorded.

## MCP Tools

### `health`

Returns a lightweight health payload with a UTC timestamp.

### `tts_status`

Returns current runtime state, including:

- whether the model is loaded
- whether preload is in progress
- the resolved device
- the configured model ID
- idle-unload settings
- key lifecycle timestamps
- output directory
- the last recorded error, if any
- detached speech submission state
- the last recorded detached playback error, if any

### `load_model`

Loads the TTS model into memory. If preload is already running, the call waits for preload to finish before deciding whether another load is needed.

### `unload_model`

Unloads the model manually and reports updated runtime state.

### `set_idle_unload_timeout`

Updates the automatic idle-unload timeout in seconds.

### `generate_audio`

Synthesizes a single audio file and returns metadata including:

- output path
- format
- sample rate
- sample count
- duration
- model ID
- resolved device
- normalized language

### `speak_text`

Queues speech for local playback on the host machine without retaining an output file as part of the tool contract.

Internally, `speak_text` chunks long text only to stay within model-friendly text sizes, writes one small job payload to disk, and spawns a detached local helper process. That helper performs one `generate_voice_design(...)` batch call for the full request, receives one ordered waveform list back, buffers a small initial amount of generated audio, opens one live `sounddevice` output stream, and writes the waveforms to that stream in FIFO order. The MCP call returns after handoff, so stdio-session teardown does not kill playback.

## Configuration

Environment variables:

- `SPEAK_TO_USER_MODEL_ID`
  Default: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `SPEAK_TO_USER_IDLE_UNLOAD_SECONDS`
  Default: `1200`
- `SPEAK_TO_USER_OUTPUT_DIR`
  Default: `generated-audio`
- `SPEAK_TO_USER_JOB_DIR`
  Default: `/tmp/speak-to-user-jobs`
- `SPEAK_TO_USER_DEVICE`
  Allowed values: `auto`, `mps`, `cpu`
  Default: `auto`

## Output Behavior

- Generated audio is written to the configured output directory only for `generate_audio`.
- Relative output directories are resolved from the current working directory at runtime.
- When no `filename_stem` is provided, files get a UTC timestamp-based name.
- `filename_stem` values are sanitized to keep paths predictable and filesystem-safe.
- `speak_text` keeps playback in memory and does not persist playback audio to disk.
- `speak_text` returns after a detached helper process has been launched; playback continues asynchronously on the host machine.
- Playback uses one model batch call per speech request, then streams generated waveforms through one live audio output stream.
- Detached playback keeps running even if the stdio MCP server process exits immediately after the tool result is returned.

## Language And Voice Inputs

The current runtime expects:

- `text`: the speech content to synthesize
- `voice_description`: an instruction-style description of the desired voice
- `language`: defaults to `en`

Some language aliases are normalized internally:

- `en` and `english` become `English`
- `zh` and `chinese` become `Chinese`
- `auto` becomes `Auto`

## Design Notes

- The server is intentionally local-first.
- Stdout is kept clean so stdio MCP framing is not polluted by model-side prints.
- Errors are captured in runtime state and surfaced via `tts_status` instead of crashing the process whenever possible.
- The model can be unloaded after inactivity to reduce resource usage on a development machine.
- The tool surface is deliberately small so agent integrations stay easy to reason about.

## Development

Project layout:

- `app/server.py`: FastMCP entrypoint and tool registration
- `app/runtime.py`: runtime lifecycle, preload, idle unloading, generation, and playback
- `app/tools.py`: tool adapters and text chunking handoff
- `app/text_chunking.py`: paragraph-, sentence-, and word-level chunking helpers
- `tests/test_runtime.py`: runtime tests
- `tests/test_server.py`: server and tool behavior tests
- `ROADMAP.md`: planned future work
- `WORKFLOWS.md`: end-to-end tool and data-flow reference

Validation commands:

```bash
uv run pytest
uv run ruff check .
uv run mypy .
```
