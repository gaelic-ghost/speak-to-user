# speak-to-user

`speak-to-user` is a local [FastMCP](https://gofastmcp.com/) server for coding agents that need a dependable, host-local text-to-speech path for user-facing replies.

It wraps [`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign), keeps model lifecycle management inside the server process, writes generated audio to disk when explicitly requested, and can play speech directly from memory on the host machine.

The current focus is simple and practical: give an agent one reliable way to speak to a local user without introducing a large framework surface or a remote service dependency.

## Why This Exists

Agent workflows often end with text output even when spoken output would be more natural, faster to consume, or more accessible. This server gives agents a small MCP tool surface for:

- checking runtime readiness
- loading and unloading the TTS model
- generating local audio files
- speaking a response out loud on the host machine

That makes it useful as a local building block for Codex-style workflows, agent tooling experiments, accessibility support, and hands-free feedback loops.

## Current Capabilities

- FastMCP server with a focused TTS-oriented tool surface
- Background model preload on startup
- Automatic idle unloading to reclaim memory after inactivity
- Runtime status inspection, including preload timing and last-error visibility
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

On startup, the server:

1. builds a `TTSRuntime` from environment configuration
2. starts the idle-unload watchdog
3. begins background preload of the TTS model

Startup is intentionally non-blocking. If preload is still in progress, the server still comes up, and agents can inspect readiness through `tts_status`. The `ready` field only flips to `true` once the model is actually loaded and no runtime error is recorded.

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

Generates speech and plays it locally on the host machine without retaining an output file as part of the tool contract. Internally it chunks long text, sends the full chunk list through one batched `generate_voice_design(...)` call, concatenates the returned waveform list into one in-memory audio buffer, and plays that single buffer directly instead of persisting temporary files. This tool supports FastMCP background task execution and can also be called synchronously by clients that do not send task metadata, which improves compatibility with simpler MCP tool runners. Long text is automatically chunked into paragraph-oriented units, with sentence and word fallback when a single paragraph is still too large. It reports progress for:

- generation
- buffer concatenation
- buffer playback
- completion

## Configuration

Environment variables:

- `SPEAK_TO_USER_MODEL_ID`
  Default: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `SPEAK_TO_USER_IDLE_UNLOAD_SECONDS`
  Default: `1200`
- `SPEAK_TO_USER_OUTPUT_DIR`
  Default: `generated-audio`
- `SPEAK_TO_USER_DEVICE`
  Allowed values: `auto`, `mps`, `cpu`
  Default: `auto`

## Output Behavior

- Generated audio is written to the configured output directory only for `generate_audio`.
- Relative output directories are resolved from the current working directory at runtime.
- When no `filename_stem` is provided, files get a UTC timestamp-based name.
- `filename_stem` values are sanitized to keep paths predictable and filesystem-safe.
- `speak_text` is the exception: it keeps playback in memory and does not persist chunk audio to disk.

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
- Errors are captured in runtime state and surfaced via `tts_status` instead of crashing the process whenever possible.
- The model can be unloaded after inactivity to reduce resource usage on a development machine.
- The tool surface is deliberately small right now so agent integrations stay easy to reason about.

## Development

Project layout:

- `app/server.py`: FastMCP entrypoint and tool registration
- `app/runtime.py`: runtime lifecycle, preload, idle unloading, generation, and playback
- `app/tools.py`: tool adapters between FastMCP context and runtime operations
- `tests/test_runtime.py`: runtime tests
- `tests/test_server.py`: server and tool behavior tests
- `ROADMAP.md`: planned future work
- `WORKFLOWS.md`: end-to-end tool and data-shape flow reference

Validation commands:

```bash
uv run pytest
uv run ruff check .
uv run mypy .
```

## Future Work

Planned work currently includes:

- queue-based speech generation
- persistent file storage for batch generation
- agent-facing guidance through MCP Prompts and Resources
- guidance for chunking long replies
- voice description profile support
- voice profile selection through MCP user elicitation
- automatic voice profile switching using FastMCP Context
- FastMCP Apps and UI support
- a related agent skill for consistent usage guidance
- support for additional TTS runtimes
- packaging and distribution on PyPI
- composition with a FastAPI service
- distribution as a macOS `MenuBarExtra` app

Those plans are intentionally aspirational, not committed API guarantees. The current implementation is still centered on a compact, local MCP speech server.
