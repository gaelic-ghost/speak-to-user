# speak-to-user

`speak-to-user` is a local [FastMCP](https://gofastmcp.com/) server with one job: take text jobs and speak them through the host machine.

It uses a resident voice-design model for normal TTS and can also keep a separate clone model resident for reference-audio voice cloning:

- [`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
- [`Qwen/Qwen3-TTS-12Hz-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)

The intended deployment mode is one long-lived local Streamable HTTP MCP service so multiple Codex clients can share the same resident TTS model instead of launching separate stdio subprocesses.

## Model Roles

- `speak_text` uses the resident 1.7B voice-design model for normal narrated replies driven by `voice_description`.
- `speak_text_as_clone` uses the resident 0.6B clone model for one-off reference-audio cloning.
- `generate_speech_profile` and `speak_with_profile` use the same 0.6B clone model, but save and reuse a precomputed clone prompt so repeated playback does not need to rebuild the prompt from the original reference clip every time.

The two models solve different problems. Voice design is the default spoken-reply path. Clone tools are for matching a specific voice from reference audio.

## Tools

- `health`: simple smoke-test response.
- `tts_status`: reports queue status, observability events, and separate status for the voice-design and clone models.
- `speak_text`: enqueues one full text job for the voice-design model.
- `speak_text_as_clone`: enqueues one full text job for the clone model using a local reference audio file and optional reference text.
- `generate_speech_profile`: stores a named reusable clone prompt artifact in the server's persistent state store.
- `generate_speech_profile_from_voice_design`: synthesizes a seed clip with the 1.7B voice-design model, then turns that clip into a reusable clone profile with the 0.6B clone model.
- `list_speech_profiles`: lists saved speech profiles and their metadata.
- `delete_speech_profile`: deletes a saved speech profile by name.
- `speak_with_profile`: enqueues one full text job using a previously saved speech profile.

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

Server startup blocks until all enabled models are loaded. After that, both `speak_text` and `speak_text_as_clone` push one full text job into one in-process FIFO queue. Every request is chunked before playback, and the worker synthesizes ahead into a bounded waveform queue.

With the default `sounddevice` backend, the runtime itself owns preroll and writes later chunks through one in-process output stream. With the optional `wavbuffer` backend, the runtime instead wraps each synthesized chunk into a complete WAV buffer and streams those buffers into a prebuilt native Swift playback binary, which owns preroll and underrun detection for that path. During active work, `tts_status` exposes whether the runtime is still synthesizing audio or has reached device playback, plus which synthesis mode and playback backend are active.

Because playback is intentionally global and serial, multiple MCP clients can stay connected at once, but only one speech job plays at a time.

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

### Clone Best Practices

For the best results with the 0.6B clone model:

- use a short, clean sample from one speaker only
- prefer roughly 5 to 20 seconds of speech
- avoid music, reverb, cross-talk, clipping, heavy background noise, and long silences
- trim leading and trailing silence when possible
- prefer WAV or FLAC over lossy formats
- keep the speaker emotionally and acoustically consistent across the clip
- if you have an accurate transcript for the reference clip, include it as `reference_text`

`reference_text` is worth using when it is genuinely accurate. It gives the model stronger conditioning than speaker-embedding-only mode. If the transcript is wrong, partial, or loosely paraphrased, omit it and let the service use `x_vector_only_mode=True`.

For one-off requests, use `speak_text_as_clone`. For a voice you expect to reuse, create a named profile first and then use `speak_with_profile`.

## Speech Profiles

Profiles are reusable named clone prompts backed by the FastMCP server's persistent state store.

- `generate_speech_profile` takes:
  - `name`
  - `reference_audio_path`
  - optional `reference_text`
- `generate_speech_profile_from_voice_design` takes:
  - `name`
  - `text`
  - `voice_description`
  - optional `language`
- `speak_with_profile` takes:
  - `name`
  - `text`
  - optional `language`

Profile behavior:

- profiles are bound to the active clone model ID at creation time
- profile names must be unique; duplicate creation fails
- profiles store a precomputed Qwen clone prompt artifact rather than the original file path alone
- voice-designed profiles also persist their seed text and voice description alongside the saved prompt artifact
- `speak_with_profile` fails if the current clone model ID does not match the saved profile

Recommended profile workflow:

- use `generate_speech_profile` once with a clean reference clip
- use `generate_speech_profile_from_voice_design` when you want the server to synthesize the seed clip for you before building the reusable clone prompt
- provide `reference_text` only when it closely matches the spoken clip
- use `list_speech_profiles` to confirm the saved profile metadata
- use `speak_with_profile` for repeat playback instead of resupplying the same reference audio every time
- delete old or low-quality profiles and recreate them from better source audio rather than trying to reuse a weak prompt forever

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
- `SPEAK_TO_USER_PLAYBACK_UNDERFLOW_RETRIES`
  Default: `2`
- `SPEAK_TO_USER_PLAYBACK_BACKEND`
  Allowed values: `sounddevice`, `wavbuffer`
  Default: `sounddevice`
- `SPEAK_TO_USER_WAVBUFFER_BINARY_PATH`
  Default: `app/vendor/wavbuffer/macos-arm64/wavbuffer` inside this repo
- `SPEAK_TO_USER_WAVBUFFER_QUEUE_DEPTH`
  Default: `8`
- `SPEAK_TO_USER_WAVBUFFER_PREROLL_MODE`
  Allowed values: `auto`, `seconds`, `buffers`
  Default: `seconds`
- `SPEAK_TO_USER_OUTPUT_STREAM_LATENCY`
  Allowed values: `low`, `high`, or a positive number
  Default: `high`
- `SPEAK_TO_USER_LOG_LEVEL`
  Allowed values: `minimal`, `info`, `debug`
  Default: `info`

Runtime language inputs accept either full language names understood by the model, short codes like `en`, or common locale variants like `en-US` and `pt_BR`.

Operational notes:

- enabling both resident models increases steady RAM usage
- startup time increases when both models are enabled because preload waits for both
- `tts_status` is the fastest way to confirm which models are loaded, which mode is active, and whether playback is already busy
- `SPEAK_TO_USER_OUTPUT_STREAM_LATENCY` affects the `sounddevice` backend only; it does not reduce model inference time
- `SPEAK_TO_USER_PLAYBACK_UNDERFLOW_RETRIES` controls how many times playback retries the current chunk after a playback underflow or `wavbuffer` starvation event
- the `wavbuffer` backend expects a prebuilt binary path; do not point it at `swift run wavbuffer`
- this repo includes a bundled `wavbuffer` binary for macOS arm64 at [app/vendor/wavbuffer/macos-arm64/wavbuffer](/Users/galew/Workspace/speak-to-user/app/vendor/wavbuffer/macos-arm64/wavbuffer)
- `SPEAK_TO_USER_WAVBUFFER_BINARY_PATH` remains available as an override when you want to point at a fresh development build instead of the bundled binary
- `SPEAK_TO_USER_WAVBUFFER_PREROLL_MODE=auto` currently resolves to `seconds` so the runtime passes exactly one preroll flag to `wavbuffer`
- when `SPEAK_TO_USER_PLAYBACK_BACKEND=wavbuffer`, native preroll and underrun reporting come from the Swift playback binary rather than the Python runtime
- first-audio latency is usually dominated by model synthesis, especially on the 1.7B voice-design model

## LaunchAgents

This repo includes LaunchAgent plists in [launchd/com.galew.speak-to-user.stable.plist](/Users/galew/Workspace/speak-to-user/launchd/com.galew.speak-to-user.stable.plist) and [launchd/com.galew.speak-to-user.dev.plist](/Users/galew/Workspace/speak-to-user/launchd/com.galew.speak-to-user.dev.plist).

They call [scripts/run_service.sh](/Users/galew/Workspace/speak-to-user/scripts/run_service.sh), which sets a Homebrew-friendly `PATH` so tools like `sox` resolve correctly under `launchd`.

- Stable service: `http://127.0.0.1:8765/mcp`
- Dev service: `http://127.0.0.1:8766/mcp`

Runtime observability is split between the LaunchAgent stderr logs and `tts_status`.
At the default `info` log level, the runtime emits structured JSON events for job queueing, synthesis, preroll, stream open/close, chunk playback, handoff completion, underflow recovery, completion, and failure. It also emits `speech_memory_snapshot` events around chunk synthesis, preroll satisfaction, output opening, and playback start so you can correlate memory swings with the playback pipeline in the LaunchAgent stderr log. `tts_status` also includes a bounded in-memory `recent_events` history plus the latest event name and timestamps for the current job, chunk, and phase.

The checked-in LaunchAgent templates currently pin the service to the native `wavbuffer` backend with `SPEAK_TO_USER_PLAYBACK_PREROLL_SECONDS=5.0`. They now rely on the bundled repo copy of `wavbuffer` by default, rather than pointing at a sibling Swift build output. That is a service-level setting for the included launchd setup, not a change to the runtime-wide default documented in the configuration table above.

To refresh the bundled `wavbuffer` binary from the sibling Swift repo after rebuilding it there:

```bash
cp ~/Workspace/swiftly-play/.build/release/wavbuffer app/vendor/wavbuffer/macos-arm64/wavbuffer
chmod +x app/vendor/wavbuffer/macos-arm64/wavbuffer
```

When diagnosing clone quality or playback problems, check both:

- `tts_status` for live queue, model, and recent-event state
- the LaunchAgent stderr log for the full structured event stream across requests and reconnects

Operational warning:

- SoundSource.app processing or routing can destabilize playback for this service. In local testing it could produce garbled or static-heavy output and even crash SoundSource itself. Prefer excluding `speak-to-user` from SoundSource processing and routing when possible.
- If SoundSource interference is suspected, prefer testing with `SPEAK_TO_USER_PLAYBACK_BACKEND=wavbuffer` first so playback runs through the native Swift sink instead of the in-process Python audio stream.

Install or refresh the LaunchAgents:

```bash
cp launchd/com.galew.speak-to-user.stable.plist ~/Library/LaunchAgents/
cp launchd/com.galew.speak-to-user.dev.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist 2>/dev/null || true
launchctl unload ~/Library/LaunchAgents/com.galew.speak-to-user.dev.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist
launchctl load ~/Library/LaunchAgents/com.galew.speak-to-user.dev.plist
```

Start the installed LaunchAgents:

```bash
launchctl load ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist
launchctl load ~/Library/LaunchAgents/com.galew.speak-to-user.dev.plist
```

Stop the installed LaunchAgents:

```bash
launchctl unload ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist
launchctl unload ~/Library/LaunchAgents/com.galew.speak-to-user.dev.plist
```

Manage each service individually:

```bash
# Stable / prod
launchctl load ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist
launchctl unload ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist

# Dev
launchctl load ~/Library/LaunchAgents/com.galew.speak-to-user.dev.plist
launchctl unload ~/Library/LaunchAgents/com.galew.speak-to-user.dev.plist
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
