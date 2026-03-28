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
- `load_model`: loads one resident model by model id.
- `unload_model`: unloads one resident model by model id.
- `set_startup_model`: persists which model or models preload on server startup.
- `speak_text`: enqueues one full text job for the voice-design model.
- `speak_text_as_clone`: enqueues one full text job for the clone model using a local reference audio file and optional reference text.
- `generate_speech_profile`: stores a named reusable clone prompt artifact in the server's persistent state store.
- `generate_speech_profile_from_voice_design`: synthesizes a seed clip with the 1.7B voice-design model, then turns that clip into a reusable clone profile with the 0.6B clone model.
- `list_speech_profiles`: lists saved speech profiles and their metadata.
- `delete_speech_profile`: deletes a saved speech profile by name.
- `speak_with_profile`: enqueues one full text job using a previously saved speech profile.

There is no idle auto-unload, no detached helper process, and no file-generation path.

## Prompts and Resources

The server also exposes a compact FastMCP guidance surface for agent clients.

- Prompts:
  - `choose_speak_to_user_workflow`
  - `guide_speech_profile_workflow`
- Resources:
  - `guide://speak-to-user/usage`
  - `state://speak-to-user/status`
  - `state://speak-to-user/profiles`

These are meant to help agents choose the right tool, inspect live server state, and understand saved-profile usage without reading the full README every time.

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

Server startup now blocks only until the configured startup model selection is loaded. That selection is persisted with FastMCP storage through `set_startup_model`, and it can be `none`, `all`, or one of the concrete model ids. After startup, speech tools require their model to already be loaded; when the connected client supports FastMCP elicitation, the server can ask whether it should load the missing model before retrying. Every request is still chunked before playback, and the worker synthesizes ahead into a bounded waveform queue.

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
- the upstream Qwen docs describe the model as capable of rapid cloning from about a 3-second reference, so you do not need a long clip to get started
- in practice, a few clean spoken seconds is usually better than a longer but noisier clip
- avoid music, reverb, cross-talk, clipping, heavy background noise, and long silences
- trim leading and trailing silence when possible
- prefer WAV or FLAC over lossy formats
- keep the speaker emotionally and acoustically consistent across the clip
- if you have an accurate transcript for the reference clip, include it as `reference_text`

`reference_text` is worth using when it is genuinely accurate. It gives the model stronger conditioning than speaker-embedding-only mode. If the transcript is wrong, partial, or loosely paraphrased, omit it and let the service use `x_vector_only_mode=True`.

The upstream Qwen clone examples also make two practical points that matter here:

- the model itself accepts several `ref_audio` forms, but this server intentionally narrows that to a readable local file path for predictability
- `x_vector_only_mode=True` is a supported fallback, but Qwen explicitly notes that cloning quality may be reduced when you skip `reference_text`

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
- `generate_speech_profile_from_voice_design` is intentionally a short-seed workflow and rejects seed text longer than 240 characters
- `speak_with_profile` fails if the current clone model ID does not match the saved profile

Recommended profile workflow:

- use `generate_speech_profile` once with a clean reference clip
- prefer a short, accurate reference clip and transcript over a longer clip with messy timing, noise, or paraphrased text
- use `generate_speech_profile_from_voice_design` when you want the server to synthesize the seed clip for you before building the reusable clone prompt
- keep `generate_speech_profile_from_voice_design` seed text short and focused; use `speak_with_profile` for longer playback once the profile exists
- provide `reference_text` only when it closely matches the spoken clip
- use `list_speech_profiles` to confirm the saved profile metadata
- use `speak_with_profile` for repeat playback instead of resupplying the same reference audio every time
- delete old or low-quality profiles and recreate them from better source audio rather than trying to reuse a weak prompt forever

## Configuration

- `SPEAK_TO_USER_ENABLE_VOICE_DESIGN_MODEL`
  Default: `true`
  Used as the default startup preload preference when no persisted `set_startup_model` value exists yet.
- `SPEAK_TO_USER_ENABLE_CLONE_MODEL`
  Default: `true`
  Used as the default startup preload preference when no persisted `set_startup_model` value exists yet.
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
- `SPEAK_TO_USER_STATE_DIR`
  Default: `~/.local/gaelic-ghost/speak-to-user/profiles`
- `SPEAK_TO_USER_PLAYBACK_PREROLL_SECONDS`
  Default: `3.0`
- `SPEAK_TO_USER_PLAYBACK_PREROLL_CHUNKS`
  Default: `2`
- `SPEAK_TO_USER_PLAYBACK_WAVEFORM_QUEUE_MAXSIZE`
  Default: `16`
- `SPEAK_TO_USER_PLAYBACK_UNDERFLOW_RETRIES`
  Default: `2`
- `SPEAK_TO_USER_PLAYBACK_BACKEND`
  Allowed values: `sounddevice`, `wavbuffer`, `null`
  Default: `sounddevice`
- `SPEAK_TO_USER_WAVBUFFER_BINARY_PATH`
  Default: `app/vendor/wavbuffer/macos-arm64/wavbuffer` inside this repo
- `SPEAK_TO_USER_WAVBUFFER_QUEUE_DEPTH`
  Default: `8`
- `SPEAK_TO_USER_WAVBUFFER_PREROLL_MODE`
  Allowed values: `auto`, `seconds`, `buffers`
  Default: `seconds`
- `SPEAK_TO_USER_TTS_CHUNK_MAX_CHARS`
  Default: `240`
- `SPEAK_TO_USER_TTS_MAX_NEW_TOKENS`
  Default: `384`
- `SPEAK_TO_USER_TTS_MAX_CHUNK_SYNTH_SECONDS`
  Default: `30.0`
- `SPEAK_TO_USER_TTS_MAX_CHUNK_AUDIO_SECONDS`
  Default: `20.0`
- `SPEAK_TO_USER_OUTPUT_STREAM_LATENCY`
  Allowed values: `low`, `high`, or a positive number
  Default: `high`
- `SPEAK_TO_USER_LOG_LEVEL`
  Allowed values: `minimal`, `info`, `debug`
  Default: `info`

Runtime language inputs accept either full language names understood by the model, short codes like `en`, or common locale variants like `en-US` and `pt_BR`.

Operational notes:

- enabling both resident models increases steady RAM usage
- startup time increases when the persisted startup option resolves to both models because preload waits for both
- `load_model` and `unload_model` operate on concrete model ids reported by `tts_status`
- `unload_model` refuses to evict a model while that model has queued or active jobs
- `set_startup_model` accepts `none`, `all`, or one of the configured model ids and persists that choice in the FastMCP state store
- `SPEAK_TO_USER_STATE_DIR` controls where the FastMCP file-backed state store keeps persisted startup settings and saved speech profiles
- when `SPEAK_TO_USER_STATE_DIR` is not explicitly set, the server automatically migrates persisted state from the old default path at `~/Library/Application Support/speak-to-user/state` into `~/.local/gaelic-ghost/speak-to-user/profiles` if the new default is still empty
- if a speech or profile tool needs a model that is not loaded, the client should load it and retry; clients with FastMCP elicitation support can accept an in-band load prompt instead
- `tts_status` is the fastest way to confirm which models are loaded, which mode is active, and whether playback is already busy
- `SPEAK_TO_USER_OUTPUT_STREAM_LATENCY` affects the `sounddevice` backend only; it does not reduce model inference time
- `SPEAK_TO_USER_PLAYBACK_UNDERFLOW_RETRIES` controls how many times playback retries the current chunk after a playback underflow or `wavbuffer` starvation event
- `SPEAK_TO_USER_PLAYBACK_BACKEND=null` is a test-only silent sink that still runs synthesis but skips real audio output
- the `wavbuffer` backend expects a prebuilt binary path; do not point it at `swift run wavbuffer`
- this repo includes a bundled `wavbuffer` binary for macOS arm64 at `app/vendor/wavbuffer/macos-arm64/wavbuffer`
- `SPEAK_TO_USER_WAVBUFFER_BINARY_PATH` remains available as an override when you want to point at a fresh development build instead of the bundled binary
- `SPEAK_TO_USER_WAVBUFFER_PREROLL_MODE=auto` currently resolves to `seconds` so the runtime passes exactly one preroll flag to `wavbuffer`
- when `SPEAK_TO_USER_PLAYBACK_BACKEND=wavbuffer`, native preroll and underrun reporting come from the Swift playback binary rather than the Python runtime
- first-audio latency is usually dominated by model synthesis, especially on the 1.7B voice-design model
- `SPEAK_TO_USER_TTS_CHUNK_MAX_CHARS=240` is the current conservative default for this repo because even `320` to `360` character clone chunks on Gale's M4 Pro could still expand into 20+ second audio segments or 30+ second synth stalls
- `SPEAK_TO_USER_TTS_MAX_NEW_TOKENS=384` is passed explicitly into Qwen instead of relying on upstream defaults so the service has a local cap on per-chunk generation budget
- `SPEAK_TO_USER_TTS_MAX_CHUNK_AUDIO_SECONDS=20.0` fails chunks that decode into obviously reply-inappropriate audio durations instead of letting one absurd output dominate the worker
- `SPEAK_TO_USER_TTS_MAX_CHUNK_SYNTH_SECONDS=30.0` is a post-call guardrail: it marks an overlong synthesis as failed once Qwen returns, but it does not preempt the Qwen Python call mid-generation
- when `SPEAK_TO_USER_WAVBUFFER_PREROLL_MODE=buffers`, the runtime now caps the effective wavbuffer chunk-preroll target dynamically so 1- and 2-chunk jobs start after the first ready chunk while 3+-chunk jobs cap at 2 buffers even if the configured value is higher

## LaunchAgents

This repo includes LaunchAgent plists in [launchd/com.galew.speak-to-user.stable.plist](launchd/com.galew.speak-to-user.stable.plist) and [launchd/com.galew.speak-to-user.dev.plist](launchd/com.galew.speak-to-user.dev.plist).

They call [scripts/run_service.sh](scripts/run_service.sh), which sets a Homebrew-friendly `PATH` so tools like `sox` resolve correctly under `launchd`.
The checked-in plists use `$HOME` for the user-specific path prefix, but they still assume the default repo locations of `~/Workspace/speak-to-gale` for stable and `~/Workspace/speak-to-user` for dev. If you keep the repo somewhere else, edit the shell command in the plist before installing it.

- Stable service: `http://127.0.0.1:8765/mcp`
- Dev service: `http://127.0.0.1:8766/mcp`

Runtime observability is split between the LaunchAgent stderr logs and `tts_status`.
At the default `info` log level, the runtime emits structured JSON events for job queueing, synthesis, preroll, stream open/close, chunk playback, handoff completion, underflow recovery, completion, and failure. It also emits `speech_memory_snapshot` events around chunk synthesis, preroll satisfaction, output opening, and playback start so you can correlate memory swings with the playback pipeline in the LaunchAgent stderr log. The log now also includes `speech_chunk_ready_for_playback`, `speech_first_audio_started`, and `speech_job_metrics_summary` so inter-chunk delays, first-audio latency, synth-to-audio ratios, and failure reasons can be reviewed from stderr without a second metrics file. `tts_status` also includes a bounded in-memory `recent_events` history plus the latest event name and timestamps for the current job, chunk, and phase.

The checked-in LaunchAgent templates pin the service to the native `wavbuffer` backend, explicitly set `SPEAK_TO_USER_WAVBUFFER_QUEUE_DEPTH=8`, keep `SPEAK_TO_USER_PLAYBACK_PREROLL_SECONDS=5.0` for compatibility, and force `SPEAK_TO_USER_WAVBUFFER_PREROLL_MODE=buffers` with `SPEAK_TO_USER_PLAYBACK_PREROLL_CHUNKS=3`. The runtime treats that configured chunk-preroll as an upper bound rather than a literal always-wait-for-3 rule: 1- and 2-chunk jobs start after the first chunk, while 3+-chunk jobs cap at 2 buffers. The same templates also pin the current reply-sized chunking guardrails used in development: `SPEAK_TO_USER_TTS_CHUNK_MAX_CHARS=240`, `SPEAK_TO_USER_TTS_MAX_NEW_TOKENS=384`, `SPEAK_TO_USER_TTS_MAX_CHUNK_SYNTH_SECONDS=30.0`, and `SPEAK_TO_USER_TTS_MAX_CHUNK_AUDIO_SECONDS=20.0`. They rely on the bundled repo copy of `wavbuffer` by default, rather than pointing at a sibling Swift build output. That is a service-level setting for the included launchd setup, not a change to the runtime-wide default documented in the configuration table above.

To refresh the bundled `wavbuffer` binary from the sibling Swift repo after rebuilding it there:

```bash
cp "$HOME/Workspace/swiftly-play/.build/release/wavbuffer" app/vendor/wavbuffer/macos-arm64/wavbuffer
chmod +x app/vendor/wavbuffer/macos-arm64/wavbuffer
```

When diagnosing clone quality or playback problems, check both:

- `tts_status` for live queue, model, and recent-event state
- the LaunchAgent stderr log for the full structured event stream across requests and reconnects
- repeated `wavbuffer event=underrun` plus `reason="stream_starved"` means playback drained faster than the next chunk finished synthesizing; reduce chunk size or tighten generation guardrails first, then increase buffer-count preroll if starvation still remains

Operational warning:

- SoundSource.app processing or routing can destabilize playback for this service. In local testing it could produce garbled or static-heavy output and even crash SoundSource itself. Prefer excluding `speak-to-user` from SoundSource processing and routing when possible.
- If SoundSource interference is suspected, prefer testing with `SPEAK_TO_USER_PLAYBACK_BACKEND=wavbuffer` first so playback runs through the native Swift sink instead of the in-process Python audio stream.

Install or refresh the LaunchAgents:

```bash
cp launchd/com.galew.speak-to-user.stable.plist ~/Library/LaunchAgents/
cp launchd/com.galew.speak-to-user.dev.plist ~/Library/LaunchAgents/
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist 2>/dev/null || true
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.galew.speak-to-user.dev.plist 2>/dev/null || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.galew.speak-to-user.dev.plist
```

Start the installed LaunchAgents:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.galew.speak-to-user.dev.plist
```

Stop the installed LaunchAgents:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.galew.speak-to-user.dev.plist
```

Manage each service individually:

```bash
# Stable / prod
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist

# Dev
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.galew.speak-to-user.dev.plist
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.galew.speak-to-user.dev.plist
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

- [app/server.py](app/server.py)
- [app/tools.py](app/tools.py)
- [app/runtime.py](app/runtime.py)

Default checks:

```bash
uv run pytest
uv run ruff check .
uv run mypy .
```

Coverage:

```bash
uv run pytest --cov=app --cov-report=term-missing
```

Optional sequential e2e suite:

```bash
sh scripts/run_e2e_tests.sh
```

The e2e suite manages its own dedicated HTTP server on a non-live port with `SPEAK_TO_USER_PLAYBACK_BACKEND=null`, then exercises the full HTTP control surface over real server starts and restarts.
It covers prompts, resources, all public tools, high-signal failure paths, and restart persistence for the startup-model setting and persisted profiles.

Memory-safety guidance for e2e:

- stop the live stable service first so the e2e server can use that memory budget
- do not run the e2e suite in parallel
- do not run multiple e2e invocations at the same time
- do not run the e2e suite alongside another model-heavy test or dev session

Stop the installed stable service before e2e:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist
```

Restart it after e2e:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist
```

If you want to run the optional e2e tests directly without the wrapper script, provide the dedicated host, port, and path the managed test server should use:

```bash
SPEAK_TO_USER_E2E_HOST=127.0.0.1 \
SPEAK_TO_USER_E2E_PORT=8876 \
SPEAK_TO_USER_E2E_PATH=/mcp \
uv run pytest -m e2e -q -o addopts='-q --strict-markers'
```

Automation:

- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the fast default checks, plist linting, and shell syntax checks on push and pull request.
- [`.github/workflows/manual-model-e2e.yml`](.github/workflows/manual-model-e2e.yml) is an opt-in GitHub Actions workflow that validates the optional e2e harness shape without pretending hosted runners can execute the local model stack.
