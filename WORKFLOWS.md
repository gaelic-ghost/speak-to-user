# Tool Workflows

`WORKFLOWS.md` documents the live MCP paths in this repo. It follows the current implementation in `app/server.py`, `app/tools.py`, `app/runtime.py`, and `app/text_chunking.py`.

## Shared Lifespan Flow

```mermaid
flowchart LR
    A["FastMCP startup"] --> B["app_lifespan()"]
    B --> C["TTSRuntime.from_env()"]
    C --> D["TTSRuntime(...)"]
    D --> E["runtime.preload()"]
    E --> F["runtime.start_speech_worker()"]
    E --> G["runtime.load_model()"]
    B --> H["yield {'runtime': runtime} into lifespan_context"]
    H --> I["tool calls resolve ctx.lifespan_context['runtime']"]
    B --> J["runtime.shutdown()"]
```

The server creates one `TTSRuntime` per FastMCP process during lifespan setup. Preload starts the in-process speech worker and blocks until the model is resident. The runtime is then shared by every tool call in that process and shut down when the server exits.

## `health`

```mermaid
flowchart LR
    A["MCP tool: server.health()"] --> B["tools.health_payload()"]
    B --> C["dict[str, str]"]
```

`health` builds and returns a small in-process payload with no runtime interaction.

## `tts_status`

```mermaid
flowchart LR
    A["MCP tool: server.tts_status(ctx)"] --> B["tools.tts_status(ctx)"]
    B --> C["tools._runtime_from_context(ctx)"]
    C --> D["runtime.status()"]
    D --> E["_status_payload_locked()"]
```

`tts_status` resolves the shared runtime from lifespan context and returns one lock-protected status snapshot.

Important status behavior:

- `speech_phase` is `idle` when no job is active.
- `speech_phase` is `synthesizing` while the model is generating audio for the active request.
- `speech_phase` is `opening_output` while the output stream is being opened.
- `speech_phase` is `playing` while waveform chunks are being written to the audio device.

## `speak_text`

```mermaid
flowchart LR
    A["MCP tool: server.speak_text(...)"] --> B["tools.speak_text(ctx, ...)"]
    B --> C["tools._runtime_from_context(ctx)"]
    C --> D["chunk_text_for_tts(text)"]
    D --> E["runtime.speak_text(...)"]
    E --> F["normalize voice, language, chunks"]
    F --> G["enqueue one job into in-process FIFO queue"]
    G --> H["speech worker reads job"]
    H --> I["play_speech_chunks(...)"]
    I --> J["_synthesize_audio_chunk(text=chunk_1, ...)"]
    J --> K["_model.generate_voice_design(...)"]
    K --> L["small preroll buffer"]
    L --> M["_open_output_stream(...)"]
    M --> N["_write_output_stream_chunk(chunk_1)"]
    N --> O["_synthesize_audio_chunk(text=chunk_2, ...) then write chunk_2, repeat"]
```

`speak_text` is a plain MCP tool. It enqueues one full text job for the caller's request, chunking longer text first so playback stays model-friendly. Playback then happens on the already-running in-process worker.

Important behavior:

- one `speak_text` call creates one queued playback job
- one playback job keeps one output stream open while chunks are synthesized and written in order
- playback uses one output stream per speech request
- playback audio is not persisted to disk
- `speech_phase` exposes whether the job is still synthesizing or has reached playback

## Text Chunking

```mermaid
flowchart LR
    A["chunk_text_for_tts(text, max_chars)"] --> B["split by sentence"]
    B --> C{"sentence fits?"}
    C -->|yes| D["append one sentence chunk"]
    C -->|no| E["_chunk_words(...)"]
    E --> F{"word fits?"}
    F -->|yes| G["append word-packed chunk"]
    F -->|no| H["_split_long_word(...)"]
```

Chunking is always sentence-first. Each sentence becomes its own chunk whenever possible, and only individual overlong sentences fall back to word splitting. It is used for every `speak_text` request before rolling chunk synthesis and playback.
