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
    C --> D["runtime.speak_text(...)"]
    D --> E["normalize text, voice, language"]
    E --> F["enqueue one job into in-process FIFO queue"]
    F --> G["speech worker reads job"]
    G --> H["play_speech_chunks(...)"]
    H --> I["_synthesize_audio_batch(texts=chunks, ...)"]
    I --> J["_model.generate_voice_design(...)"]
    J --> K["prepare waveform arrays"]
    K --> L["small preroll buffer"]
    L --> M["_open_output_stream(...)"]
    M --> N["_write_output_stream_chunk(...) in FIFO order"]
```

`speak_text` is a plain MCP tool. It enqueues one full text request exactly as sent by the caller. Playback then happens on the already-running in-process worker.

Important behavior:

- one `speak_text` call creates one queued playback job
- one playback job results in one model batch call
- playback uses one output stream per speech request
- playback audio is not persisted to disk
- `speech_phase` exposes whether the job is still synthesizing or has reached playback

## Text Chunking

```mermaid
flowchart LR
    A["chunk_text_for_tts(text, max_chars)"] --> B{"text fits?"}
    B -->|yes| C["return [text]"]
    B -->|no| D["split by paragraph"]
    D --> E{"paragraph fits?"}
    E -->|yes| F["append paragraph"]
    E -->|no| G["_chunk_sentences(...)"]
    G --> H{"sentence fits?"}
    H -->|yes| I["append sentence"]
    H -->|no| J["_chunk_words(...)"]
    J --> K{"word fits?"}
    K -->|yes| L["append word-packed chunk"]
    K -->|no| M["_split_long_word(...)"]
```

Chunking is paragraph-first, with sentence and word fallback only when needed. It is used when longer text needs to be split into model-friendly chunks before one batched synthesis request.
