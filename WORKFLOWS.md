# Tool Workflows

`WORKFLOWS.md` documents the live MCP paths in this repo. It follows the real runtime flow in `app/server.py`, `app/tools.py`, `app/runtime.py`, `app/playback_job.py`, and `app/text_chunking.py`.

## Shared Lifespan Flow

```mermaid
flowchart LR
    A["FastMCP startup"] --> B["app_lifespan()"]
    B --> C["TTSRuntime.from_env()"]
    C --> D["TTSRuntime(...)"]
    D --> E["runtime.start_background_preload()"]
    E --> F["runtime.start_watchdog()"]
    E --> H["spawn preload thread"]
    H --> I["_background_preload_worker()"]
    I --> J["load_model()"]
    B --> K["yield {'runtime': runtime} into lifespan_context"]
    K --> L["tool calls resolve ctx.lifespan_context['runtime']"]
    B --> M["runtime.shutdown()"]
```

The server creates one `TTSRuntime` per FastMCP process during lifespan setup. The runtime is shared by every tool call in that process and shut down when the server exits.

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

## `generate_audio`

```mermaid
flowchart LR
    A["MCP tool: server.generate_audio(...)"] --> B["tools.generate_audio(ctx, ...)"]
    B --> C["runtime.generate_audio(...)"]
    C --> D["normalize text/voice/language/format"]
    D --> E["_build_output_path(...)"]
    D --> F["_synthesize_audio(...)"]
    F --> G["_synthesize_audio_batch(texts=[text], ...)"]
    G --> H["_model.generate_voice_design(...)"]
    H --> I["_coerce_waveform(...)"]
    I --> J["sf.write(...)"]
```

`generate_audio` is the file-producing path. It normalizes inputs, synthesizes one waveform in memory, writes it to disk, and returns metadata about the generated file.

## `speak_text`

```mermaid
flowchart LR
    A["MCP tool: server.speak_text(...)"] --> B["tools.speak_text(ctx, ...)"]
    B --> C["chunk_text_for_tts(text)"]
    B --> D["runtime.enqueue_speech(...)"]
    D --> E["write detached job payload"]
    E --> F["spawn python -m app.playback_job"]
    F --> G["detached helper process"]
    G --> H["runtime.play_speech_chunks(...)"]
    H --> I["_synthesize_audio_batch(texts=chunks, ...)"]
    I --> J["_model.generate_voice_design(...)"]
    J --> K["list[np.ndarray] waveforms"]
    K --> L["small playback preroll buffer"]
    L --> M["_open_output_stream(...)"]
    M --> N["_write_output_stream_chunk(...) in FIFO order"]
```

`speak_text` is a plain MCP tool. It chunks text only to stay within model-friendly text sizes, then hands the full chunk list to the runtime for detached handoff.

The detached helper process performs one model batch call for the full request, buffers a small initial amount of generated audio, opens one live output stream, and writes the generated waveforms to the host audio device in order.

Important behavior:

- one `speak_text` call creates one detached playback job
- one detached playback job results in one model batch call
- playback uses one output stream per speech request
- playback audio is not persisted to disk
- detached playback survives stdio-session teardown after the MCP call returns

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

Chunking is paragraph-first, with sentence and word fallback only when needed. It is used by `speak_text` to prepare the input list for one batch synthesis request.
