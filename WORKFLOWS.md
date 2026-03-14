# Tool Workflows

`WORKFLOWS.md` is the implementation-oriented map of the live MCP tool surface in this repo. It documents the current wired paths only: the FastMCP entrypoints in `app/server.py`, the adapter layer in `app/tools.py`, the runtime in `app/runtime.py`, and the chunking helpers in `app/text_chunking.py`.

## How To Read This

- Diagrams use Mermaid `flowchart LR`.
- Each diagram follows the real call chain from the MCP tool entrypoint down into the runtime/helper functions it depends on.
- `Input -> Output` notes call out the points where data shape or type changes.
- Error notes distinguish between tools that convert exceptions into error payloads and tools that let exceptions bubble.

## Shared Lifespan Flow

```mermaid
flowchart LR
    A["FastMCP startup"] --> B["app_lifespan()"]
    B --> C["TTSRuntime.from_env()"]
    C --> D["_normalize_device(raw)"]
    D --> E["TTSRuntime(...) instance"]
    E --> F["runtime.start_background_preload()"]
    F --> G["runtime.start_watchdog()"]
    F --> H["spawn preload thread"]
    H --> I["_background_preload_worker()"]
    I --> J["load_model()"]
    B --> K["yield {'runtime': runtime} into lifespan_context"]
    K --> L["tool calls read ctx.lifespan_context['runtime']"]
    B --> M["shutdown path"]
    M --> N["runtime.unload_model(reason='server shutdown')"]
    M --> O["runtime.shutdown()"]
```

### Flow

The server creates one `TTSRuntime` for the process during the FastMCP lifespan hook in `app/server.py`. That runtime is stored in the lifespan context and every tool resolves it later through `ctx.lifespan_context["runtime"]`.

Startup also begins two background behaviors immediately:

- the idle watchdog thread, which periodically checks whether the loaded model has been idle long enough to unload
- the preload thread, which opportunistically runs `load_model()` before the first synthesis call

On shutdown, the server unloads the model and then stops/join the background threads.

### Data Shape Transitions

- environment variables `dict[str, str | None]` -> normalized runtime constructor inputs `str | int | Path`
- runtime constructor inputs -> `TTSRuntime`
- `TTSRuntime` -> lifespan payload `dict[str, TTSRuntime]`
- lifespan payload -> per-tool `Context.lifespan_context`

## `health`

```mermaid
flowchart LR
    A["MCP tool: server.health()"] --> B["tools.health_payload()"]
    B --> C["dict[str, str]"]
```

### Flow

`health` is the shortest path in the repo. The FastMCP entrypoint calls `health_payload()` and returns its result unchanged.

### Data Shape Transitions

- no tool input payload -> health payload `{"status": str, "timestamp": str}`

### Errors

This path does not wrap errors because it only builds a small in-process dictionary.

## `tts_status`

```mermaid
flowchart LR
    A["MCP tool: server.tts_status(ctx)"] --> B["tools.tts_status(ctx)"]
    B --> C["tools._runtime_from_context(ctx)"]
    C --> D["runtime.status()"]
    D --> E["_status_payload_locked()"]
    E --> F["dict[str, object]"]
```

### Flow

`tts_status` resolves the shared runtime from the FastMCP context and returns the runtime status snapshot. The runtime holds the lock while constructing the payload so the fields reflect one coherent state.

### Data Shape Transitions

- `Context` -> `TTSRuntime`
- runtime internal fields -> status payload `dict[str, object]`
- internal `datetime | None` fields -> serialized ISO-8601 strings or `None`
- internal `Path` output directory -> `str`

### Errors

If the runtime is missing from lifespan context, `_runtime_from_context()` raises `RuntimeError`. `tts_status` does not convert that failure into an error payload.

## `load_model`

```mermaid
flowchart LR
    A["MCP tool: server.load_model(ctx)"] --> B["tools.load_model(ctx)"]
    B --> C["tools._runtime_from_context(ctx)"]
    C --> D["runtime.load_model()"]
    D --> E{"preload already running?"}
    E -->|yes| F["wait on _preload_complete"]
    E -->|no| G["continue"]
    F --> G
    G --> H{"model already loaded?"}
    H -->|yes| I["_touch_locked()"]
    H -->|no| J["_load_model_impl()"]
    J --> K["_resolve_device(torch)"]
    K --> L["Qwen3TTSModel.from_pretrained(device_map=resolved_device)"]
    I --> M["_status_payload_locked()"]
    L --> M
    M --> N["dict[str, object]"]
```

### Flow

`load_model` is an adapter-wrapped runtime call. The adapter resolves the runtime, calls `runtime.load_model()`, and catches exceptions so the MCP caller receives a structured error payload instead of a raised exception.

Inside the runtime, `load_model()` either:

- waits for the background preload thread to finish if it is already loading on another thread
- refreshes usage timestamps if the model is already loaded
- or loads the model directly through `_load_model_impl()`

The actual model load path resolves the device and then calls `Qwen3TTSModel.from_pretrained(...)`.

### Data Shape Transitions

- `Context` -> `TTSRuntime`
- runtime state -> load result payload `dict[str, object]`
- `device_preference: str` + torch backend availability -> resolved device `str`
- model identifier `str` + resolved device `str` -> loaded model object in `_model`

### Errors

- adapter behavior: exceptions become `{"result": "error", "loaded": False, "error": str(exc), ...status}`
- runtime behavior: `_load_model_impl()` exceptions are recorded in `last_error` and re-raised

## `unload_model`

```mermaid
flowchart LR
    A["MCP tool: server.unload_model(ctx)"] --> B["tools.unload_model(ctx)"]
    B --> C["tools._runtime_from_context(ctx)"]
    C --> D["runtime.unload_model(reason='manual')"]
    D --> E{"model loaded?"}
    E -->|no| F["_status_payload_locked()"]
    E -->|yes| G["_unload_locked(reason)"]
    G --> H["gc.collect()"]
    G --> I["torch cache cleanup for active backend"]
    F --> J["dict[str, object]"]
    I --> J
```

### Flow

`unload_model` also uses an adapter try/except wrapper. If a model is present, the runtime clears `_model`, records `last_unloaded_at`, runs garbage collection, and empties backend caches for MPS or CUDA when available.

### Data Shape Transitions

- `Context` -> `TTSRuntime`
- runtime loaded model state -> unloaded status payload `dict[str, object]`
- internal unload timestamps `datetime` -> serialized strings in the returned payload

### Errors

- adapter behavior: exceptions become `{"result": "error", "loaded": True, "error": str(exc), ...status}`
- runtime behavior: most unload logic is best-effort and does not raise if `torch` is unavailable during cache cleanup

## `set_idle_unload_timeout`

```mermaid
flowchart LR
    A["MCP tool: server.set_idle_unload_timeout(seconds, ctx)"] --> B["tools.set_idle_unload_timeout(ctx, seconds)"]
    B --> C["tools._runtime_from_context(ctx)"]
    C --> D["runtime.set_idle_unload_timeout(seconds)"]
    D --> E{"seconds > 0?"}
    E -->|yes| F["store idle_unload_seconds"]
    E -->|no| G["raise ValueError"]
    F --> H["_status_payload_locked()"]
    H --> I["adapter adds result/info envelope"]
```

### Flow

The adapter calls the runtime setter and wraps the updated runtime status with a small success envelope. This is one of the clearer examples of a shape change added by the adapter layer.

### Data Shape Transitions

- MCP scalar input `seconds: int` -> runtime state mutation `idle_unload_seconds: int`
- runtime status payload `dict[str, object]` -> adapter result payload `{"result": "success", "info": str, ...status}`

### Errors

- adapter behavior: exceptions become `{"result": "error", "error": str(exc), ...status}`
- runtime behavior: non-positive integers raise `ValueError("seconds must be greater than zero")`

## `generate_audio`

```mermaid
flowchart LR
    A["MCP tool: server.generate_audio(...)"] --> B["tools.generate_audio(ctx, ...)"]
    B --> C["tools._runtime_from_context(ctx)"]
    C --> D["runtime.generate_audio(...)"]
    D --> E["normalize text/voice_description/output_format/language"]
    E --> F["_build_output_path(filename_stem, output_format)"]
    E --> G["_synthesize_audio(text, voice_description, language)"]
    G --> H["load_model() if _model is None"]
    G --> I["_model.generate_voice_design(...)"]
    I --> J["_coerce_waveform(wavs[0])"]
    J --> K["np.ndarray waveform + sample_rate"]
    F --> L["Path output_path"]
    K --> M["sf.write(output_path, waveform, sample_rate)"]
    L --> M
    M --> N["metadata dict[str, object]"]
```

### Flow

`generate_audio` is the explicit file-producing path. The tool adapter resolves the runtime and catches exceptions, but the runtime does the interesting work:

1. validate and normalize user-facing inputs
2. derive the final output path
3. synthesize the waveform in memory through `_synthesize_audio()`
4. write the waveform to disk with `soundfile`
5. return metadata about the generated file

`_synthesize_audio()` is shared with `runtime.speak_text()`, so generation and playback share one synthesis path up to the point where the waveform is either written to disk or sent to `sounddevice`.

### Data Shape Transitions

- MCP tool inputs `text: str`, `voice_description: str`, `language: str`, `output_format: str`, `filename_stem: str | None` -> normalized scalar values
- `language: str` -> normalized language alias `str` such as `English`, `Chinese`, or `Auto`
- `filename_stem: str | None` + `output_format: str` -> `Path`
- model output `(wavs, sample_rate)` -> first waveform object `Any`
- waveform object `Any` -> `np.ndarray[np.float32]`
- `np.ndarray` + `sample_rate: int` + `Path` -> persisted audio file on disk
- runtime synthesis metadata -> adapter-facing result payload `dict[str, object]`

### Errors

- adapter behavior: exceptions become `{"result": "error", "error": str(exc), ...status}`
- runtime validation raises for empty text, empty voice description, or unsupported output format
- synthesis errors set `last_error` before re-raising

## `speak_text`

```mermaid
flowchart LR
    A["MCP tool task: server.speak_text(...)"] --> B["tools.speak_text(ctx, progress, ...)"]
    B --> C["tools._runtime_from_context(ctx)"]
    B --> D["chunk_text_for_tts(text)"]
    D --> E["_chunk_sentences(...) / _chunk_words(...) / _split_long_word(...)"]
    D --> F["list[str] chunks"]
    F --> G["deque[str] chunk_queue"]
    G --> H["async Progress setup"]
    H --> I["loop over FIFO chunks"]
    I --> J["asyncio.to_thread(runtime.speak_text, chunk, ...)"]
    J --> K["runtime.speak_text(...)"]
    K --> L["normalize text/voice_description/language"]
    L --> M["_synthesize_audio(...)"]
    M --> N["np.ndarray waveform + sample_rate"]
    N --> O["sd.play(waveform, sample_rate)"]
    O --> P["sd.wait()"]
    P --> Q["playback_result dict[str, object]"]
    Q --> R["append per-chunk record"]
    R --> S["list[dict[str, object]] generated_chunks"]
    S --> T["aggregate final success payload"]
```

### Flow

`speak_text` is the most layered path in the repo and the only tool registered as a required FastMCP background task.

The adapter layer:

1. resolves the runtime from context
2. chunks the input text
3. initializes FastMCP task progress
4. iterates over the chunk queue in FIFO order
5. hands each chunk to `runtime.speak_text()` through `asyncio.to_thread(...)`
6. accumulates per-chunk playback metadata
7. assembles one aggregate result payload after the last chunk

The runtime path for each chunk:

1. validates and normalizes the chunk input
2. synthesizes audio in memory through `_synthesize_audio()`
3. passes the waveform buffer directly to `sounddevice`
4. waits for playback completion
5. returns chunk playback metadata

Unlike older disk-backed playback designs, this path does not persist temporary audio files.

### Data Shape Transitions

- input `text: str` -> `list[str]` chunk list
- `list[str]` -> `deque[str]`
- `deque[str]` -> one chunk `str` per loop iteration
- async task state -> progress side effects through `Progress.set_total()`, `set_message()`, and `increment()`
- chunk `str` -> worker-thread runtime call through `asyncio.to_thread(...)`
- normalized chunk inputs -> runtime synthesis metadata `dict[str, object]`
- model output `(wavs, sample_rate)` -> `np.ndarray[np.float32]` waveform + `int` sample rate
- waveform buffer + sample rate -> host playback side effect through `sd.play()` / `sd.wait()`
- runtime playback result `dict[str, object]` -> per-chunk record `dict[str, object]` with adapter-added `index`, `text`, and `text_length`
- `list[dict[str, object]]` -> final aggregate success payload `dict[str, object]`
- per-chunk durations `list[float]` -> summed `duration_seconds: float`

### Errors

- adapter behavior: blank input after chunking raises `ValueError("text must not be empty")`
- unlike the synchronous adapter tools, `tools.speak_text()` does not wrap exceptions into `{"result": "error"}` payloads
- runtime validation and synthesis/playback failures therefore bubble out of the task

## Chunking Helper Flow

```mermaid
flowchart LR
    A["chunk_text_for_tts(text, max_chars)"] --> B{"blank or short?"}
    B -->|blank| C["return []"]
    B -->|short enough| D["return [normalized_text]"]
    B -->|too long| E["split paragraphs"]
    E --> F{"paragraph fits?"}
    F -->|yes| G["append to current paragraph chunk"]
    F -->|no| H["_chunk_sentences(paragraph, max_chars)"]
    H --> I{"sentence fits?"}
    I -->|yes| J["append to current sentence chunk"]
    I -->|no| K["_chunk_words(sentence, max_chars)"]
    K --> L{"word fits?"}
    L -->|yes| M["append word"]
    L -->|no| N["_split_long_word(word, max_chars)"]
```

### Flow

The chunker prefers semantic boundaries in this order:

1. keep the entire text intact if it already fits
2. split by paragraph boundaries
3. split oversized paragraphs by sentence boundaries
4. split oversized sentences by words
5. hard-split a single overlong word when no softer boundary remains

This helper is used only by `tools.speak_text()` today, but it is an important part of the end-to-end playback behavior because it determines both progress totals and FIFO chunk order.

### Data Shape Transitions

- raw input `text: str` -> stripped `normalized: str`
- long normalized string -> `list[str]` paragraph candidates
- oversized paragraph `str` -> `list[str]` sentence chunks
- oversized sentence `str` -> `list[str]` word chunks
- oversized word `str` -> `list[str]` hard-split segments
- accumulated mutable `current: str` values -> final `list[str]`
