from __future__ import annotations


def speak_text_process_resource() -> str:
    return """# speak_text process

`speak_text` is the playback-oriented tool for this server.

## What the tool does

1. Accepts `text`, `voice_description`, and optional `language`.
2. Chunks long text into model-friendly segments.
3. Writes one detached playback job payload to disk.
4. Launches a detached local helper process with `python -m app.playback_job <job.json>`.
5. Returns success to the MCP client as soon as that detached helper has been launched.

## What the detached helper does

1. Loads the job payload from disk.
2. Creates its own `TTSRuntime`.
3. Calls the Qwen voice-design model once for the full request chunk list.
4. Buffers a small preroll of generated audio.
5. Streams the generated waveforms through one live local audio output stream.
6. Deletes the job payload file when playback finishes.

## Why the helper exists

With stdio MCP, the server process often exits immediately after the tool result
is returned. If playback stayed inside that server process, local audio could be
cut off before it finished. The detached helper keeps host playback alive after
the stdio MCP session ends, so the MCP call can return quickly without waiting
for full playback.

## Agent guidance

- Use `speak_text` when you want immediate local playback and do not need a saved file.
- Use `generate_audio` only when you explicitly want a retained audio file.
- Do not expect `speak_text` to keep the server process alive for playback.
- Do expect `speak_text` to hand playback off to a detached local helper.
- If playback appears broken, inspect stderr warnings from the model stack and
  confirm a detached `app.playback_job` process is being spawned.
"""
