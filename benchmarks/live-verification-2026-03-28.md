# Live Verification 2026-03-28

## Purpose

Record the live stable-service verification that justified the current reply-sized clone defaults on Gale's M4 Pro and preserve the exact evidence used to accept them.

## Accepted Defaults

- `SPEAK_TO_USER_TTS_CHUNK_MAX_CHARS=160`
- `SPEAK_TO_USER_TTS_MAX_NEW_TOKENS=384`
- `SPEAK_TO_USER_TTS_MAX_CHUNK_SYNTH_SECONDS=30.0`
- `SPEAK_TO_USER_TTS_MAX_CHUNK_AUDIO_SECONDS=20.0`
- `SPEAK_TO_USER_PLAYBACK_BACKEND=wavbuffer`
- `SPEAK_TO_USER_WAVBUFFER_QUEUE_DEPTH=8`
- `SPEAK_TO_USER_WAVBUFFER_PREROLL_MODE=buffers`
- `SPEAK_TO_USER_PLAYBACK_PREROLL_CHUNKS=3`
- runtime startup admission policy:
  - 1-chunk replies can start immediately
  - 2- and 3-chunk replies wait until all chunks are buffered
  - 4+-chunk replies defer first audio until buffered chunk count and buffered audio lead clear the observed synth deficit

## Why These Defaults Won

`240`-character chunks still allowed later clone chunks to take 12 to 24 seconds to synthesize while only producing roughly 8 to 12 seconds of audio. That still lost the real-time race and produced `wavbuffer` underruns and `stream_starved` failures on longer replies. Reducing to `160` shrank chunk variance enough that the runtime could make correct admission decisions with reply-sized buffers instead of starting too early and hoping later chunks caught up.

The startup admission logic is the real fix for the long-gap failure mode. The service now prefers holding first audio longer over beginning playback in a state that is predictably going to starve. Short and medium replies buffer fully before playback. Longer replies can still speak without fully synthesizing the entire answer first, but they only start once the buffered lead is materially ahead of the observed synth deficit.

## Probe Texts

### Two-chunk probe

`Short live verification after the second fix. This should now wait until both chunks are buffered if needed, then speak cleanly without the short-reply starvation we saw a moment ago.`

### Three-chunk probe

`Second live verification. This reply is medium length and intentionally crosses several chunks. The goal here is to make sure startup still feels reasonable while the service chooses a safer admission point, and that once speech begins it keeps moving without collapsing into silence between phrases.`

### Five-chunk probe

`Third live verification. This longer assistant-style answer is the one that matters most, because the older builds could begin talking and then fall apart later when the next chunk was not ready in time. This build should either hold startup until it has enough buffered lead to stay smooth, or refuse to start before playback if the later chunks still look too risky. What it should not do is start confidently and then collapse into a long dead air pocket or a stream-starved retry in the middle.`

## Stable Log Window

- Service restart completed before verification
- verification log slice began at stderr line `23418`
- the reusable summary command is:

```bash
uv run scripts/summarize_stable_log.py \
  "$HOME/Library/Logs/speak-to-user.stable.stderr.log" \
  --from-line 23418 \
  --max-job-id 3
```

## Results

### Job 1: two chunks

- chunk count: `2`
- total chars: `182`
- startup deferrals: `1`
- admission reason: `safe_buffer_threshold_met`
- first audio latency: `16803ms`
- first chunk ready to first audio: `12031ms`
- max chunk synth: `11467ms`
- max chunk audio: `7.52s`
- max synth-to-audio ratio: `1.694`
- min real-time margin: `-8667ms`
- retries: `false`
- forced-input-completed: `false`
- result: completed cleanly

Interpretation:
The service waited for both chunks, then started cleanly with `wavbuffer reason="threshold_met"` and completed without underrun or retry. This was the short-reply failure shape that previously still starved after the earlier build.

### Job 2: three chunks

- chunk count: `3`
- total chars: `298`
- startup deferrals: `2`
- admission reason: `safe_buffer_threshold_met`
- first audio latency: `26311ms`
- first chunk ready to first audio: `17002ms`
- max chunk synth: `12056ms`
- max chunk audio: `8.24s`
- max synth-to-audio ratio: `1.534`
- min real-time margin: `-5816ms`
- retries: `false`
- forced-input-completed: `false`
- result: completed cleanly

Interpretation:
The service buffered all three chunks before playback. Even though one chunk still had a negative real-time margin, the reply no longer started early and collapsed later. There were no guardrail failures, underruns, or retries.

### Job 3: five chunks

- chunk count: `5`
- total chars: `494`
- startup deferrals: `4`
- admission reason: `safe_buffer_threshold_met`
- required safe buffer at start: `22.22s`
- max buffered audio before start: `29.28s`
- first audio latency: `50555ms`
- first chunk ready to first audio: `45780ms`
- max chunk synth: `15240ms`
- max chunk audio: `9.2s`
- max synth-to-audio ratio: `2.688`
- min real-time margin: `-13480ms`
- retries: `false`
- forced-input-completed: `false`
- result: completed cleanly

Interpretation:
This is the long-reply proof point. The runtime repeatedly deferred startup while the buffer was still unsafe, including one defer with `buffered_audio_seconds=20.96` and `required_buffered_audio_seconds=22.22`. Only after the fifth chunk was ready and the buffer reached `29.28s` did playback begin. The reply then completed without `stream_starved`, underrun, retry, or forced-start fallback.

## Acceptance Decision

These defaults were accepted because the live stable service completed the short, medium, and long assistant-style probes with:

- zero `stream_starved`
- zero `wavbuffer event=underrun`
- zero playback retries
- zero `forced_input_completed`
- zero guardrail failures

The remaining cost is startup latency on longer replies. That is intentional: the current live policy prefers delayed first audio over audible mid-reply collapse.

## Regression Checklist

Re-run the same three probe texts and compare the resulting log summary whenever:

- chunking defaults change
- Qwen package versions change
- model ids change
- `wavbuffer` is rebuilt or replaced
- launchd playback settings change
- Gale reports new sentence gaps, starvation, retries, or static

Treat any of the following as a regression:

- any `stream_starved`
- any underrun during accepted playback
- any playback retry on reply-sized probes
- any `forced_input_completed` start reason on normal replies
- a short or medium reply beginning playback before all chunks are buffered
