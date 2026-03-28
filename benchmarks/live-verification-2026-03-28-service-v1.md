# Live Verification 2026-03-28 Service-V1

## Purpose

Record the first live stable-service verification after moving buffering authority fully into the Swift `wavbuffer` service and bundling the matching `v0.1.0` binary into `speak-to-user`.

## Release Pair

- `speak-to-user` `v0.4.5`
- `swiftly-play` `v0.1.0`

## Stable Log Window

- verification log slice began at stderr line `24480`
- summary command:

```bash
uv run scripts/summarize_stable_log.py \
  "$HOME/Library/Logs/speak-to-user.stable.stderr.log" \
  --from-line 24480 \
  --max-job-id 3
```

## Probe Texts

The same two-, three-, and five-chunk probe texts from `live-verification-2026-03-28.md` were reused so the comparison stays fair.

## Results

### Job 1: two chunks

- completed cleanly
- `first_audio_latency_ms=16996`
- `first_chunk_ready_to_first_audio_ms=12113`
- `underflow_events=0`
- `start_admission_reason="swift_owned_buffering"`

Interpretation:
The short probe stayed comparable to the previous accepted baseline. Startup timing was effectively unchanged, and there was no underrun or retry.

### Job 2: three chunks

- completed cleanly
- `first_audio_latency_ms=22966`
- `first_chunk_ready_to_first_audio_ms=12373`
- `underflow_events=0`
- `start_admission_reason="swift_owned_buffering"`

Interpretation:
The medium probe was slightly faster than the old baseline and remained clean. Swift-owned buffering hit `threshold_met` without any starvation, retry, or forced-start fallback.

### Job 3: five chunks

- completed cleanly
- `first_audio_latency_ms=18100`
- `first_chunk_ready_to_first_audio_ms=13361`
- `underflow_events=1`
- `start_admission_reason="swift_owned_buffering"`

Interpretation:
The long probe was much faster to first audio than the old Python-admission baseline, but it was not as smooth. The stream hit one real underrun, entered `waiting_for_audio`, then resumed when later validated audio arrived. The important improvement is that playback stayed alive and resumed without a subprocess restart or garbled replay path. The important regression relative to the older accepted no-underrun baseline is that the long probe still paused mid-stream once.

## Comparison To Prior Baseline

Compared with `live-verification-2026-03-28.md`:

- the two-chunk probe is essentially unchanged
- the three-chunk probe is modestly faster
- the five-chunk probe is much faster to first audio (`18100ms` vs `50555ms`)
- the five-chunk probe is not yet equivalently smooth because it recorded one underrun and one live wait-for-audio pause

## Acceptance Read

This service-v1 build proves that the new architecture is more resilient:

- no playback subprocess restarts
- no playback retries
- no guardrail failures
- no garbled restart path
- successful pause-and-resume when generation fell behind

But it does **not** yet re-establish the stricter old benchmark standard of zero underruns on the long probe. Treat this as a mixed result: startup performance improved, resilience improved, but long-form smoothness still needs another pass if zero audible pauses remains the target.
