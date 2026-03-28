#!/usr/bin/env -S uv run
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize structured speak-to-user LaunchAgent stderr events for reply playback "
            "verification and regression tracking."
        )
    )
    parser.add_argument("log_path", type=Path, help="Path to the stable or dev stderr log")
    parser.add_argument(
        "--from-line",
        type=int,
        default=1,
        help="Only consider events starting from this 1-based line number",
    )
    parser.add_argument(
        "--max-job-id",
        type=int,
        default=None,
        help="Only include jobs whose numeric job_id is less than or equal to this value",
    )
    return parser.parse_args()


def _job_bucket(job_id: int, mode: str | None) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "mode": mode,
        "queued_at": None,
        "completed_at": None,
        "failed": False,
        "failure_reason": None,
        "chunk_count": None,
        "chunk_char_count_total": None,
        "configured_preroll_chunks": None,
        "effective_preroll_chunks": None,
        "startup_deferred_count": 0,
        "start_admission_reason": None,
        "required_buffered_audio_seconds_at_start": None,
        "max_buffered_audio_seconds_before_start": None,
        "first_audio_latency_ms": None,
        "first_chunk_ready_to_first_audio_ms": None,
        "max_chunk_synth_duration_ms": None,
        "avg_chunk_synth_duration_ms": None,
        "max_chunk_audio_seconds": None,
        "avg_chunk_audio_seconds": None,
        "max_synth_to_audio_ratio": None,
        "min_real_time_margin_ms": None,
        "playback_retry_occurred": False,
        "wavbuffer_forced_input_completed": False,
        "wavbuffer_playback_started_reason": None,
        "underflow_events": 0,
        "guardrail_events": 0,
        "defer_reasons": [],
    }


def summarize_events(
    log_path: Path,
    *,
    from_line: int = 1,
    max_job_id: int | None = None,
) -> dict[str, Any]:
    jobs: dict[int, dict[str, Any]] = {}
    line_count = 0
    invalid_lines = 0
    event_counts: Counter[str] = Counter()

    with log_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line_count = line_number
            if line_number < from_line:
                continue
            stripped = raw_line.strip()
            if not stripped.startswith("{"):
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            event_name = event.get("event")
            if not isinstance(event_name, str):
                continue
            event_counts[event_name] += 1
            job_id = event.get("job_id")
            mode = event.get("mode")
            if isinstance(job_id, int) and max_job_id is not None and job_id > max_job_id:
                continue
            if isinstance(job_id, int):
                bucket = jobs.setdefault(
                    job_id,
                    _job_bucket(job_id, mode if isinstance(mode, str) else None),
                )
                if isinstance(mode, str):
                    bucket["mode"] = mode
            else:
                bucket = None

            if event_name == "speech_job_queued" and bucket is not None:
                bucket["queued_at"] = event.get("timestamp")
                bucket["chunk_count"] = event.get("chunk_count")
                bucket["chunk_char_count_total"] = event.get("chunk_char_count")
            elif event_name == "speech_playback_start_deferred" and bucket is not None:
                bucket["startup_deferred_count"] += 1
                reason = event.get("defer_reason")
                if isinstance(reason, str):
                    bucket["defer_reasons"].append(reason)
            elif event_name == "speech_wavbuffer_event_received" and bucket is not None:
                wavbuffer_event = event.get("wavbuffer_event")
                wavbuffer_fields = event.get("wavbuffer_fields")
                if wavbuffer_event == "playback_started" and isinstance(wavbuffer_fields, dict):
                    reason = wavbuffer_fields.get("reason")
                    if isinstance(reason, str):
                        bucket["wavbuffer_playback_started_reason"] = reason
                        bucket["wavbuffer_forced_input_completed"] = (
                            reason == "forced_input_completed"
                        )
                if wavbuffer_event == "underrun":
                    bucket["underflow_events"] += 1
            elif event_name == "speech_chunk_playback_underflow" and bucket is not None:
                bucket["underflow_events"] += 1
            elif event_name == "speech_chunk_playback_retrying" and bucket is not None:
                bucket["playback_retry_occurred"] = True
            elif event_name == "speech_chunk_synthesis_guardrail_tripped" and bucket is not None:
                bucket["guardrail_events"] += 1
                reason = event.get("reason")
                if isinstance(reason, str):
                    bucket["failure_reason"] = reason
                    bucket["failed"] = True
            elif event_name == "speech_job_metrics_summary" and bucket is not None:
                for key in (
                    "chunk_count",
                    "chunk_char_count_total",
                    "configured_preroll_chunks",
                    "effective_preroll_chunks",
                    "startup_deferred_count",
                    "start_admission_reason",
                    "required_buffered_audio_seconds_at_start",
                    "max_buffered_audio_seconds_before_start",
                    "first_audio_latency_ms",
                    "first_chunk_ready_to_first_audio_ms",
                    "max_chunk_synth_duration_ms",
                    "avg_chunk_synth_duration_ms",
                    "max_chunk_audio_seconds",
                    "avg_chunk_audio_seconds",
                    "max_synth_to_audio_ratio",
                    "min_real_time_margin_ms",
                    "playback_retry_occurred",
                    "wavbuffer_forced_input_completed",
                    "failure_reason",
                ):
                    bucket[key] = event.get(key)
                if bucket["failure_reason"] is not None:
                    bucket["failed"] = True
            elif event_name == "speech_job_completed" and bucket is not None:
                bucket["completed_at"] = event.get("timestamp")
            elif event_name == "speech_job_failed" and bucket is not None:
                bucket["completed_at"] = event.get("timestamp")
                bucket["failed"] = True
                if isinstance(event.get("error"), str):
                    bucket["failure_reason"] = event["error"]

    return {
        "log_path": str(log_path),
        "from_line": from_line,
        "max_job_id": max_job_id,
        "line_count": line_count,
        "invalid_json_lines": invalid_lines,
        "jobs": list(jobs.values()),
        "event_counts": dict(event_counts),
    }


def main() -> int:
    args = parse_args()
    summary = summarize_events(
        args.log_path,
        from_line=args.from_line,
        max_job_id=args.max_job_id,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
