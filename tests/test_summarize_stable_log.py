from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_stable_log.py"
SPEC = importlib.util.spec_from_file_location("summarize_stable_log", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summarize_events_collects_job_metrics(tmp_path: Path) -> None:
    log_path = tmp_path / "stable.stderr.log"
    events = [
        {
            "event": "speech_job_queued",
            "job_id": 7,
            "mode": "clone",
            "chunk_count": 3,
            "chunk_char_count": 240,
        },
        {
            "event": "speech_playback_start_deferred",
            "job_id": 7,
            "mode": "clone",
            "defer_reason": "waiting_for_safe_buffer",
        },
        {
            "event": "speech_wavbuffer_event_received",
            "job_id": 7,
            "mode": "clone",
            "wavbuffer_event": "playback_started",
            "wavbuffer_fields": {"reason": "threshold_met"},
        },
        {
            "event": "speech_chunk_playback_retrying",
            "job_id": 7,
            "mode": "clone",
        },
        {
            "event": "speech_job_metrics_summary",
            "job_id": 7,
            "mode": "clone",
            "chunk_count": 3,
            "chunk_char_count_total": 240,
            "configured_preroll_chunks": 3,
            "effective_preroll_chunks": 2,
            "startup_deferred_count": 1,
            "start_admission_reason": "safe_buffer_threshold_met",
            "required_buffered_audio_seconds_at_start": 18.0,
            "max_buffered_audio_seconds_before_start": 24.5,
            "first_audio_latency_ms": 18000,
            "first_chunk_ready_to_first_audio_ms": 12000,
            "max_chunk_synth_duration_ms": 9000,
            "avg_chunk_synth_duration_ms": 7000.0,
            "max_chunk_audio_seconds": 7.5,
            "avg_chunk_audio_seconds": 5.0,
            "max_synth_to_audio_ratio": 1.8,
            "min_real_time_margin_ms": -3200,
            "playback_retry_occurred": True,
            "wavbuffer_forced_input_completed": False,
            "failure_reason": None,
        },
        {
            "event": "speech_job_completed",
            "job_id": 7,
            "mode": "clone",
            "timestamp": "2026-03-28T05:00:00+00:00",
        },
    ]
    log_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    summary = MODULE.summarize_events(log_path)

    assert summary["invalid_json_lines"] == 0
    assert summary["event_counts"]["speech_job_metrics_summary"] == 1
    assert len(summary["jobs"]) == 1
    job = summary["jobs"][0]
    assert job["job_id"] == 7
    assert job["startup_deferred_count"] == 1
    assert job["defer_reasons"] == ["waiting_for_safe_buffer"]
    assert job["wavbuffer_playback_started_reason"] == "threshold_met"
    assert job["playback_retry_occurred"] is True
    assert job["max_buffered_audio_seconds_before_start"] == 24.5
    assert job["completed_at"] == "2026-03-28T05:00:00+00:00"


def test_summarize_events_respects_from_line(tmp_path: Path) -> None:
    log_path = tmp_path / "stable.stderr.log"
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"event": "speech_job_queued", "job_id": 1, "mode": "clone"}),
                json.dumps({"event": "speech_job_queued", "job_id": 2, "mode": "clone"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = MODULE.summarize_events(log_path, from_line=2)

    assert [job["job_id"] for job in summary["jobs"]] == [2]


def test_summarize_events_respects_max_job_id(tmp_path: Path) -> None:
    log_path = tmp_path / "stable.stderr.log"
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"event": "speech_job_queued", "job_id": 1, "mode": "clone"}),
                json.dumps({"event": "speech_job_queued", "job_id": 2, "mode": "clone"}),
                json.dumps({"event": "speech_job_queued", "job_id": 3, "mode": "clone"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = MODULE.summarize_events(log_path, max_job_id=2)

    assert [job["job_id"] for job in summary["jobs"]] == [1, 2]
