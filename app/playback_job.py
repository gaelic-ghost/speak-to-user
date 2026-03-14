from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.runtime import TTSRuntime


def main() -> int:
    if len(sys.argv) != 2:
        return 2

    job_path = Path(sys.argv[1]).resolve()
    job_payload = json.loads(job_path.read_text(encoding="utf-8"))

    runtime = TTSRuntime.from_env()
    try:
        runtime.play_speech_chunks(
            chunks=list(job_payload["chunks"]),
            voice_description=str(job_payload["voice_description"]),
            language=str(job_payload["language"]),
        )
    finally:
        runtime.shutdown()
        try:
            job_path.unlink()
        except FileNotFoundError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
