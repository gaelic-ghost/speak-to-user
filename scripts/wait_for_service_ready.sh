#!/bin/sh
set -eu

TIMEOUT_SECONDS="${1:-120}"
TARGET_URL="${2:-${SPEAK_TO_USER_E2E_BASE_URL:-http://127.0.0.1:8765/mcp}}"

if [ -z "${TARGET_URL}" ]; then
    echo "Target MCP URL must not be empty." >&2
    exit 1
fi

ATTEMPT=0
MAX_ATTEMPTS=$((TIMEOUT_SECONDS / 2))
if [ "${MAX_ATTEMPTS}" -lt 1 ]; then
    MAX_ATTEMPTS=1
fi

while [ "${ATTEMPT}" -lt "${MAX_ATTEMPTS}" ]; do
    RESULT=$(
        SPEAK_TO_USER_WAIT_READY_URL="${TARGET_URL}" uv run python - <<'PY' 2>/dev/null || true
import asyncio
import json
import os

from fastmcp import Client


async def main() -> None:
    async with Client(os.environ["SPEAK_TO_USER_WAIT_READY_URL"], timeout=30) as client:
        result = await client.call_tool("tts_status", timeout=30)
        payload = getattr(result, "structuredContent", None)
        if not isinstance(payload, dict):
            for content in getattr(result, "content", []):
                text = getattr(content, "text", None)
                if isinstance(text, str):
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        payload = parsed
                        break
        if not isinstance(payload, dict):
            print("invalid-payload")
            return
        if payload.get("ready") is True:
            print("ready")
            return
        print(f"not-ready:{payload.get('speech_last_event') or 'unknown'}")


asyncio.run(main())
PY
    )

    if [ "${RESULT}" = "ready" ]; then
        exit 0
    fi

    LAST_RESULT="${RESULT:-unreachable}"
    ATTEMPT=$((ATTEMPT + 1))
    sleep 2
done

echo "Timed out waiting for ${TARGET_URL} to report tts_status.ready=true (${LAST_RESULT})." >&2
exit 1
