#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
LOCK_DIR="${TMPDIR:-/tmp}/speak-to-user-e2e.lock"
E2E_HOST="${SPEAK_TO_USER_E2E_HOST:-127.0.0.1}"
E2E_PORT="${SPEAK_TO_USER_E2E_PORT:-8876}"
E2E_PATH="${SPEAK_TO_USER_E2E_PATH:-/mcp}"
E2E_BASE_URL="http://${E2E_HOST}:${E2E_PORT}${E2E_PATH}"
LOG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/speak-to-user-e2e.XXXXXX")
SERVER_STDOUT="${LOG_DIR}/server.stdout.log"
SERVER_STDERR="${LOG_DIR}/server.stderr.log"
SERVER_PID=""

cleanup() {
    if [ -n "${SERVER_PID}" ]; then
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    rm -rf "${LOCK_DIR}" "${LOG_DIR}"
}

trap cleanup EXIT INT TERM

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "Another speak-to-user e2e run appears to be active."
    echo "Do not run the model e2e suite simultaneously or in parallel on this machine."
    exit 1
fi

if launchctl print "gui/$(id -u)/com.galew.speak-to-user.stable" >/dev/null 2>&1; then
    echo "The live stable service appears to be running."
    echo "Stop it before running the model e2e suite to save unified memory:"
    echo "  launchctl unload ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist"
    echo "Restart it after the suite finishes with:"
    echo "  launchctl load ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist"
    exit 1
fi

echo "Starting a dedicated e2e server at ${E2E_BASE_URL}"
echo "This suite is intentionally sequential. Do not run another model-heavy test job at the same time."

(
    cd "${ROOT_DIR}" && \
        SPEAK_TO_USER_HOST="${E2E_HOST}" \
        SPEAK_TO_USER_PORT="${E2E_PORT}" \
        SPEAK_TO_USER_MCP_PATH="${E2E_PATH}" \
        SPEAK_TO_USER_PLAYBACK_BACKEND="null" \
        uv run python app/server.py >"${SERVER_STDOUT}" 2>"${SERVER_STDERR}"
) &
SERVER_PID=$!

READY=0
ATTEMPT=0
while [ "${ATTEMPT}" -lt 60 ]; do
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        echo "The dedicated e2e server exited before it became ready."
        echo "stderr log:"
        cat "${SERVER_STDERR}" || true
        exit 1
    fi

    if SPEAK_TO_USER_E2E_BASE_URL="${E2E_BASE_URL}" uv run python - <<'PY'
import asyncio
import os

from fastmcp import Client


async def main() -> None:
    async with Client(os.environ["SPEAK_TO_USER_E2E_BASE_URL"], timeout=30) as client:
        await client.call_tool("health", timeout=30)


asyncio.run(main())
PY
    then
        READY=1
        break
    fi

    ATTEMPT=$((ATTEMPT + 1))
    sleep 2
done

if [ "${READY}" -ne 1 ]; then
    echo "Timed out waiting for the dedicated e2e server at ${E2E_BASE_URL}"
    echo "stderr log:"
    cat "${SERVER_STDERR}" || true
    exit 1
fi

echo "Running uv run pytest -m e2e -q"
(
    cd "${ROOT_DIR}" && \
        SPEAK_TO_USER_E2E_BASE_URL="${E2E_BASE_URL}" \
        uv run pytest -m e2e -q -o addopts='-q --strict-markers'
)
