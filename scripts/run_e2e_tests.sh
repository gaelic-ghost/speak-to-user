#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
LOCK_DIR="${TMPDIR:-/tmp}/speak-to-user-e2e.lock"
E2E_HOST="${SPEAK_TO_USER_E2E_HOST:-127.0.0.1}"
E2E_PORT="${SPEAK_TO_USER_E2E_PORT:-8876}"
E2E_PATH="${SPEAK_TO_USER_E2E_PATH:-/mcp}"
E2E_BASE_URL="http://${E2E_HOST}:${E2E_PORT}${E2E_PATH}"

cleanup() {
    rm -rf "${LOCK_DIR}"
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
    echo "  launchctl bootout gui/\$(id -u) ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist"
    echo "Restart it after the suite finishes with:"
    echo "  launchctl bootstrap gui/\$(id -u) ~/Library/LaunchAgents/com.galew.speak-to-user.stable.plist"
    exit 1
fi

echo "Running uv run pytest -m e2e -q"
(
    cd "${ROOT_DIR}" && \
        SPEAK_TO_USER_E2E_HOST="${E2E_HOST}" \
        SPEAK_TO_USER_E2E_PORT="${E2E_PORT}" \
        SPEAK_TO_USER_E2E_PATH="${E2E_PATH}" \
        uv run pytest -m e2e -q -o addopts='-q --strict-markers'
)
