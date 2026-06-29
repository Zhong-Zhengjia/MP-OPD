#!/usr/bin/env bash
# Source this file from Slurm/training scripts to auto-start the math verify HTTP server.
#
# Usage:
#   source "${REPO_ROOT}/math_eval_server/start_math_verify_server.sh"
#   start_math_verify_server || exit 1
#   trap stop_math_verify_server EXIT

_math_verify_server_pid=""

_math_verify_health_ok() {
    local url="$1"
    MATH_VERIFY_HEALTH_URL="$url" python3 -c '
import os
import sys
import urllib.request

try:
    with urllib.request.urlopen(os.environ["MATH_VERIFY_HEALTH_URL"], timeout=3) as resp:
        sys.exit(0 if resp.status == 200 else 1)
except Exception:
    sys.exit(1)
'
}

_math_verify_fetch_health() {
    local url="$1"
    MATH_VERIFY_HEALTH_URL="$url" python3 -c '
import json
import os
import urllib.request

with urllib.request.urlopen(os.environ["MATH_VERIFY_HEALTH_URL"], timeout=3) as resp:
    print(resp.read().decode())
' 2>/dev/null || true
}

start_math_verify_server() {
    local repo_root="${1:-${REPO_ROOT:-}}"
    if [[ -z "${repo_root}" ]]; then
        echo "ERROR: REPO_ROOT is not set" >&2
        return 1
    fi

    if [[ -n "${MATH_VERIFY_SERVER_URL:-}" ]]; then
        local health_url="${MATH_VERIFY_SERVER_URL%/verify}/health"
        if _math_verify_health_ok "${health_url}"; then
            echo "[math-verify] Reusing existing server: ${MATH_VERIFY_SERVER_URL}"
            return 0
        fi
        echo "[math-verify] MATH_VERIFY_SERVER_URL unreachable (${MATH_VERIFY_SERVER_URL}), starting local server ..."
        unset MATH_VERIFY_SERVER_URL
    fi

    export MATH_VERIFY_HOST="${MATH_VERIFY_HOST:-0.0.0.0}"
    if [[ -z "${MATH_VERIFY_PORT:-}" ]]; then
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            export MATH_VERIFY_PORT=$((7683 + SLURM_JOB_ID % 1000))
        else
            export MATH_VERIFY_PORT=7683
        fi
    fi

    export MATH_VERIFY_SERVER_URL="http://127.0.0.1:${MATH_VERIFY_PORT}/verify"
    local health_url="http://127.0.0.1:${MATH_VERIFY_PORT}/health"

    echo "[math-verify] Starting server on port ${MATH_VERIFY_PORT} ..."
    (
        cd "${repo_root}/math_eval_server" || exit 1
        exec python3 -u reward_verify_server.py
    ) &
    _math_verify_server_pid=$!

    local attempt
    for attempt in $(seq 1 120); do
        if _math_verify_health_ok "${health_url}"; then
            echo "[math-verify] Server ready: ${MATH_VERIFY_SERVER_URL}"
            _math_verify_fetch_health "${health_url}"
            echo
            return 0
        fi
        if ! kill -0 "${_math_verify_server_pid}" 2>/dev/null; then
            echo "ERROR: math verify server exited before becoming healthy" >&2
            return 1
        fi
        sleep 1
    done

    echo "ERROR: math verify server failed health check: ${health_url}" >&2
    stop_math_verify_server
    return 1
}

stop_math_verify_server() {
    if [[ -n "${_math_verify_server_pid}" ]] && kill -0 "${_math_verify_server_pid}" 2>/dev/null; then
        echo "[math-verify] Stopping server (pid=${_math_verify_server_pid}) ..."
        kill "${_math_verify_server_pid}" 2>/dev/null || true
        wait "${_math_verify_server_pid}" 2>/dev/null || true
    fi
    _math_verify_server_pid=""
}
