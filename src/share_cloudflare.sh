#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PORT="${PORT:-8012}"

if command -v cloudflared >/dev/null 2>&1 && [[ "${LOCAL_ONLY:-false}" != "true" ]]; then
    python3 run_uvicorn.py &
    SERVER_PID=$!
    trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT INT TERM
    cloudflared tunnel --url "http://127.0.0.1:${PORT}"
else
    exec python3 run_uvicorn.py
fi
