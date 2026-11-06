#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/src"
echo "Starting CanopyGIS Server on http://127.0.0.1:8000 ..."
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000
