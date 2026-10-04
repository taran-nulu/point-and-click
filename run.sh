#!/usr/bin/env bash
# Bootstrap a virtualenv on first run, then hand off to the pnc CLI.
set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
if [ ! -d "$VENV" ]; then
    echo "==> creating virtualenv in $VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/python" -m pip install --quiet --upgrade pip
    echo "==> installing point-and-click (this pulls mediapipe, ~100 MB)"
    "$VENV/bin/python" -m pip install --quiet -e ".[dev]"
fi

exec "$VENV/bin/pnc" "$@"