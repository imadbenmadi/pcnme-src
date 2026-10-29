#!/usr/bin/env bash
# Environment setup for Claude Code cloud sessions (Linux). Installs exactly the versions in uv.lock (PLAN M0.4).
set -euo pipefail
command -v uv >/dev/null || python3 -m pip install "uv>=0.13.6,<0.14"
# --locked: fail instead of changing uv.lock. uv fetches Python 3.12 if the machine has another version.
uv sync --locked --python 3.12 --extra dev --extra analysis --extra nsga
# Put the venv first on PATH so pytest, ruff and mypy work in later shells without activating it.
line="export PATH=\"$PWD/.venv/bin:\$PATH\""
grep -qxF "$line" "$HOME/.bashrc" 2>/dev/null || echo "$line" >> "$HOME/.bashrc"
