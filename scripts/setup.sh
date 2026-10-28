#!/usr/bin/env bash
# Environment setup for Claude Code cloud sessions (Linux). Installs the package and dev tools.
set -euo pipefail
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev,analysis]"
