# Home desktop setup (Windows, PowerShell). Run from the repo root on drive F:.
# Requires Python 3.12 (py launcher: https://www.python.org/downloads/windows/).
$ErrorActionPreference = "Stop"
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev,analysis]"
.\.venv\Scripts\pre-commit install
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")
Write-Host "Done. Open a new terminal, then: .\.venv\Scripts\Activate.ps1 ; pytest -q"
