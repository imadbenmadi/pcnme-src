# Home desktop setup (Windows, PowerShell). Run from the repo root on drive F:.
# Requires Python 3.12 installed (py launcher).
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1   then run: pytest -q"
