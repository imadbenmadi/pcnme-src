# Home desktop setup (Windows, PowerShell). Run from the repo root on drive F:.
# Requires uv (winget install astral-sh.uv). Installs exactly the versions in uv.lock (PLAN M0.4).
$ErrorActionPreference = "Stop"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv not found: winget install astral-sh.uv, then open a new terminal"
}
# The uv cache is on C: and the repo on F:, so hard links are impossible; copy without a warning.
$env:UV_LINK_MODE = "copy"
# --locked: fail instead of changing uv.lock. ml = PyTorch (CUDA 12.6 build) and pymoo, for M5-M6 and calibration.
uv sync --locked --python 3.12 --extra dev --extra analysis --extra ml
.\.venv\Scripts\pre-commit install
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")
Write-Host "Done. Open a new terminal, then: .\.venv\Scripts\Activate.ps1 ; pytest -q"
