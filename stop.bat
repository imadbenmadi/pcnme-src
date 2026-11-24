@echo off
rem Double-click to stop everything PCNME runs: containers, networks, the scripts, the ports (prints the plan first).
rem Options: stop.bat --dry-run  |  --images  |  --volumes
cd /d "%~dp0"
set PYTHONUTF8=1
uv run python stop_all.py %*
pause
