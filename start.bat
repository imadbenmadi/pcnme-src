@echo off
rem Double-click to open the PCNME website (no live run). For live: start.bat --live
cd /d "%~dp0"
set PYTHONUTF8=1
uv run python start.py %*
pause
