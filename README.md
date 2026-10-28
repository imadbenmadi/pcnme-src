# PCNME

Predictive task offloading for connected cars across fog zones and the cloud. MSc thesis implementation (Imed-eddin Benmadi, ELTE).

| What | Where |
| --- | --- |
| Design (the specification) | [`docs/final-design.md`](docs/final-design.md), printable [`docs/PCNME-Final-Design.pdf`](docs/PCNME-Final-Design.pdf) |
| Build plan, one milestone at a time | [`docs/PLAN.md`](docs/PLAN.md) |
| Every number | [`config/default.yaml`](config/default.yaml), checked by `python tools/calc_defaults.py` |
| Log and metrics schema | [`docs/METRICS.md`](docs/METRICS.md) |
| Design issues found while coding | [`docs/CHANGES.md`](docs/CHANGES.md), [`docs/QUESTIONS.md`](docs/QUESTIONS.md) |
| Rules for Claude Code | [`CLAUDE.md`](CLAUDE.md), skills in `.claude/skills/` |

## Setup

Every environment installs exactly the versions in `uv.lock` (the desktop venv, CI, cloud sessions and every Docker container). Install [uv](https://docs.astral.sh/uv/) first.

**Home desktop (Windows, repo on F:)**

```powershell
winget install astral-sh.uv                # once; then open a new terminal
cd F:\projects\pcnme-src
powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1
.\.venv\Scripts\Activate.ps1
pytest -q
```

The setup installs the `dev`, `analysis` and `ml` extras. PyTorch comes from the lock: the CUDA 12.6 build on Windows (newer CUDA builds may drop Pascal, compute 6.1), the CPU build on Linux, version 2.14.1 on both. Check that it sees the GTX 1070:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"   # True (6, 1)
```

**Claude Code cloud session (Linux):** set the environment's setup script to `bash scripts/setup.sh`.

**Changing a dependency.** Edit `pyproject.toml`, run `uv lock` (or `uv lock --upgrade-package <name>`), commit `uv.lock` with it, in its own PR. `uv sync --locked` (setup scripts, CI, Docker) refuses an out-of-date lock.

## Everyday commands

```
pytest -q                 # fast tests
pytest -m slow -n auto    # statistical tests (M/M/1, Erlang C)
ruff check src tests tools   # lint
mypy                      # types
python tools/calc_defaults.py
pcnme --help              # experiment commands (filled in milestone by milestone)
```

## Monitoring and testbed (Docker Desktop, WSL2 backend)

```
copy docker\.env.example docker\.env      # then change the passwords
docker compose -f docker/compose.yaml --profile monitoring up -d   # Grafana http://localhost:3000, InfluxDB :8086, MinIO :9001
docker compose -f docker/compose.yaml --profile testbed up -d      # the testbed (milestone M9)
```

## Folders that are not specifications

`visual/` and `excalidraw/` are illustrations with example numbers. `reference/vfc/` is the VFC paper's code (no license: read only, never copied).
