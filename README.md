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

**Home desktop (Windows, repo on F:)**

```powershell
cd F:\pcnme-src
powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1
.\.venv\Scripts\Activate.ps1
pytest -q
```

PyTorch for the GTX 1070 (needed from M5): install a CUDA 12.4 or 12.6 wheel, then check it sees the GPU. Newer CUDA builds may have dropped Pascal (compute 6.1), so verify:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"   # True (6, 1)
pip install -e ".[ml]"
```

**Claude Code cloud session (Linux):** set the environment's setup script to `bash scripts/setup.sh`.

## Everyday commands

```
pytest -q                 # fast tests
pytest -m slow -n auto    # statistical tests (M/M/1, Erlang C)
ruff check src tests      # lint
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
