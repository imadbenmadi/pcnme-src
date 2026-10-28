# PCNME

Predictive task offloading for connected cars across fog zones and the cloud. MSc thesis implementation (Imed-eddin Benmadi, ELTE).

- **Design (the specification):** [`docs/final-design.md`](docs/final-design.md), printable as [`docs/PCNME-Final-Design.pdf`](docs/PCNME-Final-Design.pdf)
- **Build plan:** [`docs/PLAN.md`](docs/PLAN.md)
- **Numbers:** [`config/default.yaml`](config/default.yaml), checked by `python tools/calc_defaults.py`
- **Rules for Claude Code:** [`CLAUDE.md`](CLAUDE.md) and the skills in `.claude/skills/`

## Setup

Windows (home desktop, repo on F:): `scripts/setup-windows.ps1`. Linux or a Claude Code cloud session: `scripts/setup.sh`. Then `pytest -q`.

## Folders that are not specifications

`visual/` and `excalidraw/` are illustrations with example numbers. `reference/vfc/` is the VFC paper's code (no license: read only, never copied).
