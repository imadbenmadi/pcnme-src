# PCNME

[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24500522.svg)](https://doi.org/10.5281/zenodo.24500522)
[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24526626.svg)](https://doi.org/10.5281/zenodo.24526626)

**Predictive Cloud-Native Mobile Edge** — predictive task offloading for connected vehicles across fog zones and
the cloud. MSc thesis implementation, Imed-eddin Benmadi, Eötvös Loránd University (ELTE).

A rule-based TOF-Broker sends tasks that no fog node can finish in time to the cloud. In every fog zone, a D3QN
agent places the remaining tasks on its own node, a neighbouring zone or the cloud. The agent is pretrained with
DQfD on NSGA-II knee-point plans and keeps learning online. It is evaluated against five learning baselines and two
rules in a deterministic simulator on two city maps (VFC grid, LuST Luxembourg) and on a Docker testbed.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked --all-extras
```

| Action | Windows | Any system |
| --- | --- | --- |
| Open the results viewer (http://127.0.0.1:8765) | double-click `start.bat` | `uv run python start.py` |
| Viewer with a live testbed run (Docker required) | `start.bat --live` | `uv run python start.py --live` |
| Stop everything | double-click `stop.bat` | `uv run python stop_all.py` |

## Documentation

| | |
| --- | --- |
| [`READ.md`](READ.md) | Code structure and where to start reading |
| [`RUN_TRAINING.md`](RUN_TRAINING.md) | Training, evaluation and testbed commands |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Results for both maps |
| [`docs/final-design.md`](docs/final-design.md) | Design specification |
| [`docs/DATASHEET.md`](docs/DATASHEET.md) | The archived dataset |
| [`docs/METRICS.md`](docs/METRICS.md), [`data/DATA.md`](data/DATA.md) | Log schema and data layout |

## Development

```bash
uv run pytest -q -n auto
uv run ruff check src tests tools
uv run mypy
```

All environments install the exact versions pinned in `uv.lock`.

## Licence and citation

Code: MIT ([`LICENSE`](LICENSE)), [doi:10.5281/zenodo.24526626](https://doi.org/10.5281/zenodo.24526626).
Data: CC BY 4.0 ([`LICENSE-DATA.txt`](LICENSE-DATA.txt)),
[doi:10.5281/zenodo.24500522](https://doi.org/10.5281/zenodo.24500522). See [`CITATION.cff`](CITATION.cff).
