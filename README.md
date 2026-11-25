# PCNME

[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24500522.svg)](https://doi.org/10.5281/zenodo.24500522)
[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24526626.svg)](https://doi.org/10.5281/zenodo.24526626)

**Predictive Cloud-Native Mobile Edge**: predictive task offloading for connected cars across fog zones and the
cloud. This repository is the implementation behind the MSc thesis of Imed-eddin Benmadi (Eötvös Loránd
University, Budapest).

A fixed rule, the TOF-Broker, sends tasks that no fog node can finish in time to the cloud. A D3QN agent in every
fog zone places the remaining tasks: on its own node, on one of up to four neighbour zones, or in the cloud. The
agent is pretrained with DQfD on the knee-point plans of an NSGA-II latency–energy front and then keeps learning
online. It is evaluated against five learning baselines and two rules, in a deterministic discrete-event
simulator on two city maps (the VFC grid and LuST, Luxembourg) and on a Docker testbed with real MQTT and HTTP.

## Results and data

- **Results:** [`docs/RESULTS.md`](docs/RESULTS.md): tables with 95 % confidence intervals, paired tests,
  training curves, the testbed comparison and the limitations, for both maps.
- **Dataset:** every result, trained agent and log is archived on Zenodo,
  [doi:10.5281/zenodo.24500522](https://doi.org/10.5281/zenodo.24500522). Its contents are described in
  [`docs/DATASHEET.md`](docs/DATASHEET.md).

## Quick start

Install [uv](https://docs.astral.sh/uv/), then from the repository root:

```bash
uv sync --locked --all-extras
uv run python start.py          # the results viewer at http://127.0.0.1:8765 (Windows: double-click start.bat)
uv run python stop_all.py       # stop everything again (Windows: stop.bat)
```

Training, evaluation and the testbed are described step by step in [`RUN_TRAINING.md`](RUN_TRAINING.md).

## Documentation

| Topic | File |
| --- | --- |
| Reading guide: how the code is organised | [`READ.md`](READ.md) |
| Design (the specification the code follows) | [`docs/final-design.md`](docs/final-design.md) |
| Notation and symbols | [`docs/NOTATION.md`](docs/NOTATION.md) |
| Every parameter, with its source | [`config/default.yaml`](config/default.yaml), checked by `tools/calc_defaults.py` |
| Log and metric schema | [`docs/METRICS.md`](docs/METRICS.md) |
| Data folders | [`data/DATA.md`](data/DATA.md) |
| Results viewer | [`src/pcnme/viewer/README.md`](src/pcnme/viewer/README.md) |
| Build plan, open questions, run history | [`docs/PLAN.md`](docs/PLAN.md), [`docs/QUESTIONS.md`](docs/QUESTIONS.md), [`docs/RUN_LOG.md`](docs/RUN_LOG.md) |

## Development

Every environment (desktop, CI, Docker) installs exactly the versions pinned in `uv.lock`.

```bash
uv run pytest -q -n auto          # test suite
uv run pytest -m slow -n auto     # statistical tests (M/M/1, Erlang C)
uv run ruff check src tests tools
uv run mypy
```

On Windows, `scripts/setup-windows.ps1` sets up the virtual environment; PyTorch is the CUDA 12.6 build, which
still supports Pascal GPUs such as the GTX 1070. On Linux, `scripts/setup.sh` installs the CPU build. To change a
dependency, edit `pyproject.toml`, run `uv lock` and commit `uv.lock` in the same change.

The Docker testbed and the monitoring stack (InfluxDB, Grafana) need Docker Desktop: copy `docker/.env.example`
to `docker/.env`, set the passwords, and see `RUN_TRAINING.md`.

`visual/` and `excalidraw/` hold early illustrations with example numbers; `reference/vfc/` is the VFC paper's
code, kept for reading only. None of them is a specification.

## Licence and citation

Code: MIT ([`LICENSE`](LICENSE)), archived as [doi:10.5281/zenodo.24526626](https://doi.org/10.5281/zenodo.24526626) (release `results-v1`). Data: CC BY 4.0 ([`LICENSE-DATA.txt`](LICENSE-DATA.txt)). To cite the
software or the data, see [`CITATION.cff`](CITATION.cff).
