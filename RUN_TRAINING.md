# Running training, evaluation and the testbed

All commands run in **cmd** from `F:\projects\pcnme-src`. Outputs go to `data/<map>/` (layout: `src/pcnme/paths.py`).
Each script prints its plan and estimate, then waits for Enter. It is resumable (rerun the same command) and refuses
to mix outputs of another config or commit (`--fresh` archives them to `data/_archive/`).

## Setup (once)

```bat
uv sync --locked --all-extras
```

## Checks (2–3 minutes each, write to data\smoke\)

```bat
uv run python train.py --maps vfc --smoke --yes
uv run python evaluate.py --maps vfc --smoke --yes
uv run python train.py --maps lust --smoke --yes
uv run python run_testbed.py --maps vfc --smoke
uv run pytest -q -n auto
```

## VFC (tonight): about 10–11 h of training on 11 workers

```bat
uv run python train.py --maps vfc --windows 6 --population 40 --generations 20 --budget 1000000
```

Then the evaluation: 6 scenarios × 8 methods × 20 seeds, about 2–3 h.

```bat
uv run python evaluate.py --maps vfc
```

For Friday, if short of time, S1–S3 first (the rest later, on the same commit):
`uv run python evaluate.py --maps vfc --scenarios S1 S2 S3`.

## Transfer test: the VFC models on LuST, no LuST training (PLAN T3b)

```bat
uv run python train.py --maps lust --until traces
uv run python evaluate.py --maps lust --models-from vfc
```

The first makes only the LuST traces (about 30–60 min); the second deploys the VFC models and the VFC w* on
LuST. Results: `data/lust/results-transfer/` (transfer.md: each method on VFC vs on LuST).

## LuST, full training (later): the traces first (about 30–60 min), then about 10–11 h

```bat
uv run python train.py --maps lust --windows 6 --population 40 --generations 20 --budget 1000000
uv run python evaluate.py --maps lust
```

To make only the traces: add `--until traces`. To run only NSGA-II and w*: add `--until nsga`.

## Where results land

| What | Where |
| --- | --- |
| NSGA-II fronts, one row per window | `data/<map>/nsga/fronts.csv` |
| w* | `data/<map>/w_star.json` |
| Demonstrations | `data/<map>/demos/` |
| Per method and scenario: episodes.csv, train.log, training.json, trained.pt | `data/<map>/train/<scenario>-<method>/` |
| What was run (commit, config hash, budget, times) | `data/<map>/manifest.json` |
| One row per evaluation run | `data/<map>/results/summary.csv` |
| Tables and figures | `data/<map>/results/report.md`, `*.png` |
| PCNME vs each baseline (paired t test over the seeds) | `data/<map>/results/paired.md`, `paired.csv` |
| Each metric over S1–S6, all methods | `data/<map>/results/across-<metric>.png` |
| Transfer test (VFC models on LuST) | `data/lust/results-transfer/` (`transfer.md`, `transfer-*.png`) |

## Testbed

`uv run python run_testbed.py --maps vfc` stops and lists what is missing (PLAN M9.3, M9.4). It stays paused
while training runs.
