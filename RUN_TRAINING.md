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

## LuST (tomorrow): the traces first (about 30–60 min), then about 10–11 h

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

## Testbed

`uv run python run_testbed.py --maps vfc` stops and lists what is missing (PLAN M9.3, M9.4). It stays paused
while training runs.
