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

## VFC (done 5 Nov): 36 h of training on 11 workers, then 5.5 h of evaluation

```bat
uv run python train.py --maps vfc --windows 6 --population 40 --generations 20 --budget 1000000
```

Then the evaluation: 6 scenarios × 8 methods × 20 seeds (960 runs).

```bat
uv run python evaluate.py --maps vfc
```

Some scenarios only: add `--scenarios S1 S2 S3`. The rest must follow on the same commit: summary.csv refuses
rows of another commit (`--fresh` starts a new one). Results: `docs/RESULTS_VFC.md`, history: `docs/RUN_LOG.md`.

## Transfer test: the VFC models on LuST, no LuST training (PLAN T3b, not run yet)

```bat
uv run python train.py --maps lust --until traces
uv run python evaluate.py --maps lust --models-from vfc
```

The first makes only the LuST traces (about 30–60 min); the second deploys the VFC models and the VFC w* on
LuST. Results: `data/lust/results-transfer/` (transfer.md: each method on VFC vs on LuST).

## LuST, full training (later): the traces first (about 30–60 min), then about 36 h like VFC

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
| Training curves per scenario | `data/<map>/results/training-<S>.png` |
| Transfer test (VFC models on LuST) | `data/lust/results-transfer/` (`transfer.md`, `transfer-*.png`) |
| Testbed runs and fast-mode vs testbed gaps | `data/vfc/testbed/results/<run_id>/`, `data/vfc/testbed/gap-*.md` |

## Testbed (Docker Desktop running; never while train.py or evaluate.py runs)

```bat
uv run --extra testbed python -m pcnme.testbed.runner --scenario S2 --seed 3000 --method greedy --seconds 60 --containers 2
uv run python tools/testbed_gap.py --scenario S2 --method greedy --seed 3000 --seconds 60 --cars 40
```

The first runs the testbed at 40 cars (2 car containers; the per-zone load is the 200-car one); the second runs
fast mode on the same seed and fleet and writes the gap table (`data/vfc/testbed/gap-*.md`). `--method` takes any
method (the trained ones from `data/vfc/train/`); PCNME on the testbed has an open bug (RUN_LOG 5 Nov).
`PCNME_PROFILE=1` writes each zone's profile next to the runs. `run_testbed.py --maps vfc` prints the open items.

## Viewer

```bat
uv run python view.py
uv run python -m pcnme.viewer.export
```

The first opens the replay viewer (http://127.0.0.1:8765, any run, two side by side). The second writes a copy
that runs on another machine with Python alone: `F:\projects\pcnme-history-viewer` (then `python serve.py`).

## Checks on the results

```bat
uv run python tools/s6_feasibility.py
```

Which S6 tasks could meet their deadline with empty queues, and the GPU capacity S6 asks for (QUESTIONS Q60).
