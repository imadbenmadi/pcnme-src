# Training, evaluation and the testbed

All commands run from the repository root. Outputs go to `data/<map>/` (the layout is defined in
`src/pcnme/paths.py` and described in `data/DATA.md`). Each script prints its plan and an estimate, then waits
for Enter (`--yes` skips the prompt). Every script is resumable: rerunning the same command continues where it
stopped. A script refuses to mix its outputs with those of another config or commit; `--fresh` archives the old
outputs to `data/_archive/` and starts again.

## Setup

```bash
uv sync --locked --all-extras
```

## Smoke checks (2–3 minutes each, written to `data/smoke/`)

```bash
uv run python train.py --maps vfc --smoke --yes
uv run python evaluate.py --maps vfc --smoke --yes
uv run python train.py --maps lust --smoke --yes
uv run python run_testbed.py --maps vfc --smoke
uv run pytest -q -n auto
```

## Training and evaluation

One map at a time; replace `vfc` with `lust` for the Luxembourg map.

```bash
uv run python train.py --maps vfc --windows 6 --population 40 --generations 20 --budget 1000000
uv run python evaluate.py --maps vfc
```

Training takes about 36 hours on 11 worker processes: NSGA-II planning, demonstrations, DQfD pretraining and
online training of the six learning methods in all six scenarios. The evaluation then runs 6 scenarios × 8
methods × 20 seeds (960 runs, about 5.5 hours).

- `--scenarios S1 S2 S3` restricts a run to some scenarios; the others must follow on the same commit.
- On LuST, `train.py` first records the SUMO traces (about 30–60 minutes). `--until traces` stops after them,
  `--until nsga` after NSGA-II and w*.
- `--scenarios S6` on a finished map retrains one scenario as a revision: the old windows, demonstrations and
  models move to `data/<map>/_archive/`, and the evaluation moves the old rows to `results/archive/`. Use the
  same `--windows`, `--population`, `--generations` and `--budget` as the original run, and never `--fresh`.

Afterwards, the extra metrics and the results document:

```bash
uv run python tools/extra_metrics.py --maps vfc lust
uv run python tools/results_doc.py
```

## Where the outputs are

| What | Where |
| --- | --- |
| NSGA-II fronts, one row per window; w* | `data/<map>/nsga/fronts.csv`, `data/<map>/w_star.json` |
| Demonstrations | `data/<map>/demos/` |
| Training per scenario and method: curves, logs, weights | `data/<map>/train/<scenario>-<method>/` |
| What was run (commit, config hash, budget, times) | `data/<map>/manifest.json` |
| One row per evaluation run | `data/<map>/results/summary.csv` |
| Tables, paired tests, figures | `data/<map>/results/report.md`, `paired.md`, `*.png` |
| Testbed runs and the testbed–fast-mode comparison | `data/vfc/testbed/results/`, `data/vfc/testbed/gap.md` |
| The results document | `docs/RESULTS.md` |

## Testbed (Docker Desktop; never while `train.py` or `evaluate.py` is running)

The full batch runs every method in all scenarios on 5 seeds, each next to its fast-mode twin (about 13 hours):

```bash
uv run python run_testbed.py --maps vfc
```

The results are `data/vfc/testbed/summary.csv`, `gap.md` and `gap.csv`. Each batch starts with a pre-flight
check (one short S2 run per method) and refuses to continue if a zone crashed, a learning zone made no online
updates, or more than 2 % of the tasks ended without an acknowledgement. `--scenarios`, `--methods` and `--redo`
rerun part of a batch; `--redo` moves the replaced rows to `archive/superseded/`.

A single testbed run and its comparison with fast mode:

```bash
uv run --extra testbed python -m pcnme.testbed.runner --scenario S2 --seed 3000 --method greedy --seconds 60 --containers 2
uv run python tools/testbed_gap.py --scenario S2 --method greedy --seed 3000 --seconds 60 --cars 40
```

The testbed runs 40 cars in 2 car containers, with the per-zone load of the 200-car fleet. Any method can be
used; the learning methods load their trained models from `data/vfc/train/`.

## Live view

```bash
uv run python start.py --live
```

This starts a 10-minute testbed run (`run_live.py`; `--scenario`, `--method` and `--minutes` change it) together
with the viewer. Every zone pushes its tasks, slots, cars and online learning to InfluxDB once a second; the
viewer's "Live testbed" source and Grafana (http://localhost:3000/d/pcnme-live) show them a few seconds behind
real time. A live run is a demonstration, not a measurement: its rows go to `data/vfc/testbed/live/results/`.

## Viewer

```bash
uv run python start.py                    # or view.py; Windows: start.bat
uv run python -m pcnme.viewer.export      # a self-contained copy in ../pcnme-history-viewer
```

The viewer replays any finished run (two side by side), the testbed runs and the training curves. Its
"Training" source follows a `train.py` that is still running.

## Stopping everything

```bash
uv run python stop_all.py                 # Windows: stop.bat; --dry-run prints the plan only
```

## Further checks

```bash
uv run python tools/reproduce.py --maps vfc lust --methods pcnme greedy --seed 3000   # reruns and compares with summary.csv
uv run python tools/s6_feasibility.py     # which S6 tasks can meet their deadline with empty queues
```
