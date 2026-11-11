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

## S6 with loose deadlines (QUESTIONS Q60): rerun S6 alone

```bat
uv run python train.py --maps vfc --scenarios S6 --windows 6 --population 40 --generations 20 --budget 1000000 --archive-label tight
uv run python evaluate.py --maps vfc --scenarios S6
```

Training: about 8–13 h. NSGA-II's 6 S6 windows take about 40 min; the 6 methods train in parallel, the D3QN
family in 3–4.5 h and Tang & Wong and QECO in 8–13 h (they took 11 and 13 h next to 10 other jobs). Evaluation:
about 2 h (160 runs; QECO's 20 seeds alone take about 1.7 h).

`--scenarios S6` on the finished map is a revision (src/pcnme/pipeline/revision.py): the settings must match the
finished run's (hence the same --windows, --population, --generations, --budget); S6's old windows,
demonstrations and models move to `data/vfc/_archive/S6-tight-<time>/`; w* stays 0.414; S1–S5 are not touched;
the manifest records the revision. It resumes like a full run (same command, same commit). Never add `--fresh`
here (it is refused: it would archive every scenario). The evaluation moves the old S6 rows and runs to
`results/archive/S6-tight/`, and report.md, paired.md and docs/RESULTS_VFC.md show the new S6 and
"S6 tight (archived)". Afterwards: a dated backup (PLAN B1).

## LuST traces only (no training): the map in the viewer

```bat
uv run python train.py --maps lust --until traces --yes
```

About 30 min: SUMO records the crop at three times of day (data/lust/traces/), then the coverage is measured
(data/lust/calibrated.yaml). It is not recorded in the manifest (any commit; later training reuses the traces).
The viewer then shows LuST's traffic (roads, zones, vehicles) until LuST is trained and evaluated.

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
| Testbed runs and fast-mode vs testbed gaps | `data/vfc/testbed/results/<run_id>/`, `data/vfc/testbed/gap-*.md` |

## Testbed (Docker Desktop running; never while train.py or evaluate.py runs)

```bat
uv run --extra testbed python -m pcnme.testbed.runner --scenario S2 --seed 3000 --method greedy --seconds 60 --containers 2
uv run python tools/testbed_gap.py --scenario S2 --method greedy --seed 3000 --seconds 60 --cars 40
```

The first runs the testbed at 40 cars (2 car containers; the per-zone load is the 200-car one); the second runs
fast mode on the same seed and fleet and writes the gap table (`data/vfc/testbed/gap-*.md`). `--method` takes any
method (the trained ones from `data/vfc/train/`). `PCNME_PROFILE=1` writes each zone's profile next to the runs.

The whole batch (every method, S1-S6, 5 seeds, each next to its fast-mode twin; resumable; about 13 h):

```bat
uv run python run_testbed.py --maps vfc
```

Results: `data/vfc/testbed/summary.csv`, `gap.md`, `gap.csv`. Each batch starts with a pre-flight check (one short
S2 run per method) and refuses to start if a zone crashed, a learning zone made no online updates or more than
2 % of the tasks ended without an ack.

After the 11 Nov fixes (the learning methods ran frozen on the testbed before; a long-run slowdown), first archive
the frozen rows once, then rerun:

```bat
uv run python run_testbed.py --maps vfc --archive-frozen
uv run python run_testbed.py --maps vfc --scenarios S2 S3 --methods pcnme greedy qeco d3qn-greedy --redo
```

The second is 40 runs (2 scenarios x 4 methods x 5 seeds, about 2.5 h plus the pre-flight); `--redo` moves the
earlier Greedy S2/S3 rows to `archive/superseded/`.

## Live: watch the testbed and the training as they go

```bat
uv run python run_live.py --scenario S2 --method pcnme --minutes 10
uv run python view.py
```

The first runs one long testbed run with live points on: every zone pushes its closed tasks, slots, cars and its
agent's online learning to InfluxDB once a second. Watch it in the viewer (source "Live testbed": the city, the
tasks, the statistics and each zone's learning, a few seconds behind real time) or in Grafana
(http://localhost:3000/d/pcnme-live, user admin, password GRAFANA_PASSWORD in docker/.env). It is not a
measurement (the live points cost the zones a little CPU); its rows go to `data/vfc/testbed/live/results/`.
InfluxDB and Grafana stay up afterwards; stop them with
`docker compose -f docker/compose.yaml --profile testbed down`.

The viewer's source "Training" shows every training of the learning methods (data/<map>/train/): on time, loss,
mean ê and updates per episode, the decision budget and train.log; one that is running (train.py) updates by itself
every few seconds.

## Viewer

```bat
uv run python view.py
uv run python -m pcnme.viewer.export
```

The first opens the viewer (http://127.0.0.1:8765): any finished run, two side by side; the live testbed; the
training. The second writes a copy
that runs on another machine with Python alone: `F:\projects\pcnme-history-viewer` (then `python serve.py`).

## Checks on the results

```bat
uv run python tools/s6_feasibility.py
```

Which S6 tasks could meet their deadline with empty queues, and the GPU capacity S6 asks for (QUESTIONS Q60).
