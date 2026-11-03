# First fast-mode results (for the 6 Nov meeting)

Status: **in progress** (started 3 Nov). The tables and figures are added here when the runs finish.

## What was done, in order

| Step | What | Where it ran | Output |
| --- | --- | --- | --- |
| 0 | Calibration: YOLO run times and energy on the GTX 1070, JPEG sizes on BDD100K; config updated (QUESTIONS Q52) | Windows | `docs/calibration/`, `config/default.yaml` |
| 1 | Cloud time-table rows from Ultralytics' published numbers + measured pre/post-processing | Windows | `config/default.yaml` |
| 2 | Front check: one S2 window at the full budget (100 × 100): **28 distinct front points** (≥ 5 needed), greedy dominates none | WSL | `results/nsga-S2-w0-full.txt` (WSL) |
| 3 | NSGA-II expert plans: S1–S6, 6 windows each, population 40 × 20 generations (13.2) | WSL | `results/nsga-first/` (WSL) |
| 4 | w* = median knee weight over the 27 windows with a knee = **0.414** (13.3), written to `config/default.yaml` (`reward.w`) | by hand | config |
| 5 | Demonstrations: every front plan replayed (13.4): 27 windows, 89 134 transitions (96 % expert) | WSL | `results/demos-first/` |
| 6 | Rules: greedy and random, S1–S6, 20 evaluation seeds each | Windows | `results/first/` |
| 7 | Learning methods: PCNME (DQfD pretraining on the demonstrations, then online), D3QN from scratch, D3QN + greedy demonstrations, VFC PPO, Tang & Wong, QECO; S1–S6; each trains on 1 M decisions, then 20 evaluation seeds | Windows | `results/first/` |
| 8 | Table + training curves (after step 7) | Windows | this file, `docs/figures/` |

`results/` paths are under `PCNME_DATA_DIR` = `F:\projects\pcnme-data` on Windows, and under `~/pcnme-src` in WSL.

## Reduced budget (state it in the meeting)

The full budget (final-design 18) does not fit before Friday on one desktop, so these are **first results**:

| | Full design | First results |
| --- | --- | --- |
| NSGA-II windows per scenario | 20 | 6 |
| NSGA-II population × generations | 100 × 100 | 40 × 20 |
| Training budget per learning method | 8.75 M decisions | 1 M decisions (the same for every method, 16.1) |
| Evaluation seeds | 20 | 20 (3000–3019) |
| Knee reference (16.1) | S2, S4, 5 seeds | not run yet |

Known limits: NSGA-II fronts get thin under heavy load (S4–S6: 1–3 points in most windows), so S6 has demonstrations from only 1 window. The testbed (M9) runs at a reduced scale later; it only gives the fast-mode vs testbed gap.

## Every command

All Windows commands run in **cmd** from `F:\projects\pcnme-src`. First, once per window:

```bat
set PCNME_DATA_DIR=F:\projects\pcnme-data
```

**See where the runs are** (safe any time; it only reads files):

```bat
uv run python tools/status.py
```

**Step 2, front check** (WSL, about 40 min):

```bash
cd ~/pcnme-src && .venv/bin/python tools/nsga_runs.py --scenario S2 --windows 1 --population 100 --generations 100 --workers 12 --mode population
```

**Step 3, NSGA-II expert plans** (WSL, about 1.5 h): for each scenario S1 … S6 (two at a time):

```bash
cd ~/pcnme-src && .venv/bin/python tools/nsga_runs.py --scenario S1 --windows 6 --population 40 --generations 20 --workers 6 --out results/nsga-first
```

Each run prints every window's front, its knee and w, and the median w* of that scenario (`results/nsga-first/S1.txt`).

**Step 4, w\***: the median of the 27 knee weights printed in step 3 (windows with a single front point have no knee). Set by hand in `config/default.yaml`, `reward.w: 0.414`.

**Step 5, demonstrations** (WSL, about 1 h), then copy them to Windows:

```bash
cd ~/pcnme-src && .venv/bin/python tools/make_demos.py --nsga results/nsga-first --out results/demos-first
```

```bash
mkdir -p /mnt/f/projects/pcnme-data/results/demos-first && cp ~/pcnme-src/results/demos-first/*.parquet /mnt/f/projects/pcnme-data/results/demos-first/
```

**Step 6, rules** (Windows, about 2 h):

```bat
uv run python tools/first_results.py --scenarios S1,S2,S3,S4,S5,S6 --methods greedy,random --parallel 3
```

**Step 7, learning methods** (Windows, about 13 h; two windows in parallel):

```bat
uv run python tools/first_results.py --scenarios S1,S2,S3,S4,S5,S6 --methods d3qn-scratch,vfc-ppo,d3qn-greedy,tang-wong,qeco --parallel 8
```

```bat
uv run python tools/first_results.py --scenarios S1,S2,S3,S4,S5,S6 --methods pcnme --parallel 3
```

`tools/first_results.py` runs, for every scenario and method, `pcnme run --scenario S --method M --seeds 3000-3019 --budget 1000000 --out results/first` (PCNME adds `--demos results/demos-first/S-w*.parquet`; D3QN + greedy adds `--windows 6`). It is **restartable**: a run whose 20 seeds are all there is skipped, so after a stop, run the same command again. A run stopped in the middle of training starts that run over.

**Step 8, table and curves** (after step 7):

```bat
uv run pcnme report %PCNME_DATA_DIR%\results\first --out %PCNME_DATA_DIR%\results\first-report
```

```bat
uv run python tools/plot_training.py %PCNME_DATA_DIR%\results\first --out docs\figures
```

`report.md` gives, per scenario and method: on time, late, dropped, failed (%), mean / p99 / worst-1 % latency, energy per task, refusals, wasted energy, Jain's index, early on-time share and mean ê, as mean ± 95 % interval over the seeds.

**Stop everything** (only if needed; running trainings are lost):

```bat
taskkill /F /IM python.exe
```

## Results

To be filled when step 7 ends.
