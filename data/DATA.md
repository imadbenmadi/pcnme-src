# data/: what each folder is, where it comes from, how to rebuild it

`PCNME_DATA_DIR` defaults to this folder (`src/pcnme/paths.py`). Downloaded maps live in `data/maps/`
(`PCNME_MAPS_DIR`). Big files are git-ignored (`.gitignore`, section `data/`); the small ones that record what
was done are committed. Every output names its producer: the run's `manifest.json` holds the commit, the config
hash, the seeds and the budgets. Commands run in cmd from the repository root.

## Datasets

| Folder | What | Committed? | Source |
| --- | --- | --- | --- |
| `calibration/frames/bdd100k_300/` | 300 BDD100K 720p frames, **for calibration only** (run times, JPEG sizes) | no (17 MB) | Hugging Face mirror `dgural/bdd100k`, the first 300 files of `data/` |
| `calibration/weights/` | YOLOv8 n, n-seg, l weights, as Ultralytics downloads them | no | Ultralytics releases (downloaded by the calibration tool) |
| `maps/lust/LuSTScenario-master/` | the LuST scenario (Codeca et al. 2015): network, 24 h routes, configs; MIT licence | no (~200 MB) | github.com/lcodeca/LuSTScenario |
| VFC grid | built in code (`src/pcnme/mobility/grid.py`, config `cars.grid`) | - | - |

Download LuST again:

```bat
curl -L -o data\maps\lust\LuSTScenario.zip https://github.com/lcodeca/LuSTScenario/archive/refs/heads/master.zip
tar -xf data\maps\lust\LuSTScenario.zip -C data\maps\lust
```

## Calibration (`calibration/`, committed: the measurements; QUESTIONS Q52)

| File | What | Produced by |
| --- | --- | --- |
| `calibration_gtx1070.json` | T1, T3, T2 run times (mean, p5, p95, longest step), GPU energy per run, JPEG sizes | `tools/calibrate_yolo.py`, 3 Nov, GTX 1070 at PCIe x4 |
| `calibration_t4.json` | T4 (yolov8n + EasyOCR recogniser) | same tool, T4 only |
| `cpu-power.md` | the CPU slot's power estimate (no RAPL on Windows) | by hand |

The values are in `config/default.yaml` (`time_table_placeholder`, `hardware`). To measure again, use a separate venv with
the CUDA build of torch, so the project's lock is untouched. These are the commands that produced the JSON files:

```bat
uv venv --python 3.12 data\calibration\venv
uv pip install -p data\calibration\venv torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -p data\calibration\venv ultralytics nvidia-ml-py easyocr
cd data\calibration\weights
..\venv\Scripts\python ..\..\..\tools\calibrate_yolo.py ..\frames\bdd100k_300 ..\calibration_new.json
..\venv\Scripts\python ..\..\..\tools\calibrate_yolo.py ..\frames\bdd100k_300 ..\calibration_t4_new.json T4
```

That takes about 25 min on the GPU; nothing else may run meanwhile. Then copy the numbers into the config and run
`uv run python tools/calc_defaults.py`.

Map calibration of the VFC grid (coverage and hotspot share, config `cars.calibrated`, Q29 / Q30; 30 s):

```bat
uv run python tools/calibrate_map.py
```

## Per map: `data/<map>/` (`vfc`, `lust`), written by train.py and evaluate.py

| Path | What | Committed? | Rebuild |
| --- | --- | --- | --- |
| `manifest.json` | the training run: config hash, commit, plan (windows, budget...), start / end, w* | yes | train.py |
| `nsga/<S>-w<k>.json` | one NSGA-II window: genomes, front, w_p, knee | no | train.py (stage nsga) |
| `nsga/fronts.csv` | one row per window: genes, greedy, front points, distinct, knee, w | yes | train.py |
| `w_star.json` | w* = the median knee weight (13.3) and the knee weights | yes | train.py |
| `demos/nsga-<S>-w<k>.parquet`, `demos/greedy-<S>-w<k>.parquet` | demonstrations: every front plan (PCNME's) / the greedy plan (D3QN + greedy expert's) | no | train.py (stage demos) |
| `demos/demos.csv` | per window: plans, transitions, expert share, the plans' w | yes | train.py |
| `train/<S>-<method>/episodes.csv` | one row per training episode (docs/METRICS.md) | yes | train.py |
| `train/<S>-<method>/train.log`, `training.json` | the job's log; budget, episodes, pretraining agreement | log no, json yes | train.py |
| `train/<S>-<method>/trained.pt`, `model.pt`, `checkpoint.pt` | the trained method (evaluate.py loads it), its weights, the resume checkpoint | no | train.py |
| `results/summary.csv` | one row per evaluation run, every 16.3 metric (docs/METRICS.md) | yes | evaluate.py |
| `results/report.md`, `report.csv`, `<S>.png`, `<S>-load.png` | tables (mean +- 95 % interval) and figures per scenario | yes | evaluate.py |
| `results/paired.md`, `paired.csv`, `across-<metric>.png` | PCNME vs each baseline (paired t test); each metric over S1-S6 | yes | evaluate.py |
| `results/training-<S>.png` | the training curves per scenario | yes | evaluate.py |
| `results/runs/<run_id>/` | each run's Parquet logs and run.json | no | evaluate.py |
| `traces/<time>.npz`, `traces/<time>.json` (lust) | the replayable SUMO traces and their numbers | npz no, json yes | train.py (stage traces) |
| `calibrated.yaml` (lust) | coverage measured on the traces, merged into the map's config | yes | train.py |
| `_archive/<S>-<label>-<time>/` | a revised scenario's old training outputs (train.py --scenarios, Q60) | no | train.py |
| `results/archive/<S>-<label>/` | its old evaluation rows (summary.csv, committed) and runs (not) | csv yes | evaluate.py |
| `testbed/results/<run_id>/` (vfc) | testbed runs and their fast-mode twins | no | pcnme.testbed.runner, tools/testbed_gap.py |
| `testbed/gap-<S>-<method>-<seed>-<cars>cars.md` (vfc) | fast mode vs testbed, one table | yes | tools/testbed_gap.py |

Rebuild everything of one map: `uv run python train.py --maps <map> --fresh ...` (the old outputs go to
`data/_archive/`), then `uv run python evaluate.py --maps <map>`. Exact commands: `RUN_TRAINING.md`.
Only the traces: `uv run python train.py --maps lust --until traces`.
Only NSGA-II and w*: `uv run python train.py --maps vfc --until nsga`.

## Other folders

| Folder | What | Committed? |
| --- | --- | --- |
| `smoke/` | outputs of `--smoke` runs (2–3 min checks) | no |
| `_archive/` | outputs moved aside by `--fresh` | no |

Outside the repository: `F:\projects\pcnme-results-backup\` (a copy of the big ignored files: trained models,
Parquet runs, testbed runs; one folder per date) and `F:\projects\pcnme-history-viewer\` (the viewer export,
`uv run python -m pcnme.viewer.export`).
