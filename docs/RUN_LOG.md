# Run log: what was run, when, and what came out

Newest at the bottom. Commands: `RUN_TRAINING.md`; data files: `data/DATA.md`.

## 3 Nov 2026: calibration (PLAN C, Q52)

- What: real YOLO on the GTX 1070 (PCIe x4, FP32), 300 BDD100K frames; T1, T3, T2, then T4 (yolov8n + EasyOCR).
  Separate venv with CUDA torch (commands in data/DATA.md).
- Results: T1 22.6 ms, T3 26.5 ms, T4 97.8 ms, T2 3.87 s (GPU slot means); CPU slot T1 69.5, T3 93.3, T4 382 ms;
  energy per run 0.91 / 1.12 / 3.05 / 225 J; JPEG mean 76 KB. Files: data/calibration/*.json.
- Applied to config/default.yaml with your Q52 answers: T2 loose deadline 6 s, mix T1 74.2 / T3 15.2 / T4 10.2 / T2 0.4 %,
  per-row power, CPU slot 41.7 W (estimate). Zone capacity 82.5 → 43.2 tasks/s; B_WAN 515 → 36 Mbps; ē 11.7 → 9.2.
- Cloud rows: Ultralytics' A100 and CPU ONNX numbers + measured pre/post.

## 3 Nov: NSGA-II front check (16.5)

- One S2 window at the full budget (100 × 100), WSL, 12 workers: placeholder config 35 distinct front points (39 min);
  calibrated config 28 of 28 distinct (30 min); greedy dominates none. Passed (≥ 5 needed).

## 3 Nov: first-results attempt (discarded)

- Reduced budget (6 windows, 40 × 20, 1 M decisions) with the old tools; stopped after about 4 h by your decision
  ("restart cleanly"). Outputs moved to F:\projects\pcnme-data-old-2026-11-03 (yours to delete). Not used for results.

## 3 Nov: testbed (M9.1, M9.2; paused)

- Docker tests pass (`uv run pytest -m testbed tests/testbed`, 15 tests). Full load (200 cars) holds only 654 of 9 121
  tasks on time: zone software too slow. Left: a 40–60 car scale, then M9.4 (PLAN M9). No Docker load during training.

## 3 Nov: pipeline smoke checks

- `train.py --smoke` and `evaluate.py --smoke` on vfc and lust, `run_testbed.py --smoke`: all pass. LuST tiny SUMO run:
  216 crop edges; with 5 min warm-up about 85 vehicles in the crop at peak; SUMO about 4× real time.

## 3–4 Nov: VFC training (the real run)

- Command (cmd, repo root):
  `uv run python train.py --maps vfc --windows 6 --population 40 --generations 20 --budget 1000000`
- Commit 85b2fc9, started 3 Nov 23:53, 11 workers. (A duplicate test copy started by mistake was stopped at
  23:46 and its partial outputs moved to data/_archive/vfc-aborted-*; the run above started clean.)
- NSGA-II: 36 windows in 2 h 40 min (23:53 → 02:34), 25.8 core-hours; mean per window S1 30 min, S2 44, S3 77,
  S4 38, S5 39, S6 30. w* = 0.414 (same as on 3 Nov: same seeds, deterministic).
- Learning (still running at 4 Nov ~10:00): D3QN-family jobs take 2–3 h each; the LSTM baselines (Tang & Wong,
  QECO) about 17 decisions/s, 15–18 h each. Expected end: roughly 20–24 h after the 10:00 check.
- Early signals: on-time 0.98–0.998 in S1/S2 training; S4 PCNME pretraining agreement only 0.44 vs 0.78 for
  D3QN + greedy (watch in evaluation).
- Next: `uv run python evaluate.py --maps vfc` after training ends (about 2–3 h); for Friday, S1–S3 first:
  `uv run python evaluate.py --maps vfc --scenarios S1 S2 S3`.

## To do after the VFC run

- Commit the uncommitted docs (this file and any .md edits) only after training ends: resume compares commits.
- Speed-up for the next runs: one PyTorch thread per worker (the 11 workers now compete for 12 threads; the LSTM
  baselines suffer most). Then LuST: `uv run python train.py --maps lust --windows 6 --population 40 --generations 20 --budget 1000000`
  (traces first, about 30–60 min).
- Decided 4 Nov: after VFC evaluation, a transfer test: the VFC-trained models evaluated on LuST without LuST
  training (PLAN T3b). Order: VFC evaluate -> LuST traces only (--until traces) -> evaluate.py --maps lust
  --models-from vfc (option to build first). Full LuST training later, only if time allows.
- Finished 5 Nov 11:52: all 36 jobs trained, 35 h 59 min in total (NSGA-II 2 h 40 min, learning about 33 h). The
  LSTM baselines (Tang & Wong, QECO) were the last to finish. Small result files committed to git; the big ignored
  files (trained.pt, demo parquet, NSGA genomes) backed up to F:\projects\pcnme-results-backup\vfc-2026-11-05.
