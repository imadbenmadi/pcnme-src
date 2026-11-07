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
  Parquet runs (20 GB, ignored) backed up to F:\projects\pcnme-results-backup\vfc-2026-11-05\results\.

## 5 Nov: evaluate.py additions (PLAN T3c)

- New: paired.md / paired.csv (PCNME vs each baseline, paired t test over the shared seeds), across-<metric>.png
  (on time, p99, mean ê over S1-S6), and `--models-from vfc` (transfer test into data/lust/results-transfer/,
  transfer.md + figures). Smoke: train/evaluate vfc and evaluate lust --models-from vfc all pass.
- Note: summary.csv refuses rows from another commit. Run all VFC scenarios on one commit (do not pull between
  the S1-S3 and S4-S6 halves), or start again with --fresh.

## 5 Nov: VFC evaluation

- Command: `uv run python evaluate.py --maps vfc` (commit 6bc5126), 13:02 -> about 19:00 (runs ended 18:36, then the
  report over 20 GB of logs). 6 scenarios x 8 methods x 20 seeds = 960 of 960 runs.
- Outputs: data/vfc/results/ (report.md, paired.md, across-*.png, S*.png, training-*.png, summary.csv) in git; the
  Parquet runs (20 GB, ignored) backed up to F:\projects\pcnme-results-backup\vfc-2026-11-05\results\.
- Headline (on time %, PCNME vs best baseline; paired t test, 20 seeds): S1-S2 all methods 99-100 % (ties); S3
  PCNME 98.5 vs 97.2 (D3QN + greedy demos) and 92.3 (greedy), p < 0.01; S4 94.2 vs 93.8; S5 93.3 vs 93.1.
  S6 (the CPU-fog scenario, overloaded: about 43 % of tasks dropped for every method): PCNME 56.8 % on time but
  random 65.6 and VFC-PPO 64.2 are higher, at much more energy (2.6 vs 1.7 J/task) and p99 latency (850 vs 450 ms).
  To explain in the write-up: w* = 0.414 weights energy; in S6 spreading load beats it on on-time share.
- Transfer test (T3b) postponed: no time before Friday.

## 5 Nov: testbed at reduced scale (PLAN M9.3 item 1, M9.4)

- Docker Desktop had crashed ("unable to start"): fixed without a reboot by stopping Docker, `wsl --shutdown`,
  and starting Docker Desktop again.
- Run: `uv run --extra testbed python -m pcnme.testbed.runner --scenario S2 --seed 3000 --method greedy --seconds 60 --containers 2`
  (40 cars; per-zone load as at 200 cars, lambda_car scales). Testbed runs now go to data/vfc/testbed/results/.
- Gap (`uv run python tools/testbed_gap.py --scenario S2 --method greedy --seed 3000 --seconds 60 --cars 40`,
  data/vfc/testbed/gap-S2-greedy-3000.md): on time 0.998 fast mode vs 0.146 testbed. 8 803 tasks; the losses are
  drop_no_ack 49 % (the result or the car's confirm comes too late) and drop_upload_late 25 % (a forwarded
  attempt gets no reply before its cap). By zone: z1 (3 360 tasks, the busiest) 0 % on time, z2 63 %, z3 9 %,
  z4 7 %. From the first 10 s on, so not a slow build-up. The fewer cars did not help: the zone's load stays the
  same (S2 rho 0.9), so the limit is the zone software on one CPU, as at 200 cars.

- Profiled zone-1 (PCNME_PROFILE=1): two hot spots in the HTTP client, both fixed (commit 33f3240):
  1. httpcore ran `import sniffio` on every request; sniffio is not installed, so each request searched the
     whole import path (about 15 s of 60 s). Fixed: sniffio marked absent in sys.modules (no new dependency).
  2. httpx's connection pool re-scanned every connection and polled every socket per request (about 40 s of
     60 s). Replaced by a small keep-alive HTTP/1.1 pool on asyncio streams (testbed/netloop.py).
- After the fixes, Greedy, seed 3000, 60 s:

  | run | fast mode on time | testbed on time | testbed p99 | zone-1 CPU |
  | --- | --- | --- | --- | --- |
  | S1, 40 cars | 99.9 % | 96.9 % (was 60.4 %) | - | about 40 % (was 98 %) |
  | S2, 40 cars | 99.8 % | 95.4 and 96.6 % (two runs; was 14.6 %) | 199-205 ms (fast 171 ms) | about 50 % |
  | S2, 200 cars | 99.8 % | 48.9 % (was 7 % on 3 Nov) | 410 ms | 100 % |
  | S2, 200 cars, 2 CPUs per zone | 99.8 % | 47.8 % | - | 110 % |

  Scale limit: at 200 cars the busiest zone (z1, 37 % of the tasks) is bound by Python's GIL (a second CPU
  does not help); z2-z4 hold 82-89 %. Beyond 40-60 cars needs a zone split over processes (not before Friday).
  The paper states the limit: the testbed runs at 40 cars with the same per-zone load as 200 (lambda_car scales).
- The trained agents now run in the zone containers (M9.1 step 7, Q54: CPU torch in the image, models mounted
  read-only from data/vfc/train/): `--method pcnme` works (zones load the model and decide, about 40 % CPU,
  500 MB each). OPEN BUG: 88 % of tasks end drop_no_ack: the tasks run and the results are pushed to the car,
  but the car's confirm never reaches the anchor (t_result_at_car empty). Greedy on the same build is fine.
  Checked and ruled out: the codec (results and confirms of the agent survive encode/decode), a crashed zone
  loop (no errors in the logs). Next: log the zone's POST /results replies and the MQTT ack path for one
  PCNME run, compare with Greedy.
- Gap tables: data/vfc/testbed/gap-<scenario>-<method>-<seed>-<cars>cars.md.

## 5 Nov (evening): viewer, S6 check, results summary

- Viewer: `uv run python view.py` (src/pcnme/viewer/README.md). Replays any run in data/vfc/results/runs/ and
  data/vfc/testbed/results/ from its logs; no rerun was needed (the S3 runs of every method are there).
  Example: http://127.0.0.1:8765/#vfc/S3/pcnme/3000/fast/greedy.
- S6 check (`uv run python tools/s6_feasibility.py`, read-only): 0.0 % of the 184 020 logged S6 tasks are
  infeasible with empty queues; the hard tasks need 80 % of the 4 GPU slots with 14-20 ms of slack, so an ideal
  shared queue still misses 29-36 % of them. Proposal in QUESTIONS Q60 (loose deadlines for S6, retrain S6 only).
- docs/RESULTS_VFC.md: the one-page summary for the supervisor.

## 6 Nov: docs cleanup (PLAN D1)

- Updated to the code and results: README, READ, RUN_TRAINING (real VFC times, testbed runner, viewer, S6 tool),
  CLAUDE (viewer in the layout), data/DATA (new results files, testbed, backup and export folders), PLAN (T2, M9.3
  items 1-2 and D1 done, M9.4 in progress), visual/city/README (marked as an illustration), run_testbed.py's
  message (and its --force runs now go to data/vfc/testbed/), docker comments and .env.example (no MinIO, Q55).
- Deleted: data/calibration/calc_defaults-placeholders.txt and calc_defaults-first-measurement.txt, snapshots of
  tools/calc_defaults.py from before the applied calibration (82.5 tasks/s per zone; now 43.2). The 3 Nov entry
  above keeps both numbers, and the tool prints the current ones.
- Checked, no change: CHANGES, QUESTIONS, METRICS, data/calibration/cpu-power, .claude/skills, final-design.
- Docker Desktop's data: F:\DockerDesktopWSL, on the internal SSD (F: "ssd2"); the external Seagate USB disk is
  not used.

## 6 Nov: Q60 applied (S6 with loose deadlines)

- config/default.yaml S6 `deadlines: loose`; final-design 16.2 (S6 row), 18 (TOF-Broker in S4 and S5) and its
  change list; QUESTIONS Q60 answered.
- Check that only S6 changed since training: the current config with S6 set back to tight hashes to
  b20e3c962334230c, the trained run's config hash (data/vfc/manifest.json); with loose: 13a2970378845209.
- train.py / evaluate.py `--scenarios S6` on a finished map = a revision (src/pcnme/pipeline/revision.py,
  tests/pipeline/test_revision.py): settings must match the finished run, S6's old outputs archived, w* kept
  (0.414), the manifest records it; evaluate.py archives the old S6 rows/runs to results/archive/S6-tight/ and checks
  only the evaluated scenarios. report.md, paired.md and the generated tables of docs/RESULTS_VFC.md show the new S6
  and "S6 tight (archived)".
- Smoke (data/smoke/vfc): `train.py --maps vfc --scenarios S6 --smoke --archive-label tight` (52 s: revision
  recorded, w* kept, S1 untouched), `evaluate.py --maps vfc --scenarios S6 --smoke` (S1's old rows kept); then the
  S6 rows marked as the old config and evaluated again: 16 rows and their runs archived, report sections S1, S6,
  "S6 tight (archived)".
- To run (tonight, by Imed): RUN_TRAINING.md, section "S6 with loose deadlines".
- The agent: its 48 inputs, 6 actions, masks and reward are unchanged; S6's deadline values change, so the inputs
  that read the deadline (slack, D_norm, the masks' feasibility) see larger values in S6.
- Incident (6 Nov, 09:02): a dry check piped Enter into tonight's train.py command and started the S6 revision.
  Stopped within about 2 minutes, before any S6 window or model finished. It had archived the old S6 outputs (as
  designed); everything was moved back and the manifest, fronts.csv and demos.csv restored from git: 36 NSGA-II
  windows, 63 demonstration files, 36 trained models, as in the backup. It confirmed that the command's settings
  pass the revision check. Lesson: never pipe input into train.py or evaluate.py.

## 6 Nov: plan changes

- The transfer test (VFC models on LuST without LuST training, T3b) is dropped (Imed). Removed from PLAN,
  RUN_TRAINING, data/DATA, RESULTS_VFC and the viewer's LuST page; the entries above stay as history.
  evaluate.py keeps its --models-from option (built and tested on 5 Nov), unused.
- Rehearsal of tonight's S6 revision on a copy of data/vfc (nothing real touched): 6 S6 windows and 6 models
  archived, 30 + 30 kept, w* 0.41389 kept, a resume archives nothing twice; evaluation archives the 160 old S6
  rows and runs, keeps the 800 S1-S5 rows, and queues all 160 new S6 runs.
- PLAN: roadmap R (S6, the VFC testbed to the end with live metrics, Grafana and V3, LuST v1, one platform for both
  maps, LuST v2 as an idea) and backups B (B1: a dated copy after tonight's run).
- Two folders next to the repo that are not data in use: F:\projects\pcnme-data\calib-venv (the CUDA venv used
  for calibration on 3 Nov; only needed to calibrate again) and F:\projects\pcnme-data-old-2026-11-03 (the
  discarded first attempt; RUN_LOG 3 Nov). Both can be deleted by Imed.

## 6 Nov: LuST traces (PLAN L5), no training

- Command: `uv run python train.py --maps lust --until traces --yes` (14:21 -> 14:57, 36 min). Not recorded in
  the manifest (traces-only runs no longer are: later LuST training on any commit reuses them).
- 1.2 km crop, 216 road edges, 45 min simulated per time of day (15 min warm-up + 30 min recorded):

  | time of day | vehicles through the crop | at once (mean / max) | SUMO wall time |
  | --- | --- | --- | --- |
  | off-peak 11:00 | 1 288 | 99 / 157 | 11 min |
  | midday 15:00 | 1 163 | 89 / 128 | 11 min |
  | peak 08:00 | 3 840 | 310 / 398 | 34 min |

- Coverage on the traces (data/lust/calibrated.yaml): off-peak 0.42, midday 0.46, peak 0.87; per mobility class:
  low 0.53, high 0.87 (Q29).
- The viewer's LuST tab now shows each time of day as a traffic preview (the crop's streets from lust.net.xml,
  the zones, SUMO's vehicles; no tasks until LuST is trained). Log: data/lust/logs/traces-2026-11-06.log.

## 6-7 Nov: S6 with loose deadlines (Q60), trained and evaluated

- Training: `train.py --maps vfc --scenarios S6 --windows 6 --population 40 --generations 20 --budget 1000000
  --archive-label tight`, 6 Nov 15:05 -> 7 Nov 03:16 (12.2 h). Revision recorded in the manifest; w* kept
  0.41389; old S6 outputs in data/vfc/_archive/S6-tight-2026-11-06T150527/. All 6 S6 windows gave demonstrations
  (7-9 plans each); PCNME's pretraining agreement 0.62.
- Evaluation: `evaluate.py --maps vfc --scenarios S6`: 160 new rows; 160 old rows and runs archived to
  results/archive/S6-tight/; report.md, paired.md and RESULTS_VFC.md regenerated with "S6 tight (archived)".
  The S6 rows record commit 62304916e198-dirty: config/default.yaml had an uncommitted comment line at the time
  (Imed's note "why 2 cpus 2 gpus", added 6 Nov 16:19); comments do not change the config or its hash, so the
  results are those of commit 6230491's config.
- Result (on time %): Greedy 96.9, D3QN + greedy demos 92.4, PCNME 92.1, D3QN scratch 91.9, VFC PPO 89.3,
  Random 81.0, QECO 80.9, Tang & Wong 79.2. Greedy beats PCNME on on-time, p99 and energy (p < 0.001); PCNME is
  level with the D3QN variants and ahead of the rest. The learners were still improving at the end of the budget.
  Analysis: PLAN T6.
- Backup B1: F:\projects\pcnme-results-backup\vfc-2026-11-07\ (7.9 GB; the new S6 and the archived tight S6).

## 7 Nov: the PCNME testbed bug (PLAN Q1)

- Method: counters at each step of a result's way back to the car (zone.counts, RemoteCar replies, the event
  loop's lateness and slow callbacks; each zone writes counts-<zone>.json), Greedy and PCNME on the same seed
  (S2, seed 3000, 40 cars). Ruled out on the way: the codec (every PCNME bus message encodes), CPU load (zones
  30-60 %), a blocked loop (slow callbacks under 1 s per zone in 90 s).
- Bug 1 (fixed, commit 7a2b307): the cars attached once at start, while PCNME's zones were still loading the
  model; the attach failed and was never retried, so those zones never knew the cars and could not push their
  results. Fix: cars repeat the attach until the zone confirms it; compose starts the cars once every zone is
  healthy. PCNME 11 % -> 79-83 % on time.
- Bug 2 (stand-in, QUESTIONS Q61): a zone loop sometimes 200+ ms late lets T_leave (300 ms) fire before the
  queued GPS; the zone drops a car that never left, the car is not told, and all its later results are lost
  (whole cars, 10-30 % of tasks). Stand-in: the zone's reply to POST /gps names the cars it does not serve and
  they attach again. drop_no_ack 1 238 -> 56.
- Now (S2, seed 3000, 60 s, 40 cars; data/vfc/testbed/gap-S2-pcnme-3000-40cars.md): PCNME 89.8 % on time on the
  testbed against 99.8 % in fast mode; p99 270 ms against 200 ms. The rest is mostly drop_late (7.1 %; Greedy
  1.2 %): results after the deadline, both at home and through neighbours (backhaul hops with netem jitter, Q56).
  A real fast-mode vs testbed gap, not a lost message.

## 7 Nov: S4 check and where the testbed's extra latency goes (before the 240-run batch)

- S4 (T1 deadline 50 ms), seed 3000, 40 cars, 60 s, testbed vs fast mode (on time): Greedy 38.7 vs 94.6 %,
  PCNME 23.3 vs 95.4 %. Tight deadlines collapse on the testbed for both methods.
- Per-hop breakdown (testbed/trace.py, tools/testbed_hops.py; every 5th task; the shared host clock, final-design
  5.4: analysis only), S4 Greedy, medians (p90): upload 7.8 (12.0) ms, decision 0.4, forward and queue 0.2 (9.6),
  run 23.2 (32.9), executor publish 0.2, MQTT publish -> result at the serving zone 15.8 (37.3), push 0.1,
  HTTP to the car 4.7 (6.4); total 59.6 (108.9) ms.
- Avoidable: the MQTT hop. paho kept the default 20-message window for QoS 1 and no TCP_NODELAY (Nagle's
  algorithm meeting delayed acks: the ~40 ms p90). Fixed in testbed/mqtt.py (no window limit, TCP_NODELAY on
  every connection; a transport setting, the agent untouched): MQTT hop 9.2 (12.1) ms, total 49.0 (100.8) ms.
  S4 after the fix: Greedy 66.1 %, PCNME 47.6 % (fast 94.6 / 95.4 %).
- What is left is the testbed's real cost: netem's delays on each crossing (radio 3 +- 2 ms, backhaul 3 +- 2 ms,
  as configured) plus the software of HTTP and MQTT, about 10 ms more than fast mode's modelled hops. With a
  50 ms deadline and a 23 ms run, the median task sits at the deadline: the tight scenarios (S4, S5) and S6 show
  the largest gaps. Reducing it further means changing the transport (persistent sockets instead of HTTP
  requests, fewer broker hops): not cheap, not done.
