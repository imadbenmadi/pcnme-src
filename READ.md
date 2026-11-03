# Reading guide

How to understand this code base before Friday. Every path below exists; click it from the repository root.

## 1. The big picture

Cars make computing tasks (camera frames) and upload them to the fog zone they drive in. Each zone decides
where each task runs: on its own node, on one of up to 4 neighbour zones, or in the cloud.

1. The **TOF-Broker** sends tasks that no fog node can finish in time straight to the cloud ("boulders").
2. For every other task ("pebbles"), a **D3QN agent** in the zone chooses one of 6 options from 48 inputs.
3. The agent first learns from an expert: **NSGA-II** finds good plans for short windows of traffic, and their
   knee points become **demonstrations**. It pretrains on them with **DQfD**, then keeps learning online.
4. It is compared with 5 learning baselines and 2 rules (16.1) on scenarios S1–S6 and on two city maps: the VFC
   grid and LuST (Luxembourg, SUMO traces).

Everything runs in **fast mode**: a discrete-event simulator, deterministic per seed. A Docker **testbed**
(M9, paused) re-runs the same code on a real network.

## 2. Read first, in this order

1. [`docs/final-design.md`](docs/final-design.md): the specification. Code cites its sections, e.g. "(6A.3)".
2. [`config/default.yaml`](config/default.yaml): every number, each tagged P (paper), R (reasoning),
   C (calibrated) or M (measured). `uv run python tools/calc_defaults.py` prints what follows from them.
3. [`RUN_TRAINING.md`](RUN_TRAINING.md): how to train and evaluate. [`data/DATA.md`](data/DATA.md): what every
   data file is.
4. [`docs/PLAN.md`](docs/PLAN.md): what is built and what is left. [`docs/QUESTIONS.md`](docs/QUESTIONS.md)
   and [`docs/CHANGES.md`](docs/CHANGES.md): every gap found while coding and your answer to it.
5. [`docs/METRICS.md`](docs/METRICS.md): the logs, `summary.csv` and `episodes.csv` columns.

## 3. Folder map

| Folder | What | Design |
| --- | --- | --- |
| [`src/pcnme/config.py`](src/pcnme/config.py), [`maps.py`](src/pcnme/maps.py), [`paths.py`](src/pcnme/paths.py) | config loader and derived values; a map's config (overrides, hash); the data/<map>/ layout | 18 |
| [`src/pcnme/sim/`](src/pcnme/sim/) | event clock, seeded random streams, ns rounding | 15.2 |
| [`src/pcnme/net/`](src/pcnme/net/) | FIFO pipe, radio (Shannon, shared airtime), per-hop delays | 4 |
| [`src/pcnme/node/`](src/pcnme/node/) | run-time table, executor (queues, slots, refusals, wait estimate) | 8, 14.4 |
| [`src/pcnme/zone/`](src/pcnme/zone/) | the zone: intake, TOF-Broker, masks, state store, Greedy | 6, 7, 9 |
| [`src/pcnme/car/`](src/pcnme/car/) | tasks, budgets, duplicate filter, confirms | 6A, 9.5 |
| [`src/pcnme/mobility/`](src/pcnme/mobility/) | VFC grid, LuST traces ([`lust.py`](src/pcnme/mobility/lust.py)) and replayer ([`traces.py`](src/pcnme/mobility/traces.py)), predictor | 10, 16.2 |
| [`src/pcnme/bus/`](src/pcnme/bus/) | in-process MQTT and HTTP calls (fast mode) | 5 |
| [`src/pcnme/agent/`](src/pcnme/agent/) | 48 inputs, reward, D3QN network, DQfD, replay, online learning | 11, 13.5 |
| [`src/pcnme/expert/`](src/pcnme/expert/) | NSGA-II windows, knee, demonstrations | 13.2–13.4 |
| [`src/pcnme/methods/`](src/pcnme/methods/) | rules and baselines; the resumable training session | 16.1 |
| [`src/pcnme/energy.py`](src/pcnme/energy.py), [`metrics/`](src/pcnme/metrics/), [`fl/`](src/pcnme/fl/) | energy accounting; logs, statistics, report; FedAvg | 12, 16.3, 13.7 |
| [`src/pcnme/experiments/`](src/pcnme/experiments/) | the city (one episode), runner, CLI `pcnme`, map calibration | 16 |
| [`src/pcnme/pipeline/`](src/pcnme/pipeline/) | the stages behind the three root scripts, live screen, checkpoints, manifests | 16 |
| [`src/pcnme/testbed/`](src/pcnme/testbed/), [`docker/`](docker/) | the Docker testbed (M9) | 15.4 |
| [`tests/`](tests/) | mirrors src/; [`tests/zone/test_delivery.py`](tests/zone/test_delivery.py) has hand-checked runs | 17 |
| [`tools/`](tools/) | calc_defaults, calibrate_yolo, calibrate_map, input_ranges, plot_curves, testbed_gap | |
| [`data/`](data/) | calibration, maps, one folder per map ([`data/DATA.md`](data/DATA.md)) | |

## 4. One task through the code (fast mode)

| Step | Where | Design |
| --- | --- | --- |
| The car creates a task and its budget B_car | `Car._create`, `Car.budget` in [`src/pcnme/car/car.py`](src/pcnme/car/car.py) | 6A.1–2 |
| The city picks the serving zone and uploads (or holds the frame in a gap) | `City._task`, `City._upload` in [`src/pcnme/experiments/city.py`](src/pcnme/experiments/city.py) | 9.7, 10 |
| Upload over the shared radio; the task exists at t_rx | `Zone.upload` in [`src/pcnme/zone/zone.py`](src/pcnme/zone/zone.py) → `RadioUplink.send` in [`src/pcnme/net/radio.py`](src/pcnme/net/radio.py) | 6A.2–3 |
| Intake: the deadline d_A, the anchor role | `Zone._intake` → `anchor_deadline` in [`src/pcnme/zone/intake.py`](src/pcnme/zone/intake.py) | 6A.3, 6C |
| TOF-Broker: a boulder goes to the cloud with no decision | `TofBroker.classify` in [`src/pcnme/zone/tof.py`](src/pcnme/zone/tof.py) | 7 |
| Masks, inputs and the decision | `views` in [`src/pcnme/zone/masks.py`](src/pcnme/zone/masks.py); `Zone._decide` asks the placement: the agent in [`src/pcnme/agent/online.py`](src/pcnme/agent/online.py) (`build_inputs` in [`src/pcnme/agent/inputs.py`](src/pcnme/agent/inputs.py)) or Greedy in [`src/pcnme/zone/decision.py`](src/pcnme/zone/decision.py) | 9.1, 11.1–11.2 |
| Send the attempt | `Zone._send` / `Zone._send_cloud` → the calls in [`src/pcnme/bus/calls.py`](src/pcnme/bus/calls.py) | 6C, 9.3 |
| The executor accepts or refuses, queues, runs | `Zone.handle` → `Executor.submit` in [`src/pcnme/node/executor.py`](src/pcnme/node/executor.py); run time from [`src/pcnme/node/timetable.py`](src/pcnme/node/timetable.py) | 8 |
| The result reaches the car | `publish_report` → `Zone._anchor_result` → `Zone._at_car` → `Car.receive` | 6D, 9.5 |
| The confirm becomes an ack; the task closes | `Zone.car_confirmed` → `Zone._anchor_ack` → `Zone._close` | 9.5.2–3 |
| Energy and reward | [`src/pcnme/energy.py`](src/pcnme/energy.py); `settle` in [`src/pcnme/agent/reward.py`](src/pcnme/agent/reward.py) | 11.3–11.4, 12 |

The test `test_a_t1_run_here_by_hand` in [`tests/zone/test_delivery.py`](tests/zone/test_delivery.py) works
through this whole path by hand in its docstring.

## 5. The learning pipeline (train.py)

| Stage | Code | Output in data/<map>/ |
| --- | --- | --- |
| LuST only: SUMO traces, coverage | `stage_traces` in [`src/pcnme/pipeline/train.py`](src/pcnme/pipeline/train.py) → `make_trace` in [`src/pcnme/mobility/lust.py`](src/pcnme/mobility/lust.py); `measure_trace` in [`src/pcnme/experiments/calibrate.py`](src/pcnme/experiments/calibrate.py) | `traces/`, `calibrated.yaml` |
| NSGA-II windows, knee, w* | `stage_nsga` → `discover` in [`src/pcnme/expert/window.py`](src/pcnme/expert/window.py), `run_nsga` in [`src/pcnme/expert/nsga.py`](src/pcnme/expert/nsga.py), `knee` in [`src/pcnme/expert/knee.py`](src/pcnme/expert/knee.py) | `nsga/`, `w_star.json` |
| Demonstrations | `stage_demos` → `demonstrate_front`, `greedy_demonstrations` in [`src/pcnme/expert/demos.py`](src/pcnme/expert/demos.py) | `demos/` |
| DQfD pretraining (PCNME, D3QN + greedy) | `d3qn_session` in [`src/pcnme/methods/learners.py`](src/pcnme/methods/learners.py) → `pretrain` in [`src/pcnme/agent/dqfd.py`](src/pcnme/agent/dqfd.py) | `train/<S>-<method>/train.log` |
| Online training, every learner | `run_session` in [`src/pcnme/methods/learners.py`](src/pcnme/methods/learners.py) → `train_online` in [`src/pcnme/agent/online.py`](src/pcnme/agent/online.py); baselines: `lstm_session` in [`src/pcnme/methods/lstm.py`](src/pcnme/methods/lstm.py), `ppo_session` in [`src/pcnme/methods/ppo.py`](src/pcnme/methods/ppo.py) | `train/<S>-<method>/episodes.csv`, `trained.pt` |

The live screen is [`src/pcnme/pipeline/screen.py`](src/pcnme/pipeline/screen.py); checkpoints are
[`src/pcnme/pipeline/checkpoint.py`](src/pcnme/pipeline/checkpoint.py); the resume and refusal rules are
[`src/pcnme/pipeline/runinfo.py`](src/pcnme/pipeline/runinfo.py).

## 6. Scenarios and config

- **Scenarios** (16.2, config `scenarios`): S1–S3 vary the load (ρ_fog 0.5 / 0.9 / 1.5, loose deadlines);
  S4–S6 use tight deadlines (S6 with mixed fog classes). The load per zone comes from λ_car (18).
- **Maps** (config `maps`, `map`): `vfc` is the grid, built in code. `lust` replays SUMO traces of a
  1.2 × 1.2 km crop, with one time of day per scenario (Q58).
- **w\***: one per map, the median knee weight (13.3). It sets input 48 and the reward's time weight.
- **Mechanism tests** run on the frozen placeholder config [`tests/data/mechanism.yaml`](tests/data/mechanism.yaml),
  so recalibrating never moves them.

## 7. The three scripts

| Script | What | Code |
| --- | --- | --- |
| [`train.py`](train.py) | per map: traces → NSGA-II → w* → demonstrations → pretraining → online (PCNME + 5 baselines) | [`src/pcnme/pipeline/train.py`](src/pcnme/pipeline/train.py) |
| [`evaluate.py`](evaluate.py) | every scenario × method (8) × seed (20) → summary.csv, tables, figures | [`src/pcnme/pipeline/evaluate.py`](src/pcnme/pipeline/evaluate.py) |
| [`run_testbed.py`](run_testbed.py) | the Docker testbed; stops until M9.3 and M9.4 are done | [`src/pcnme/pipeline/testbed.py`](src/pcnme/pipeline/testbed.py) |

## 8. How to read summary.csv

`data/<map>/results/summary.csv` has one row per run: scenario, method, seed, then the metrics.
- **Shares (0–1):** on_time, late, dropped, failed, uncovered.
- **Latency (s):** latency_mean_s, latency_p99_s, latency_worst1_s.
- **Energy:** energy_per_task_j and mean_e_hat (energy relative to running the task locally; lower is
  better, capped at ē).
- **Behaviour:** refusal_rate, wasted_share, jain_load (1 = perfectly even zones), early_on_time (the first
  1,000 decisions).
- **Provenance:** wall_s, commit, config_hash, run_dir.

Compare methods per scenario with report.md beside it (mean ± 95 % interval over the seeds). The training curves
are training-<S>.png.

## 9. How to run (all in cmd, from the repo root; checked 3 Nov)

```bat
uv sync --locked --all-extras
uv run pytest -q -n auto
uv run python train.py --maps vfc --smoke --yes
uv run python train.py --maps vfc --windows 6 --population 40 --generations 20 --budget 1000000
uv run python evaluate.py --maps vfc
```

The full list, with times and outputs, is in [`RUN_TRAINING.md`](RUN_TRAINING.md).
