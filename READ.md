# Reading guide

How to read this code base, in order. Paths are clickable from the repo root.

## 1. Read first

1. [`docs/final-design.md`](docs/final-design.md): the specification. Every rule in the code cites its section
   (`# final-design 8.4 step 2`, or "(6A.3)" in a docstring).
2. [`config/default.yaml`](config/default.yaml): every number, tagged P (paper), R (reasoning), C (calibrated),
   M (measured; placeholder until calibration). `python tools/calc_defaults.py` prints what follows from them.
3. [`docs/PLAN.md`](docs/PLAN.md): what is built (`[x]`), in progress (`[~]`) and open (`[ ]`).
4. [`docs/QUESTIONS.md`](docs/QUESTIONS.md) and [`docs/CHANGES.md`](docs/CHANGES.md): every gap found while
   coding and Imed's answer (Q1–Q53); every change to the design text (C1–C10, plus the table at the end).
5. [`docs/METRICS.md`](docs/METRICS.md): the Parquet tables every run writes.

## 2. Modules, bottom up

| Package | Files | Design |
| --- | --- | --- |
| Config | [`config.py`](src/pcnme/config.py): loader, checks, derived values (B_WAN, λ_car, E_ref, ē) | 18 |
| Ids | [`ids.py`](src/pcnme/ids.py): TaskId, AttemptKey, snapshot versions | 5.5 |
| Simulation | [`sim/clock.py`](src/pcnme/sim/clock.py) event clock; [`sim/rng.py`](src/pcnme/sim/rng.py) seeded streams, keyed draws; [`sim/quantize.py`](src/pcnme/sim/quantize.py) ns rounding | 15.2, C3 |
| Network | [`net/pipe.py`](src/pcnme/net/pipe.py) FIFO pipe; [`net/radio.py`](src/pcnme/net/radio.py) Shannon rate, shared airtime; [`net/hops.py`](src/pcnme/net/hops.py) per-hop delays; [`net/series.py`](src/pcnme/net/series.py) pipes in series | 4, C1, C6 |
| Node | [`node/timetable.py`](src/pcnme/node/timetable.py) run times (lognormal); [`node/executor.py`](src/pcnme/node/executor.py) queues, slots, skip/abort/cancel, wait estimate; [`node/state.py`](src/pcnme/node/state.py) published slot state | 8, 14.4, C4, C5 |
| Zone | [`zone/zone.py`](src/pcnme/zone/zone.py) the zone (and `Cloud`); [`zone/intake.py`](src/pcnme/zone/intake.py) d_A; [`zone/tof.py`](src/pcnme/zone/tof.py) TOF-Broker; [`zone/masks.py`](src/pcnme/zone/masks.py) option masks and views; [`zone/statestore.py`](src/pcnme/zone/statestore.py) snapshots; [`zone/estimates.py`](src/pcnme/zone/estimates.py) path estimates; [`zone/decision.py`](src/pcnme/zone/decision.py) Greedy; [`zone/layout.py`](src/pcnme/zone/layout.py) zones and neighbours | 6, 7, 9 |
| Car | [`car/car.py`](src/pcnme/car/car.py) tasks, budgets, duplicate filter, confirms | 6A, 9.5 |
| Mobility | [`mobility/grid.py`](src/pcnme/mobility/grid.py) VFC grid, lights, stops; [`mobility/predictor.py`](src/pcnme/mobility/predictor.py) next zone, T_exit | 10 |
| Bus | [`bus/bus.py`](src/pcnme/bus/bus.py) in-process MQTT; [`bus/topics.py`](src/pcnme/bus/topics.py); [`bus/calls.py`](src/pcnme/bus/calls.py) the zone's HTTP calls (fast) | 5 |
| Energy | [`energy.py`](src/pcnme/energy.py) forwarding, execution, return energy | 12 |
| Agent | [`agent/inputs.py`](src/pcnme/agent/inputs.py) the 48 inputs; [`agent/reward.py`](src/pcnme/agent/reward.py) the reward; [`agent/network.py`](src/pcnme/agent/network.py) D3QN; [`agent/losses.py`](src/pcnme/agent/losses.py), [`agent/dqfd.py`](src/pcnme/agent/dqfd.py) DQfD; [`agent/replay.py`](src/pcnme/agent/replay.py) PER; [`agent/online.py`](src/pcnme/agent/online.py) the deployed agent; [`agent/transitions.py`](src/pcnme/agent/transitions.py), [`agent/nstep.py`](src/pcnme/agent/nstep.py) | 11, 13.5 |
| Expert | [`expert/window.py`](src/pcnme/expert/window.py) replay a plan; [`expert/nsga.py`](src/pcnme/expert/nsga.py) NSGA-II; [`expert/knee.py`](src/pcnme/expert/knee.py); [`expert/demos.py`](src/pcnme/expert/demos.py) demonstrations | 13.2–13.4, C9 |
| Baselines | [`methods/simple.py`](src/pcnme/methods/simple.py) random, local, cloud; [`methods/ppo.py`](src/pcnme/methods/ppo.py) VFC PPO; [`methods/lstm.py`](src/pcnme/methods/lstm.py) Tang & Wong, QECO; [`methods/learners.py`](src/pcnme/methods/learners.py) | 16.1 |
| Federated | [`fl/fedavg.py`](src/pcnme/fl/fedavg.py) | 13.7 |
| Metrics | [`metrics/runlog.py`](src/pcnme/metrics/runlog.py) Parquet + line protocol; [`metrics/stats.py`](src/pcnme/metrics/stats.py); [`metrics/report.py`](src/pcnme/metrics/report.py) | 16.3 |
| Experiments | [`experiments/city.py`](src/pcnme/experiments/city.py) the whole city in fast mode; [`experiments/runner.py`](src/pcnme/experiments/runner.py) train + evaluate; [`experiments/cli.py`](src/pcnme/experiments/cli.py) `pcnme`; [`experiments/calibrate.py`](src/pcnme/experiments/calibrate.py) map calibration | 16 |
| Testbed (M9, in progress) | [`testbed/zone.py`](src/pcnme/testbed/zone.py), [`car.py`](src/pcnme/testbed/car.py), [`cloud.py`](src/pcnme/testbed/cloud.py) entry points; [`runner.py`](src/pcnme/testbed/runner.py) one run; [`clock.py`](src/pcnme/testbed/clock.py) real clock; [`mqtt.py`](src/pcnme/testbed/mqtt.py), [`http.py`](src/pcnme/testbed/http.py), [`netloop.py`](src/pcnme/testbed/netloop.py) transport; [`radio.py`](src/pcnme/testbed/radio.py) software radio; [`codec.py`](src/pcnme/testbed/codec.py), [`messages.py`](src/pcnme/testbed/messages.py) payloads; [`docker/`](docker/) | 15.2–15.4, Q53 |

## 3. One task through the code (fast mode)

| Step | Where | Design |
| --- | --- | --- |
| The car creates a task and its budget B_car | `Car._create`, `Car.budget` in [`car/car.py`](src/pcnme/car/car.py) | 6A.1–2 |
| The city picks the serving zone and starts the upload (or holds the frame in a gap) | `City._task`, `City._upload` in [`experiments/city.py`](src/pcnme/experiments/city.py) | 9.7, 10 |
| Upload over the shared radio uplink, task exists at t_rx | `Zone.upload` in [`zone/zone.py`](src/pcnme/zone/zone.py) → `RadioUplink.send` in [`net/radio.py`](src/pcnme/net/radio.py) | 6A.2–3, C6 |
| Intake: d_A, the anchor role | `Zone._intake` → `anchor_deadline` in [`zone/intake.py`](src/pcnme/zone/intake.py) | 6A.3, 6C |
| TOF-Broker: a boulder goes to the cloud with no decision | `TofBroker.is_boulder` / `classify` in [`zone/tof.py`](src/pcnme/zone/tof.py) | 7 |
| Masks and the agent's decision | `views` in [`zone/masks.py`](src/pcnme/zone/masks.py); `Zone._decide` calls `placement.choose`: the agent in [`agent/online.py`](src/pcnme/agent/online.py) (inputs: `build_inputs` in [`agent/inputs.py`](src/pcnme/agent/inputs.py)), or Greedy in [`zone/decision.py`](src/pcnme/zone/decision.py) | 9.1, 11.1–11.2 |
| Send the attempt (backhaul or WAN pipe) | `Zone._send` / `_send_cloud` → `zone.calls.post` in [`bus/calls.py`](src/pcnme/bus/calls.py) | 6C, 9.3 |
| The executor accepts or refuses, queues, runs | `Zone.handle` → `Executor.submit` in [`node/executor.py`](src/pcnme/node/executor.py); run time from [`node/timetable.py`](src/pcnme/node/timetable.py) | 8 |
| The result is published and reaches the car | `publish_report` → `Zone._anchor_result` → `Zone._at_car` → `Car.receive` | 6D, 9.5 |
| The car's confirm becomes an ack; the task closes | `Zone.car_confirmed` → `Zone._anchor_ack` → `Zone._close` | 9.5.2–3 |
| Energy and reward, after the grace G | [`energy.py`](src/pcnme/energy.py); `settle` in [`agent/reward.py`](src/pcnme/agent/reward.py); transitions in [`agent/transitions.py`](src/pcnme/agent/transitions.py) | 11.3–11.4, 12 |

The hand-checked version of this path is [`tests/zone/test_delivery.py`](tests/zone/test_delivery.py)
(`test_a_t1_run_here_by_hand`): read its docstring next to the table.

## 4. Run one small experiment

```bash
uv run pcnme run --scenario S2 --method greedy --seeds 3000 --seconds 5 --cars 60 --out results/try
```

```bash
uv run pcnme report results/try --no-plots --out results/try/report
```

The first writes `results/try/S2-greedy-3000-fast/`; the second writes `report.md` and `report.csv`. Learning
methods (`--method pcnme`, …) train first; see `uv run pcnme run --help`.

## 5. Where results land

- Every run: `$PCNME_DATA_DIR/results/<scenario>-<method>-<seed>-<mode>/` (default `./results`, git-ignored):
  `tasks`, `attempts`, `decisions`, `transitions`, `zones`, `pipes`, `positions`, `events` `.parquet`, plus
  `run.json` and `points.lp` (InfluxDB line protocol, for Grafana only). Schema: [`docs/METRICS.md`](docs/METRICS.md).
- NSGA-II fronts: `results/nsga/<scenario>-w<window>.json` ([`tools/run_nsga.md`](tools/run_nsga.md)).
- Testbed runs: the same folder, `<scenario>-<method>-<seed>-testbed/`, plus `zones/*.json` and
  `containers.log` (`python -m pcnme.testbed.runner`; `tools/testbed_gap.py` compares with fast mode).
- Calibration: [`docs/calibration/`](docs/calibration/) (GTX 1070 measurements, Q52).
- Reported numbers always come from Parquet, never from Grafana.
