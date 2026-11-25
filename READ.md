# Code guide

## Overview

Vehicles generate tasks (camera frames) and upload them to the fog zone they are in. The zone's **TOF-Broker**
sends tasks no fog node can finish in time to the cloud; for all other tasks, a **D3QN agent** chooses one of six
options (own node, four neighbours, cloud) from 48 inputs. The agent is pretrained with **DQfD** on demonstrations
derived from **NSGA-II** knee-point plans and then learns online.

Experiments run in **fast mode**, a discrete-event simulator that is deterministic for a given seed. The **Docker
testbed** runs the same code over an emulated network. Code comments cite the design sections they implement;
for example `(6A.3)` refers to section 6A.3 of [`docs/final-design.md`](docs/final-design.md).

## Reading order

1. [`docs/final-design.md`](docs/final-design.md) — the specification.
2. [`config/default.yaml`](config/default.yaml) — every parameter and its source.
3. [`RUN_TRAINING.md`](RUN_TRAINING.md) — training, evaluation and the testbed.
4. [`docs/RESULTS.md`](docs/RESULTS.md) — the results.

## Source layout

| Package | Contents | Design |
| --- | --- | --- |
| [`config.py`](src/pcnme/config.py), [`maps.py`](src/pcnme/maps.py), [`paths.py`](src/pcnme/paths.py) | Configuration, map settings, data layout | 18 |
| [`sim/`](src/pcnme/sim/) | Event clock, seeded random streams | 15.2 |
| [`net/`](src/pcnme/net/) | Network links: FIFO pipes, shared radio, delays | 4 |
| [`node/`](src/pcnme/node/) | Run-time table, executor, queues, wait estimates | 8, 14.4 |
| [`zone/`](src/pcnme/zone/) | Fog zone: intake, TOF-Broker, masks, state store | 6, 7, 9 |
| [`car/`](src/pcnme/car/), [`mobility/`](src/pcnme/mobility/) | Vehicles, road grid, LuST traces, mobility predictor | 6A, 10, 16.2 |
| [`bus/`](src/pcnme/bus/) | In-process messaging (fast mode) | 5 |
| [`agent/`](src/pcnme/agent/) | Inputs, reward, D3QN, DQfD, replay, online learning | 11, 13.5 |
| [`expert/`](src/pcnme/expert/) | NSGA-II planning, knee point, demonstrations | 13.2–13.4 |
| [`methods/`](src/pcnme/methods/) | Baselines and training sessions | 16.1 |
| [`metrics/`](src/pcnme/metrics/), [`energy.py`](src/pcnme/energy.py), [`fl/`](src/pcnme/fl/) | Logs and statistics, energy model, federated learning | 12, 13.7, 16.3 |
| [`experiments/`](src/pcnme/experiments/), [`pipeline/`](src/pcnme/pipeline/) | Simulation runs and the pipelines behind the scripts | 16 |
| [`testbed/`](src/pcnme/testbed/) | Docker testbed services | 15.4 |
| [`viewer/`](src/pcnme/viewer/README.md) | Results viewer | — |

Tests in [`tests/`](tests/) mirror this layout. [`tests/zone/test_delivery.py`](tests/zone/test_delivery.py)
traces one task through the whole system by hand.

## Entry points

| Script | Purpose |
| --- | --- |
| `train.py` | NSGA-II planning, demonstrations, pretraining and online training |
| `evaluate.py` | Evaluation of all methods; tables and figures |
| `run_testbed.py` | Evaluation on the Docker testbed |
| `start.py`, `start.bat` | Results viewer (`--live` adds a live testbed run) |
| `stop_all.py`, `stop.bat` | Stops all services and scripts |
