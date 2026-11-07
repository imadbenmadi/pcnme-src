# PCNME build plan

Build in this order. Each milestone ends with its acceptance tests passing and Imed's review. Do not start a milestone before the previous one is approved. Section numbers refer to `docs/final-design.md`.

Status legend: `[ ]` not started, `[~]` in progress, `[x]` done and approved.

---

## M0 — Project setup

- [x] **M0.1 Packaging and tooling** (done in the starter kit, 28 Oct): `pyproject.toml` with extras `dev`, `ml`, `analysis`, `testbed`, `maps`; pytest markers; ruff, mypy, pre-commit; CI on Linux and Windows. First task of the first session: install, run `pytest -q`, `ruff check src tests` and `mypy`, and fix anything that fails on the real machine.
- [x] **M0.2 Config loader** (`config.py`). Load `config/default.yaml` into typed, frozen dataclasses. Compute the derived values exactly as `tools/calc_defaults.py` does: mean GPU run of the mix, zone capacity, λ_car per scenario, B_WAN, E_ref per type, ē, T2's tight deadline (1.1 × Ê_strong).
  - Test: derived values equal the numbers printed by `tools/calc_defaults.py` (zone capacity 82.5 tasks/s, B_WAN ≈ 515 Mbps, T2 tight D ≈ 0.792 s, …).
  - Test: a missing or misspelled key raises an error naming the key.
- [x] **M0.3 Seeded random streams** (`sim/rng.py`). Every stream derived from one run seed and told apart by the purpose's name (QUESTIONS Q4–Q6, CHANGES C3). Sequential purposes (arrivals, mobility, nsga) get one `numpy.random.Generator` per key, for example per car. Keyed purposes (run times, link delays, radio delays) are drawn by task, not by event order: a uniform in (0, 1) keyed by (task_id, attempt), plus the hop for a delay, so a task keeps the same draws in every NSGA-II plan (13.2). Seed ranges from config (`fast_mode.seed_split`) are enforced.
  - Test: same seed → identical draws; different purposes are independent; keyed draws do not depend on call order; a seed outside its range is refused.

- [x] **M0.4 Lock dependencies** so the venv, CI and every Docker container run identical library versions. Generate a lock file (`uv lock`, or `pip-compile` to `requirements.lock`) on the desktop, commit it, and install from it in `scripts/setup*.{sh,ps1}`, CI and `docker/Dockerfile`. Re-lock only on purpose, in its own PR.

## M1 — Core simulator (one node, pipes)

- [x] **M1.1 Event clock** (`sim/clock.py`). Discrete-event loop: schedule(time, callback), cancel, run_until. Ties broken by a monotonically increasing sequence number (deterministic).
  - Test: events run in time order; equal times run in scheduling order; cancelled events never run.
- [x] **M1.2 FIFO pipe** (`net/pipe.py`), section 4.1: `W_pipe = max(0, t_free − t_now)`, `t_free ← max(t_now, t_free) + S/B`, delivery at `t_now + W_pipe + S/B + d`. The delay does not hold the pipe.
  - Test: hand-computed sequence of 3 transfers (overlapping and not).
  - Test: offered load with TI > 1 → the pipe wait grows linearly over time; TI < 1 → bounded.
- [x] **M1.3 Two pipes in series** (section 4.1): cloud transfer `T = W_bh + d_bh + W_WAN + S/B_WAN + d_WAN`.
  - Test: hand calculation with both pipes busy.
- [x] **M1.4 Time table** (`node/timetable.py`), sections 14.4 and 18: rows per (type, hardware class, slot type) with mean, p5, p95, longest step, energy per run; a missing row = cannot run there. Run time draws use the run-time stream.
- [x] **M1.5 Executor: slots and queues** (`node/executor.py`), sections 8.1–8.3: c_GPU and c_CPU slots, one queue per slot type, capacity Q_q with r places reserved for hard tasks, slot-type choice by earliest estimated finish, refusal "full" / "late".
- [x] **M1.6 Scheduler** (section 8.4): skip hopeless heads, soft promotion after A_max only if every queued hard task still meets its deadline, then hard, then soft, then work stealing. Order inside a class: EDF on d_k, ties by arrival order then task id (section 8.2).
- [x] **M1.7 Drops and aborts** (section 8.6): abort at d_k (hard) or d_k + L_max (soft); the slot stays busy for min(remaining, d_abort); every drop reported at once with the energy spent; (attempt key, outcome) kept for T_keep.
- [x] **M1.8 Published state and wait estimate** (section 8.5): c_q, Q_q, n_hard, n_soft, B_hard, B_soft, R, u per slot type; Ŵ = 0 while a slot is free, otherwise the pooled formula.
- **M1 acceptance tests** (section 16.5 and 8.7), all passing and approved 28 Oct (`tests/acceptance/`):
  - M/M/1: one slot, one class, FIFO order, Poisson arrivals, exponential run times, ρ = 0.5 and 0.8, ≥ 200 000 tasks: mean wait within 3 % of λ/(μ(μ − λ)).
  - M/M/c: c = 2 and 4 against Erlang C, same tolerance.
  - One unit test per scheduler rule in 8.4 and per row of the table in 8.6.
  - Ŵ: 0 with a free slot; equals the formula otherwise; c_q = 1 exact in the mean.
  - Determinism: two runs with the same seed give the same event-log hash.

## M2 — One zone, end to end (no mobility, Greedy)

- [x] **M2.1 Identifiers** (`ids.py`), section 5.5: task_id (car_id, epoch, seq), attempt key (task_id, anchor_id, attempt), snapshot version (epoch, counter) compared epoch first.
- [x] **M2.2 Cars and tasks** (`car/`), sections 6A, 14.1–14.3: task types and mix, deadlines (loose / tight), sizes, arrivals at λ_car, the car's latency clock.
- [x] **M2.3 Radio uplink** (`net/radio.py`), sections 4.2 and 18: Shannon rate from distance (free-space path loss, 5.9 GHz, correct units, see CHANGES C1), one car at a time per zone.
- [x] **M2.4 Intake and anchor deadline**, section 6A: task exists only after the whole upload; `d_A = t_rx + B_car − Û_up`.
- [x] **M2.5 TOF-Broker** (section 7.1), including the boulder drop test.
- [x] **M2.6 Budgets and delivery** (sections 6C–6E): B_k, d_k, result path delays, the car keeps the first result and confirms.
- [x] **M2.7 Energy accounting** (`energy.py`), section 12: E_k = forwarding + execution + return, PUE, E_ref, ê = min(E/E_ref, ē), the per-attempt table in 12.5.
- [x] **M2.8 Greedy and Random** (section 16.1) over the options available in one zone (here, cloud).
- [x] **M2 acceptance tests:**
  - Hand calculation (16.5): one task, empty system, fixed numbers, once per destination (here, cloud): simulator within 1 %.
  - Energy: E_ref and ê for each task type equal `tools/calc_defaults.py`.
  - TOF: in tight deadlines T2 is a boulder; in loose it is not; a boulder whose cloud path cannot make it is dropped at intake with no transition.
  - Cloud-only at TI_WAN 1.5: the WAN queue grows without limit (VFC Fig. 6 behaviour); at 0.5 it stays bounded.

## M3 — Many zones, mobility, failures (fast mode complete, Greedy)

- [x] **M3.1 Bus and calls in fast mode** (`bus/`), sections 5.1–5.3, 15.2: in-process topic table with modelled broker delay; QoS 1 may deliver twice; retained messages; last will. Direct calls after the modelled transfer time.
- [x] **M3.2 Layout and neighbours**, sections 7.5 and 18: 5 zone positions, R_z, d_nb, neighbour positions fixed at start, the join rule.
- [x] **M3.3 Zone state plane**, sections 5.2, 9.2: state every 100 ms, snapshot versions, newest wins, staleness, R1 pending sends, the WAN-wait estimate (4.3).
- [x] **M3.4 Masks** (section 7.2) and q*.
- [x] **M3.5 Placement to neighbours and cloud**, refusals and the one re-decision (9.1, 9.2).
- [x] **M3.6 Loss detection, retry, cancel** (9.3, 9.4): upload timeout, last will, watchdog, status probe, retry mask, cancel, first result wins.
- [x] **M3.7 Duplicates, subscriptions, cleanup** (9.5, 9.6).
- [x] **M3.8 Mobility** (`mobility/`), section 10: VFC grid with lights and stops, speeds, hotspot, attach with hysteresis, GPS every 100 ms, coverage gaps, cut uploads (9.7), result follows the car (10.3).
- [x] **M3.9 Predictor** (10.2): exit time (ray–circle formula), next zone at intersections.
- [x] **M3.10 Logging** (16.3): Parquet files and InfluxDB points as in `docs/METRICS.md` (confirm or change that schema here); fix the Grafana dashboard queries in `docker/grafana/dashboards/` against real data.
- [x] **M3 acceptance tests:**
  - One scenario test per row of the failure table (9.1) and per case C1–C14 (7.4).
  - S1–S3 with Greedy, 3 seeds: ρ_fog and TI_WAN measured from the run match the config targets within 5 %; z1 load ≈ 1.6 × the mean; the system is overloaded: under Greedy the fog GPUs saturate (97–100 %); under cloud-only placement the WAN queue grows without limit (M2 test). If not, stop and debug the queueing before anything else.
  - Exit-time formula against hand calculations; infinity when stopped.

## M4 — Agent inputs and rewards (no learning yet)

- [x] **M4.1 The 48 inputs** (11.1), caps (times ≤ 2, energy ≤ ē), masked options still get numbers.
- [x] **M4.2 Rewards** (11.4) for every attempt outcome; C_drop = 5.
- [x] **M4.3 Transitions** (11.3): s, a, s', Δt at the same anchor, pending buffer keyed by (task_id, attempt), moved to replay at close + G, no transition for boulders and all-masked drops.
- [x] **M4 acceptance tests:**
  - Input vector length 48, every value in its range, order as in 11.1.
  - Property test: for random w in (0, 1] and random energies, every drop reward is lower than every delivery reward (N18).
  - Transition timing on a scripted scenario.

## M5 — NSGA-II expert and demonstrations

- [x] **M5.1 Windows** (13.2): T_win, T_warmup of greedy decisions, common random numbers, genome = action index per task, repair to "here", tasks after the window placed by greedy.
- [x] **M5.2 NSGA-II in pymoo** with the operators of 13.2 and section 18 numbers; parallel evaluation (spawn-safe).
- [x] **M5.3 Knee and w** (13.3): normalisation, knee = smallest f̃1 + f̃2, slope, w̃, w; median w*; w_p for every front point.
- [x] **M5.4 Demonstrations** (13.4): replay plans with the input builder, flag expert transitions, store to disk.
- **M5 acceptance tests:**
  - [x] Knee and w on synthetic fronts with known answers.
  - [x] Front check (16.5): S2 fronts hold at least 5 distinct points. Full budget (100 × 100, 12 workers, WSL), S2 window 0 (seed 1100): placeholder config 35 distinct of 38 (39 min); calibrated config (Q52) **28 distinct of 28** (30 min), knee (0.369, 0.999), greedy (0.398, 1.040) dominates no front point.
  - [x] Replaying a plan reproduces its objective values exactly (same seed).

## M6 — D3QN with DQfD, online learning

- [x] **M6.1 Network**: 48 → 256 → 256 → dueling heads, masked argmax (11.2, 11.3).
- [x] **M6.2 Losses** (13.5): 1-step double-DQN, n-step with the time-based discount, margin loss on expert transitions only, L2. The last transition of each anchor per episode has no s' (truncated, QUESTIONS Q36): leave it out of the bootstrapped targets (1-step and n-step).
- [x] **M6.3 Replay**: agent part (100 000) + expert part never overwritten; prioritized replay with ε_d, ε_a, α, β.
- [x] **M6.4 Pretraining loop** with held-out expert agreement and early stop.
- [x] **M6.5 Online learning** in every zone; neighbour order shuffled per episode in training (13.6).
- [x] **M6.6 Training curves** (TensorBoard or Grafana): loss, expert agreement, on-time rate and mean ê per episode.
- [x] **M6 acceptance tests:** hand-computed n-step target; margin loss is zero on non-expert transitions; masked actions never chosen; learning sanity on a two-action toy problem.

## M7 — Baselines, experiments, statistics

- [x] Baselines in the agent's seat with the same inputs, masks, reward and budget (16.1): D3QN from scratch, D3QN + greedy expert, VFC PPO, Tang & Wong, QECO, Greedy, Random, NSGA-II knee reference.
- [x] Scenario runner S1–S6, seeds from the evaluation range only, metrics and CIs (16.3), ablations and sweeps (16.4).

Open (not built in M7):

- [ ] Queue-order variants: FIFO, classes + FIFO, the full rule (8.7; 16.4).
- [ ] The fixed refusal penalty ablation (16.4; only if the refusal rate stays high).
- [ ] IQL instead of DQfD (16.4).
- [ ] 60 inputs: both slot types per option (16.4).
- [ ] Zone join, zone kill, broker restart (15.7; 16.4).
- [ ] Missing metrics (16.3): next-zone acc@1 and T_exit error, delivery success after a zone change, decisions until the on-time rate is stable, the hypervolume of the fronts in the report.

## M8 — Federated learning (phase 2)

- [x] FedAvg rounds and the three variants (13.7).

## M9 — Docker testbed

Skeleton already in `docker/` (Dockerfile, compose with every container and network of 15.4, `net-setup.sh` shaping, Grafana provisioning). Needs the home desktop (Docker Desktop, WSL2 backend).
- [~] **M9.1** Entry points `pcnme.testbed.{zone,car,cloud,runner}` on the same zone/car/cloud code as fast mode, with MQTT and HTTP adapters behind the bus and call interfaces (15.2). FedAvg + fine-tuning (Q51): on the testbed the K fine-tune steps take real time; the personal copy takes over only after they finish, and the old model decides meanwhile (13.7).
  Steps (branch `m9-1-testbed`, one commit each, tests first):
  1. [x] `testbed/clock.py` RealClock: EventClock's interface on real time; other threads hand work in with `call_soon`.
  2. [x] `testbed/codec.py` payloads ⇄ JSON (registered types only).
  3. [x] `testbed/mqtt.py` MqttBus/MqttClient behind the bus client interface (fake-paho tests).
  4. [x] `bus/calls.py` Calls / FastCalls: the zone's POST/GET/DELETE go through `zone.calls`; `testbed/http.py` HttpCalls + `make_app` (FastAPI). Fast mode unchanged: `tests/zone/test_calls_golden.py` pins an S2 run digest.
  5. [x] `testbed/radio.py`: the Shannon pipe runs in software at the zone's intake (15.3); the car's POST /tasks carries `Upload(task, b_car, rate)` and is answered at t_rx (409 if cut, 9.7); `software_radio()` leaves the base radio delays to netem; `RemoteCar` posts results to the car container and hands its Confirm to `Zone.car_confirmed`.
  6. [x] Entry points `pcnme.testbed.{zone,car,cloud}` (+ `serve.py`), Q53 built (no modelled delays, C7 from the written body, own-backhaul bookkeeping kept); zone endpoints POST /cars, POST /gps, GET /closed, GET /status; `@pytest.mark.testbed` compose test (broker, wan-gw, cloud, 4 zones, car-1 with 20 cars): tasks close on time end to end.
  7. [~] FedAvg + fine-tuning on the testbed (not yet), and the learning methods in the containers (done 5 Nov): **Q54** (proposal: Greedy for M9.2–M9.4 first; then CPU torch in the zone image).
- [x] **M9.2** Verify the network (15.4 "to verify"): routing through wan-gw, netem jitter distribution and reordering, `tcp_slow_start_after_idle=0`, one shaped crossing per hop (`tests/testbed/test_network.py`, `pytest -m testbed`). Found and fixed: jitter reordering distorted WAN transfers (netem now keeps packets in order, **Q56**); a warm 1 MB upload fits 4.2's S/B + d. MinIO cannot be pinned: its public images are gone (**Q55**, proposal: bind-mounted results).
- [~] **M9.3** Runner: per-scenario shaping, zone join / kill / broker restart (15.7), results to the bind-mounted data/ (Q55).
  Built (branch `m9-3-runner`): `python -m pcnme.testbed.runner --scenario S2 --seed 3000 --method greedy --seconds 60
  [--containers N]` sets every netem value from the config, starts the stack, applies `testbed.events` (empty:
  **Q57**), waits for the zones' rows, saves `containers.log`, and writes the run's Parquet tables (zones write JSON
  rows, `record_rows` shared with fast mode). Results go to a bind mount (Q55: no MinIO).
  Found and fixed on the full fleet: per-car GPS requests (now one GpsBatch per zone and period); ~250 HTTP threads
  per zone starving its loop (now one asyncio network thread, `testbed/netloop.py`); zone POSTs carried no input
  bytes, so the WAN saw no bulk and ŵ_WAN went negative (greedy sent most tasks to the cloud; now 5.3's input
  rides with the request).
  **5 Nov:** zone CPU fixes (HTTP client); Greedy at 40 cars holds S1 96.9 % and S2 95.4 % on time (fast mode
  99.8-99.9 %); at 200 cars the busiest zone is GIL-bound (48.9 %). Learning methods run in the zones, but
  PCNME loses the car confirms (open bug, RUN_LOG 5 Nov). Details: docs/RUN_LOG.md.
  **Left:**
  0. Fix the PCNME confirm bug (drop_no_ack 88 %), then the M9.4 gap for PCNME at 40 cars.

  1. [x] (5 Nov: 40 cars hold; 200 cars are GIL-bound in the busiest zone) Reduced scale the containers can hold: 40–60 cars (`python -m pcnme.testbed.runner --containers 2` or 3), the
     same per-zone load (lambda_car scales with the fleet, 18). Check on-time against fast mode at that fleet.
  2. [x] (5 Nov: profiled; two HTTP-client fixes, commit 33f3240) If it still lags, one timeboxed session: move the zone onto the asyncio loop of its network I/O (one thread),
     profile with `PCNME_PROFILE=1`. If it still does not hold, stop and report (the paper states the scale limit).
  3. Topology runs (Q57, low priority, last): three separate 120 s runs, event at 40 s: `start zone-5`;
     `kill zone-3` + `kill zone-4`; `restart broker`.
  Decided: Q54 Greedy first, then the agent with CPU-only torch in the zone image; Q55 bind mount, MinIO removed;
  Q56 the in-order jitter bias (+8–24 %) is reported in the gap.
- [~] **M9.4** (Greedy done 5 Nov: S1 96.9 % vs 99.9 %, S2 96.6 % vs 99.8 % on time, data/vfc/testbed/gap-*.md;
  the agent waits on the confirm bug) Same scenario and seed in fast mode and on the testbed, at M9.3's reduced scale, Greedy first (no trained model needed), then the agent: report the gap (16.3). Tool ready: `tools/testbed_gap.py` (fast-mode run with the testbed's fleet and seed, one Markdown table of the 16.3 metrics). Waits on M9.3's open item; the in-order jitter bias (Q56) is part of the gap.

## T — Training and results (the runs behind the paper; the user starts the full runs)

- [x] **T1** `train.py --maps <map>` (src/pcnme/pipeline/train.py): traces (trace maps), NSGA-II windows, knees,
  w* per map, NSGA-II and greedy demonstrations, DQfD pretraining + online training of PCNME, the learning
  baselines on the same budget and w* (16.1); live screen, per-episode CSVs, checkpoints, resume, manifests.
  `evaluate.py --maps <map>`: every scenario x method x seed, summary.csv, tables, figures. `run_testbed.py`:
  stops until M9.3/M9.4. Smoke-tested on vfc and lust (`--smoke`). Commands: RUN_TRAINING.md.
- [x] **T2** VFC: `train.py --maps vfc` then `evaluate.py --maps vfc` (done 5 Nov: 36 h training, 960 evaluation
  runs; docs/RESULTS_VFC.md, docs/RUN_LOG.md).
- [ ] **T3** LuST: `train.py --maps lust` (generates the traces first, about 30-60 min) then evaluate.
- [-] **T3b** Dropped (Imed, 6 Nov): no transfer test of the VFC models on LuST. LuST is trained and
  evaluated on its own (T3).
- [x] **T3c** (done 5 Nov: metrics/compare.py, evaluate.py --models-from; smoke-tested) evaluate.py additions (after the VFC run; nothing in src/ changes while it trains). Important, before
  Friday (Imed, 4 Nov):
  1. cross-scenario figures: one per key metric (on time, p99 latency, mean ê), all methods over S1-S6;
  2. PCNME vs each baseline: a paired test over the same seeds (is the difference real or noise?);
  3. (dropped with T3b)
  Not needed: a results table printed in the terminal (report.md and summary.csv are enough).
  Later (after Friday; fast mode, not the testbed): the 16.3 metrics still missing (next-zone acc@1, T_exit error,
  delivery success after a zone change, decisions until the on-time rate is stable, front hypervolume); they need
  small additions to the fast-mode logs, then a re-evaluation. The knee reference (16.1, Q45) is optional.
- [x] **T5** (done 7 Nov: trained 12.2 h, evaluated; RESULTS_VFC, RUN_LOG 7 Nov. 6 Nov: Q60 applied: config and final-design 16.2/18; S6-only revision built and smoke-tested,
  pipeline/revision.py; the S6 training and evaluation are the user's to run, RUN_TRAINING.md) S6 (QUESTIONS Q60): if agreed, S6 with loose deadlines: rerun S6's NSGA-II windows, check w*, retrain S6 only, evaluate S6 only. Results summary for the supervisor: docs/RESULTS_VFC.md.
  The pipeline cannot do this safely yet (checked 6 Nov), so build it first:
  1. `train.py --maps vfc --scenarios S6` refuses: the config hash and the plan changed (manifest identity), and
     `--fresh` would archive ALL of data/vfc (every trained model and result). Never use --fresh for this.
  2. stage_nsga computes w* from the windows of the scenarios it is given: with S6 only, w_star.json would be
     overwritten with the median of S6's 6 knees instead of all 36.
  3. evaluate.py's summary.csv refuses rows of another commit or config hash, so the new S6 rows cannot join it.
  Decided (Imed, 6 Nov): no variant folder; w* fixed at 0.414; S6 rerun in place as a revision of the finished
  run (all three points solved: revision record in the manifest, w* kept, per-scenario row checks and archive).
- [x] **D1** (done 6 Nov; deletions listed in RUN_LOG 6 Nov) Docs cleanup (after the evaluations, before the write-up): update every .md file to match the code
  and the results (README, READ, RUN_TRAINING, CLAUDE, data/DATA, docs/PLAN, METRICS, RUN_LOG, QUESTIONS, CHANGES,
  data/calibration/cpu-power, visual/city/README, .claude/skills/*); delete the ones nothing needs any more and list
  each deletion with its reason; check every path and command in them still works. final-design.md: content only,
  no notation renames (Imed does those later).
- [ ] **T6** S6 analysis (7 Nov finding: in S6 with loose deadlines Greedy is best, 96.9 % on time, against PCNME
  92.1 %, level with both D3QN variants). Read-only first: compare PCNME's decisions with Greedy's per anchor zone
  (F-GPU vs F-CPU), task type and option (decisions.parquet, attempts.parquet); check the training curves for
  convergence. Then, if Imed agrees: S6 with a larger budget, and the 16.4 ablation "60 inputs: both slot types per
  option" (it changes the agent's inputs: Imed decides).
- [ ] **T4** Results write-up from data/<map>/results/summary.csv: the 16.3 table per map, training curves,
  the number of seeds, what is missing (knee reference, ablations 16.4, sweeps).

## R — Roadmap (Imed, 6 Nov; the order of the next weeks)

1. **VFC S6** (T5): train and evaluate S6 with loose deadlines (tonight). Then back up (B1) and write up S6.
2. **VFC testbed** (M9), to the end:
   a. the PCNME confirm bug (drop_no_ack 88 %), then M9.4 for every method at 40 cars (fast mode vs testbed);
   b. live metrics: zones and the runner push the 16.3 points to InfluxDB while a run goes (today they are only
      written to points.lp after it); Grafana dashboards per run, zone and method (docker/grafana/);
   c. V3 live mode: the viewer's `live-testbed` source (MQTT -> server-sent events), the same CityView;
   d. the topology runs (Q57) and the scale question (a zone split over processes for 200 cars, or report 40).
3. **LuST v1** (the 1.2 km crop, 4 zones, final-design 16.2): L5 traces, T3 training and evaluation, V2 the road map
   in the viewer, then its testbed (the car containers replay traces instead of walking the grid: new work).
4. **One platform for both maps.** Keep one viewer, one Grafana and one testbed, with the map as a parameter
   everywhere (sources per map in the viewer, a `map` tag on every Influx point, `--maps` in the scripts),
   not separate copies.
5. **LuST v2 (idea, to discuss; not decided):** the whole LuST city (about 155 km^2) with about 20 fog zones,
   and a viewer where one picks a crop anywhere. Notes (6 Nov):
   - The agent need not change: it is per zone, sees itself, at most 4 neighbours and the cloud (48 inputs,
     6 actions), so 20 zones are 20 agents with the same network (13.1 already trains one network on all zones;
     FedAvg, M8, is where many zones pay off). Needed: a zone layout for the city (positions, radii), the rule
     for choosing 4 neighbours when more are within d_nb (a design question), a LuST-wide trace (24 h, many
     thousand vehicles: fast mode's speed and NSGA-II's window size grow with the fleet), and new training.
   - "Crop anywhere" in the viewer means running something on demand: SUMO traces for the crop (minutes), then a
     fast-mode run with a trained model (minutes), then a replay. Doable as a viewer feature after LuST v1.
   - The testbed would not hold 20 zones on one desktop (the 200-car limit already binds at 4 zones).
   - Suggested scope: a thesis extension or future work, after LuST v1 is complete.

## Q — Quality gate (Imed, 6 Nov: "the design is mine, the code is yours: make every result, the pipeline and the
testbed correct"). Claude owns these; an item is done only when its check passes and is recorded in RUN_LOG.

- [~] **Q1 Testbed = the design.** (7 Nov: two bugs found by counters and fixed, PCNME 11 % -> 89.8 % on time at
  S2, 40 cars, fast mode 99.8 %; the rest is lateness, a real gap; RUN_LOG 7 Nov. Left: every method at 40 cars,
  Q61 to confirm) Fix the PCNME confirm bug (a code bug: the agent's path loses the car's confirm,
  drop_no_ack 88 %, Greedy fine on the same build). Then every method (Greedy, Random and the 6 learners) runs on
  the testbed at 40 cars with no drop_no_ack beyond fast mode's, and its gap to fast mode on the same seed is
  explained metric by metric (network jitter Q56, real run times), not by a bug. A test pins the fix.
- [ ] **Q2 Fast mode = the design.** Re-read every final-design section the code cites against the code (a
  section-by-section review, findings to QUESTIONS/CHANGES, never silent fixes); the acceptance tests of M1-M8 and
  the determinism test (same config and seed, bit-identical logs) pass on the final config.
- [ ] **Q3 Results are reproducible.** From a clean checkout: `train.py --smoke`, `evaluate.py --smoke` and one
  real evaluation seed per scenario reproduce summary.csv's rows for that seed exactly (same commit, same config);
  every number in RESULTS_VFC.md is generated from summary.csv (no hand-typed numbers left).
- [ ] **Q4 Pipeline robustness.** Interrupting train.py and evaluate.py at any stage and rerunning resumes with
  bit-identical results (test_resume extended to NSGA-II, demos and the revision path); mismatched configs and
  commits are refused with a clear message; nothing is ever deleted (only archived).
- [ ] **Q5 Historical viewer = the logs.** For one run per scenario, the viewer's statistics strip at the end of
  the run equals summary.csv (on time, dropped, mean latency, J/task); the export opens on a clean machine.
- [ ] **Q6 Live viewer and Grafana = the logs.** A live testbed run (V3, R2) shows the same final numbers in the
  viewer, in Grafana and in the run's Parquet files.
- [ ] **Q7 LuST passes Q1-Q6 too** before its results are reported.

## B — Backups (outside the repository: F:\projects\pcnme-results-backup\<map>-<date>\)

- [x] **B0** vfc-2026-11-05: data/vfc after the full training and evaluation (trained models, Parquet runs,
  testbed runs). It holds the tight-deadline S6 too (models, NSGA-II windows, runs).
- [x] **B1** (done 7 Nov, 7.9 GB: the new S6, the archived tight S6, a README) After tonight's S6 run: a new dated copy, vfc-2026-11-07 (the folder pattern stays <map>-<date>),
  with the new S6 and the archived tight S6 (data/vfc/_archive/S6-tight-<time>/,
  results/archive/S6-tight/). Inside the copy the archived folders keep their names (S6-tight...).

## L — LuST map (Codeca et al. 2015; final-design 16.2, 18)

- [x] **L1** SUMO (eclipse-sumo, the `maps` extra) and the LuST scenario in data/maps/lust/ (MIT licence).
- [x] **L2** 1.2 x 1.2 km central crop (config maps.lust: centre 6.1319 E, 49.6116 N; the layout's area and
  zone layout unchanged), one SUMO run per time of day (off-peak S1, peak S3/S5, mid S2/S4/S6: Q58), FCD at
  100 ms on the crop's edges only (src/pcnme/mobility/lust.py).
- [x] **L3** Trace replayer with the grid's interface (src/pcnme/mobility/traces.py): fixed fleet, slots of
  recorded vehicles, seed-chosen slices; City uses it on a trace map; coverage measured on the traces
  (experiments/calibrate.measure_trace) scales lambda_car per map (data/lust/calibrated.yaml).
- [x] **L4** Tests: tests/mobility/test_traces.py (slots, interpolation, gaps, seeds, a City episode), a slow
  real-SUMO test tests/mobility/test_lust.py; `train.py --maps lust --smoke` end to end.
- [x] **L5** (done 6 Nov, 36 min; RUN_LOG 6 Nov) Full traces (train.py makes them on first use) and a look at the
  vehicles per crop and coverage. The viewer shows them as LuST's traffic preview until LuST is trained.

## C — Calibration (any time after M0, on the home desktop)

- [x] Measure YOLO run times and energy per run on the GTX 1070 (slot fixed to x16 first), JPEG sizes on BDD100K frames; write the time table; rerun `tools/calc_defaults.py`. Cloud rows from Ultralytics' A100 numbers plus measured pre- and post-processing.
  Done 3 Nov at PCIe x4 (`tools/calibrate_yolo.py`, `data/calibration/`), applied per Q52. Mechanism tests run on the frozen placeholder config `tests/data/mechanism.yaml`.

## V — Viewer (not research: it never changes a result)

- [x] **V1 Replay viewer** (done 5 Nov: `uv run python view.py`, src/pcnme/viewer/README.md; every scenario, method and seed, fast mode and testbed, two runs side by side) (after M3): reads one run's Parquet files (the Q28 schema: positions, events, attempts, zones) and draws the VFC grid, the zone circles, the cars and their tasks; play, pause and speed; click a car or a task to see it. The data source sits behind one interface, so V3 only adds another source.
- [ ] **V2 LuST map** (when LuST is added): the road map drawn from the SUMO net file.
- [ ] **V3 Live mode** (M9): the same viewer fed live from the MQTT broker over WebSocket.
- [ ] **V4 Live training**: a `live-training` source and view (episodes, loss, on-time, decisions changing); extension points in src/pcnme/viewer/README.md.
- [ ] **V5 Responsive, polished UI** (Imed, 6 Nov: not responsive enough yet): one layout system for every width
  from a laptop to a projector: the top bar collapses into a menu below about 1100 px (no horizontal scroll),
  the statistics strip wraps cleanly, the side panel becomes an overlay sheet on narrow screens, the map keeps
  readable zone cards at every size (no compact fallback needed above 900 px), touch targets of at least 32 px,
  compare mode stacks the two maps vertically when the window is taller than wide. A light/dark switch in the bar
  (today the page follows the OS theme only). Check at 1366x768, 1920x1080, 2560x1440 and a phone width; then
  re-export pcnme-history-viewer.
