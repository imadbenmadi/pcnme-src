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
  - [ ] Front check (16.5): S2 fronts hold at least 5 distinct points. If not, stop: the energy model needs revisiting. Open until a full-budget run on the desktop (tools/run_nsga.md); the small runs with C9 gave 3-6 distinct points.
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

- [ ] Baselines in the agent's seat with the same inputs, masks, reward and budget (16.1): D3QN from scratch, D3QN + greedy expert, VFC PPO, Tang & Wong, QECO, Greedy, Random, NSGA-II knee reference.
- [ ] Scenario runner S1–S6, seeds from the evaluation range only, metrics and CIs (16.3), ablations and sweeps (16.4).

## M8 — Federated learning (phase 2)

- [ ] FedAvg rounds and the three variants (13.7).

## M9 — Docker testbed

Skeleton already in `docker/` (Dockerfile, compose with every container and network of 15.4, `net-setup.sh` shaping, Grafana provisioning). Needs the home desktop (Docker Desktop, WSL2 backend).
- [ ] **M9.1** Entry points `pcnme.testbed.{zone,car,cloud,runner}` on the same zone/car/cloud code as fast mode, with MQTT and HTTP adapters behind the bus and call interfaces (15.2).
- [ ] **M9.2** Verify the network (15.4 "to verify"): routing through wan-gw, netem jitter distribution and reordering, `tcp_slow_start_after_idle=0`, one shaped crossing per hop. Pin the MinIO image.
- [ ] **M9.3** Runner: per-scenario shaping, zone join / kill / broker restart (15.7), result upload to MinIO.
- [ ] **M9.4** Same scenario and seed in fast mode and on the testbed: report the gap (16.3).

## C — Calibration (any time after M0, on the home desktop)

- [ ] Measure YOLO run times and energy per run on the GTX 1070 (slot fixed to x16 first), JPEG sizes on BDD100K frames; write the time table; rerun `tools/calc_defaults.py`. Cloud rows from Ultralytics' A100 numbers plus measured pre- and post-processing.

## V — Viewer (not research: it never changes a result)

- [ ] **V1 Replay viewer** (after M3): reads one run's Parquet files (the Q28 schema: positions, events, attempts, zones) and draws the VFC grid, the zone circles, the cars and their tasks; play, pause and speed; click a car or a task to see it. The data source sits behind one interface, so V3 only adds another source.
- [ ] **V2 LuST map** (when LuST is added): the road map drawn from the SUMO net file.
- [ ] **V3 Live mode** (M9): the same viewer fed live from the MQTT broker over WebSocket.
