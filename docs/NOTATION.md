# PCNME notation

Approved by Imed on 15 Nov 2026. This table decides every symbol in `docs/final-design.md`, in code comments and
in docstrings. It replaces the symbols used up to revision 128 of the design.

It changes names only. No rule, formula, number or behaviour changes.

## 1. Rules

One base letter is one kind of quantity with one unit. The subscript says which instance.

| Letter | Always means | Unit |
| --- | --- | --- |
| T | A duration that is computed, estimated or configured | s |
| t | An instant on one machine's clock | s |
| D, d | Deadline: D = its length, d = its instant on one clock | s |
| W | A waiting time (in a pipe or in a queue) | s |
| δ | A link delay | s |
| L | Latency measured by the car | s |
| S | A size | bit |
| B | A bandwidth or bit rate of a link | bit/s |
| E, e | Energy: E = of an attempt or task, e = per bit | J, J/bit |
| P | Power | W |
| C | A cost inside the objective and the reward | none |
| n | A count (tasks, slots, places) | none |
| w | The latency weight of the reward | none |
| r, Q, s, a, γ | Reward, Q-value, state, action, discount | none |
| ℓ | A length on the map | m |

| Mark | Always means |
| --- | --- |
| x̂ (hat) | An estimate made before the fact |
| x̄ (bar) | A mean from the configuration or the time table |
| x^lo, x^hi | The 5 % and 95 % quantiles from the time table |
| x̃ (tilde) | Normalised to the range of the Pareto front |
| x* | The chosen one (q* the assumed slot type, w* the deployed weight) |

| Index | Runs over |
| --- | --- |
| k | The 6 placement options: here, neighbour 1–4, cloud |
| q | Slot types: GPU, CPU |
| i | Tasks |
| j | Slots of one type on a node; steps of the n-step return |
| z | Zones |
| g | FedAvg rounds |
| p | Points of a Pareto front |

Names are not symbols: S1–S6 (scenarios), T1–T6 (task types), C1–C14 (cases), z1–z5 (zones), R1–R3 (herding
rules), F-GPU and F-CPU (hardware classes) never appear inside an equation and keep their names.

## 2. Old → new

"Sections" are the sections of `docs/final-design.md` where the symbol appears.

### 2.1 Run times, budgets and timers

| Old | New | Meaning | Unit | Sections |
| --- | --- | --- | --- | --- |
| Ê_k,q | T̄^run_k,q | Mean run time of the task's type on option k, slot type q (time table) | s | 7.2, 8.3, 8.4, 11.1, 12.4 |
| Ê_strong, Ê_cloud | T̄^run_strong, T̄^run_cloud | The same on the strongest fog class / on the cloud, fastest slot type | s | 7.1 |
| Ê^low, Ê^high | T^run,lo, T^run,hi | 5 % and 95 % quantile of the run time | s | 8.4, 8.6, 9.3 |
| T_run | T^run | The real busy time of the slot for one attempt | s | 12.2 |
| B_car | T^bud_car | Time budget the car sends: D − time since creation | s | 5.3, 5.4, 6A, 9.7 |
| B_k (budget) | T^bud_k | Time budget the anchor sends to executor k | s | 5.3, 6C |
| Û_up | T̂_radio | Time the upload took on the radio, taken off the budget | s | 6A |
| T_up,k | T^to_k | Upload timeout for a send to option k | s | 9.3 |
| d_abort | T_abort | Longest step of the task on that slot type | s | 8.6, 12.2, 12.5, 14.2 |
| A_max | T_age | Wait after which a soft task may be promoted | s | 8.4, 18 |
| ℓ (in 9.3) | L_late | Lateness a class may have: 0 for hard, L_max for soft | s | 9.3 |
| t_s | t_send | Instant the anchor starts a send | s | 6C, 9.3 |

### 2.2 Links, waits and delays

| Old | New | Meaning | Unit | Sections |
| --- | --- | --- | --- | --- |
| d, d_min, d_max | δ, δ_min, δ_max | Random delay drawn per transfer and its bounds | s | 4.1 |
| d_bh, d_WAN | δ_bh, δ_WAN | The drawn delay on the backhaul / the WAN | s | 4.1 |
| δ̂_radio, δ̂_reply, δ̂_pub | δ̄_radio, δ̄_reply, δ̄_pub | Configured mean delay of a link or path | s | 5.4, 6A, 9.3 |
| ŵ_WAN | Ŵ_WAN | Moving-average estimate of the wait in the shared WAN pipe | s | 4.3, 5.2 |
| ŵ_k | Ŵ^pipe_k | Estimated pipe wait on the way to option k | s | 4.3 |
| Ŵ_k,q^κ, Ŵ_q | Ŵ^queue_k,q, Ŵ^queue_q | Estimated queue wait for the task's class; the class is written as a word (hard, soft) | s | 7.2, 8.3, 8.5, 9.3, 11.1 |
| λ (WAN estimate) | η | Smoothing factor of the moving average | none | 4.3, 18 |
| S_r | S_res | Size of a result (S stays the input size) | bit | 12.2 |
| R_z | ℓ_z | Radius of a zone's coverage circle | m | 10.2, 18 |
| l_nb | ℓ_nb | Largest centre distance that makes two zones neighbours | m | 7.5, 18 |
| p, c (in 10.2) | x_car, x_z | Position of the car / centre of the zone | m | 10.2 |
| U_up | T_radio | The real upload time on the radio (the estimate is T̂_radio) | s | 4.1, 6A |

### 2.3 Slots and queues

| Old | New | Meaning | Unit | Sections |
| --- | --- | --- | --- | --- |
| U_k | 𝒬_k | Set of slot types usable for this task on option k | none | 7.2 |
| u_q | n^busy_q | Busy slots of type q | none | 5.2, 8.5 |
| Q_q | n^max_q | Capacity of the queue of slot type q | none | 7.2, 8.2, 18 |
| r (in 8.2) | n_res | Queue places kept for hard tasks | none | 7.2, 8.2, 18 |
| h (in 7.2) | n_head | Headroom: places the mask keeps free at a neighbour | none | 7.2, 9.2, 18 |
| B_q^hard, B_q^soft | Ω_q^hard, Ω_q^soft | Backlog: sum of the mean run times of the queued tasks of that class | s | 5.2, 8.5 |
| r_q,i | T^rem_q,j | Remaining mean time of the task on slot j (0 if free) | s | 5.2, 8.5 |
| n (in 4.2) | n_up | Uploads in progress on a zone's radio | none | 4.1, 4.2 |
| "the soft task S" | the soft task i | A task is always i | none | 8.4 |
| R_q | (in words) | Sum of the remaining slot times; the old pooled estimate, no longer used | s | 8.5 |

### 2.4 Energy, cost and reward

| Old | New | Meaning | Unit | Sections |
| --- | --- | --- | --- | --- |
| c^time | C^time | Time cost of a task: L/D plus the lateness term, or C_drop | none | 11.4, 13.2 |
| ê | C^en | Energy cost of a task: min(E / E_ref, C^en_max) | none | 11.1, 11.4, 13.2 |
| ē | C^en_max | Cap of the energy cost | none | 11.1, 11.4, 12.3, 13.2, 18 |
| B (= 1) | r_ok | Reward bonus of a delivered task | none | 11.4, 18 |
| P (= 2) | C_pen | Extra cost of a drop inside C_drop | none | 11.4, 13.2, 18 |
| p̄_k,q | P̄_k,q | Mean dynamic power of a time-table row | W | 12.2, 12.3, 12.4, 18 |
| w1, w2 | w, 1 − w | They were other names for the same number | none | 11.4, 13.3 |

### 2.5 Learning

| Old | New | Meaning | Unit | Sections |
| --- | --- | --- | --- | --- |
| s (in 13.3) | σ | Slope of the front at the knee (s stays the state) | none | 13.3 |
| R_i | Δf_i | Range of objective i on the front: f_i^max − f_i^min | none | 13.3 |
| N (in 13.2) | N_task | Tasks in one NSGA-II window | none | 13.2 |
| N1–N4 (genes) | Nb1–Nb4 | Neighbour positions as gene values | none | 13.2 |
| n (n-step) | n_step | Length of the n-step return | none | 13.5, 18 |
| k (n-step sum) | j | Step inside the n-step return | none | 13.5 |
| k (FedAvg) | g | FedAvg round | none | 5.2, 5.3, 13.7 |
| l(a_E, a) = 0.8 | m_E | Margin of the expert loss | none | 13.5, 18 |
| α (TOF-Broker) | α_TOF | Boulder threshold | none | 7.1, 16.4, 18 |
| α, β0 (replay) | α_PER, β_PER | Prioritized-replay exponents | none | 13.5, 18 |
| h (in 16.2) | h_hot | Share of cars kept on z1's roads | none | 16.2, 18 |
| S1–S4 (DAG steps) | step 1–step 4 | Internal steps of a task | none | 14.2 |
| λ_E | λ2 | Weight of the expert margin loss | none | 13.5 |
| N (early stage) | N_early | Online decisions counted for early-stage performance | none | 16.3, 18 |

## 3. Renames that depend on the context

A blind search and replace is wrong for these. Read the sentence first.

| Old spelling | It becomes | When |
| --- | --- | --- |
| Ê | T̄^run (or T^run,lo / T^run,hi) | It is a run time: Ê_k,q, Ê_q, Ê_strong, Ê_cloud, Ê^low, Ê^high, and "Ê" in 7.2, 8.3, 8.4, 8.6, 9.3 |
| Ê | Ê (unchanged) | It is the energy estimate of option k: Ê_k in 11.1, 12.3, 12.4, 13.2 |
| B_k | T^bud_k | It is the budget sent to an executor (5.3, 6C) |
| B_k | B_k (unchanged) | It is the bandwidth on the way to option k (4.3) |
| B | Ω | It is the backlog in the water-filling formula of 8.5 ("= B", "r + B") |
| B | r_ok | It is the reward bonus (11.4, 18) |
| B | B (unchanged) | It is a bandwidth |
| r | n_res | Reserved queue places (7.2 "Q_q − r − h", 8.2, 18) |
| r, r_i, r_q,i | T^rem | Remaining slot time (5.2, 8.5) |
| r, r_k | r, r_j | The reward (11.3, 11.4, 13.5) |
| h | n_head or h_hot | Mask headroom (7.2, 9.2, 18 Nodes) or hotspot share (16.2, 18 Cars) |
| d | δ | A link delay; d_A and d_k stay deadlines |
| k | g or j | FedAvg round (13.7, fl/ topics, /fl/rounds/k) or n-step index (13.5); the option index k stays |
| λ | η | Only the smoothing factor in 4.3; λ_car and λ1–λ3 stay |
| α | α_TOF or α_PER | By section: 7.1 and 16.4, or 13.5 |
| W (water level, 8.5) | Ŵ^queue_q | It is the estimate itself |

## 4. Symbols that stay

D, L, L_max, slack, d_A, d_k · t_now, t_free, t_rx, t_acc, t_wd · T_transfer, T̂_in,k, T̂_ret,k, T_exit, T_waste,
T_win, T_warmup, T_fl, T_ack, T_probe, T_keep, T_leave, G, m_wd, τ, Δt · S, B, B_k (bandwidth), B_WAN, R_car,
W_pipe, W_bh, W_WAN, W_radio, δ̄_k, TI, κ_up · ρ_fog, ρ_node, λ_car · c_q, n_q^hard, n_q^soft, q, q*, F_q ·
E_k, E^fwd, E^exec, E^ret, Ê_k (energy), E_ref, E_waste, e_bh, e_wan, e_down, PUE, P_tx · w, w3, w̃, w_p, w*,
C_drop, r, f₁, f₂, f̃ · s, a, s′, γ, Q(s, a), 𝒜(s), y, ε, K, H · J_DQ, J_n, J_E, a_E, λ1–λ3, ε_d, ε_a ·
v, 𝒰(a, b), u.

## 5. Names in the code

Since 16 Nov 2026 the code follows this notation too: Python names, config keys, message fields and CSV
columns were renamed as in section 6, and the existing result files under data/ were rewritten the same way
(for example `mean_e_hat` → `mean_c_en`). The config hash (`src/pcnme/maps.py`) covers every config key, so
it changed with the renamed keys; runs made before 16 Nov carry the old hash.

## 6. Symbol → code

(Filled by the notation sweep: for every symbol of section 2 that has one, its Python name, its config key and
its message field.)

| Symbol | Python name | Config key | Message field |
| --- | --- | --- | --- |
| T̄^run_k,q | `RunRow.mean` (node/timetable.py); `OptionView.run` (zone/decision.py) | `time_table_placeholder.<type>.<class>.<slot>` (`mean`, or the number alone) | — |
| T̄^run_strong | `Derived.t_run_strong` (config.py) | computed from the time table and `tof.strongest_class` | — |
| T̄^run_cloud | `TimeTable.fastest(task, "cloud")`, its row's `mean` (config.py) | `time_table_placeholder.<type>.cloud` | — |
| T^run,lo, T^run,hi | `RunRow.low`, `RunRow.high`; `TimeRow.p5`, `TimeRow.p95` | `p5`, `p95` of a row; `time_table_placeholder.factors.p5` / `.p95` | — |
| T^run | `Report.busy` (node/executor.py) | — | — |
| T^bud_car | `Car.budget()` (car/car.py); `Upload.t_bud_car` (testbed/messages.py) | — | `t_bud_car` (testbed upload); `T_bud_car_ms` in 5.3 |
| T^bud_k | `TaskRequest.budget` (node/executor.py); `budget` in zone/zone.py | — | `budget` (TaskRequest); `budget_ms` in 5.3 |
| T̂_radio | `upload` in `anchor_deadline()` (zone/intake.py) | — | — |
| T^to_k | `t_to` of the attempt (zone/zone.py) | `timing.kappa_up` (κ_up) | — |
| T_abort | `RunRow.t_abort` | `t_abort` of a row; `time_table_placeholder.factors.t_abort` | — |
| T_age | `Scheduler.t_age` (config.py) | `scheduler.T_age` | — |
| L_late | `_cap()` (zone/zone.py): `l_max` for soft, 0 for hard | — | — |
| t_send | `t_send` (agent/reward.py, experiments/city.py) | — | — |
| δ, δ_min, δ_max | `hops.delay()` (net/hops.py) | `radio.delay`, `radio.downlink_delay`, `links.backhaul_delay`, `links.wan_delay` (each `[min, max]`) | — |
| δ_bh, δ_WAN | `d_bh`, `d_wan` (net/series.py, zone/zone.py) | `links.backhaul_delay`, `links.wan_delay` | — |
| δ̄_radio, δ̄_reply, δ̄_pub | `mean_radio_delay` (zone/intake.py); `_reply_delay()` (zone/zone.py); `StateStore.pub_delay` | the means of the delay intervals above; `links.broker_extra_delay` | — |
| Ŵ_WAN | `StateStore.wan.estimate`, `Snapshot.wan_wait` (zone/statestore.py) | — | `wan_wait` (Snapshot); `wan_wait_ms` in 5.2 |
| Ŵ^pipe_k | `wait` in `PathEstimates.t_in_cloud()` / the `backhaul_wait` argument (zone/estimates.py) | — | — |
| Ŵ^queue_k,q, Ŵ^queue_q | `wait_estimate()` (node/state.py), `Executor.wait_estimate()`, `Accepted.wait`, `OptionView.wait` | — | `wait` (Accepted); `wait_ms` in 5.3 |
| η | `StateStore.wan_eta`; `StatePlane.wan_ewma_eta` (config.py) | `stateplane.wan_ewma_eta` | — |
| S_res | `TaskType.result_bits` (config.py) | `task_types.<type>.result_kb` | — |
| ℓ_z | `Layout.zone_radius_m`; `Site.radius`, `ZoneStatus.radius` | `layout.zone_radius_m` | `radius` (zone status) |
| ℓ_nb | `Layout.l_nb_m`; `NeighbourTable.l_nb` (zone/layout.py) | `layout.l_nb_m` | — |
| x_car, x_z | `Gps.x`, `Gps.y`; `Site.x`, `Site.y` | `layout.zone_positions_m` (x_z) | `x`, `y` (GPS, zone status) |
| 𝒬_k | `usable` (zone/masks.py) | — | — |
| n^busy_q | `SlotState.n_busy` (node/state.py) | — | `n_busy` (SlotState and 5.2) |
| n^max_q | `SlotState.n_max` | `hardware.<class>.queue.<slot>` | `n_max` (SlotState and 5.2) |
| n_res | `Scheduler.n_res` | `scheduler.n_res` | — |
| n_head | `Scheduler.n_head`; `headroom` (zone/masks.py) | `scheduler.n_head` | — |
| Ω_q^hard, Ω_q^soft | `SlotState.omega_hard`, `SlotState.omega_soft` | — | `omega_hard`, `omega_soft` (SlotState); `omega_hard_ms`, `omega_soft_ms` in 5.2 |
| T^rem_q,j | `SlotState.t_rem` | — | `t_rem` (SlotState); `t_rem_ms` in 5.2 |
| n_up | `len(RadioUplink._active)` (net/radio.py) | — | — |
| C^time | `c_time()` (agent/reward.py) | — | — |
| C^en | `c_en()` (agent/reward.py, energy.py); column `mean_c_en` | — | — |
| C^en_max | `Derived.c_en_max` (config.py) | `energy.c_en_max` (`computed`) | — |
| r_ok | `Reward.r_ok`, `RewardParams.r_ok` (config.py, agent/reward.py) | `reward.r_ok` | — |
| C_pen | `Reward.c_pen` | `reward.C_pen` | — |
| P̄_k,q | `RunRow.power` (node/timetable.py); `TimeRow.p_dyn_w` (config.py) | `energy_j` of a measured row ÷ its mean; `hardware.<class>.p_dyn_w.<slot>` otherwise | — |
| w, 1 − w | `Reward.w` | `reward.w` | — |
| σ | `Knee.slope` (expert/knee.py) | — | — |
| Δf_i | `Knee.ranges` | — | — |
| N_task | the window's task count (expert/window.py) | — | — |
| Nb1–Nb4 | action indices 1–4 (agent/inputs.py, expert/nsga.py) | — | — |
| n_step | `Dqfd.n_step` (config.py) | `dqfd.n_step` | — |
| g | `RoundMsg.k`, `FedState.k` (fl/fedavg.py) | — | `k` (fl/round) |
| m_E | `Dqfd.margin` | `dqfd.margin` | — |
| α_TOF | `Tof.alpha` (config.py) | `tof.alpha`, `tof.alpha_sweep` | — |
| α_PER, β_PER | `Replay.alpha` (agent/replay.py); `Per.alpha`, `Per.beta0` (config.py) | `dqfd.per.alpha`, `dqfd.per.beta0` (`dqfd.per.beta_final` = 1) | — |
| h_hot | `Cars.h_hot` (config.py); `self.h` (experiments/city.py) | `cars.calibrated.h_hot` | — |
| step 1–step 4 | — (inside the measured run time) | — | — |
