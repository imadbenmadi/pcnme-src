# Metrics and log schema

Proposed 28 Oct; confirmed or changed when M3.10 is built. The metrics are those of final-design 16.3. Every run writes the same data in both modes:

- **Parquet files** (the source for the paper's numbers, tables and statistics): `results/<run_id>/tasks.parquet`, `attempts.parquet`, `zones.parquet`, `pipes.parquet`, `learning.parquet`, plus `run.json` with the seed and what the report needs: `method`, `usage` (evaluation), `seconds`, `reward_w`, `training` (the learner's budget total and used, its episodes), `idle_w` (Q46) and `objective` (the 13.2.4 constants). A knee-reference run also writes `window.json` (its window tasks, front, knee and genome; QUESTIONS Q45), and a learning method `<scenario>-<method>-train/` (training.json, model.pt).
- **InfluxDB** (live and replayed views in Grafana only; never the source of a reported number): the same rows as points.

`run_id` = `<scenario>-<method>-<seed>-<mode>`, for example `S2-pcnme-3001-fast`. Fast-mode points are time-stamped at 2026-01-01T00:00:00Z plus the simulated time, so every run starts at the same instant in Grafana.

## Tags common to every point

`run_id`, `scenario` (S1–S6), `method` (pcnme, d3qn-scratch, d3qn-greedy, vfc-ppo, tang-wong, qeco, greedy, random, knee-reference), `seed`, `mode` (fast, testbed).

## Measurements

| Measurement | One point per | Extra tags | Fields |
| --- | --- | --- | --- |
| `task` | closed task (time = creation) | `anchor` zone, `type` (T1–T5), `cls` (hard, soft), `outcome` (on_time, late, or the drop's reason drop_*; dropped vs failed: QUESTIONS Q46), `boulder` (true/false) | `task_id`, `latency_s`, `deadline_s`, `attempts`, `energy_j` (dynamic, all attempts), `energy_wasted_j`, `upload_s`, `forward_s`, `wait_s`, `run_s`, `return_s`, `zone_changed` (0/1), `reward` (only a task dropped before any attempt: the drop reward, 7.3) |
| `attempt` | closed attempt | `anchor`, `executor` (z1–z5, cloud), `slot_type` (gpu, cpu), `result` (delivered, refused_full, refused_late, lost, cancelled, second, skipped, aborted) | `attempt`, `reward` and `reward_kind` (delivered, wasted, dropped; 11.4, final at close + G), `energy_j` (really spent, 12.5), `t_waste_s` (wasted attempts), `decided` (a decision sent it; false for a boulder or the soft fallback) |
| `decision` | placement decision at an anchor (11.3.1; PLAN M4.1) | `anchor`, `kind` (first, redecision, retry), `action` (here, n1-n4, cloud) | `task_id`, `attempt`, `x_<input>` for the 48 inputs in the order of 11.1 (`x_input_size` … `x_w`), `masked_<option>` (the mask's reason, empty when open) |
| `transition` | transition moved to the replay buffer (11.3; PLAN M4.3), time = its decision | `anchor`, `action` | `task_id`, `attempt`, `reward`, `dt_s` (to the anchor's next decision), `next_task_id`, `next_attempt` (that decision; s and s' are its rows in `decision`), `truncated` (moved at the episode end, 11.3.4) |
| `zone` | zone and state period (100 ms) | `zone`, `slot_type` | `rho` (busy share), `queue_hard`, `queue_soft`, `busy`, `refusals` (count in the period), `wan_wait_est_s` |
| `pipe` | pipe and 100 ms | `pipe` (radio-z1 … radio-z5, backhaul-z1 … backhaul-z5, wan) | `wait_s`, `bits` (pushed in the period), `ti` (offered / bandwidth over the period) |
| `mobility` | prediction checked at the car's next attach | `zone` | `next_zone_hit` (0/1), `t_exit_err_s` |
| `learning` | zone and 1 000 updates (PLAN M6.6) | `zone` (global in pretraining), `phase` (pretrain, online) | `update`, `td_loss`, `n_step_loss`, `margin_loss`, `total_loss`, `mean_q` (means over the block), `epsilon`, `expert_agreement` (held-out, pretraining), `demo_upsample_ratio`; `model_version` comes with FL (M8) |
| `episode` | online training episode (PLAN M6.6) | — | `episode`, `seed`, `on_time` (share, 16.3), `mean_e_hat`, `tasks`, `updates`, `mean_loss` |
| `nsga` | window | `window` | `hv`, `front_size`, `knee_f1`, `knee_f2`, `w_knee` |

## Computed after the run (from Parquet, final-design 16.3)

Shares on time / late / dropped / failed; mean, p99 and worst-1 % latency; dynamic and total energy; refusal rate; wasted-attempt energy share; delivery success after a zone change; Jain's index over zone loads; next-zone acc@1 and T_exit error; decisions until the on-time rate reaches 95 %; early-stage performance (first N decisions); hypervolume and distance of the agent's (latency, energy) point to the front. Mean ± 95 % CI over seeds (Student t, n − 1 degrees of freedom). Built so far (`pcnme report`, metrics/stats.py): the four shares and uncovered, the latencies, dynamic and total energy, refusal rate, wasted share, Jain's index and the per-zone load chart, early-stage performance, and every method's (f1, f2) on the knee-reference windows with its distance to the front. Not yet: delivery success after a zone change, next-zone acc@1 and T_exit error (no `mobility` rows are logged yet), decisions until the on-time rate is stable, the hypervolume and the up-sample ratio (in the NSGA-II and learning logs, not in an evaluation run).
