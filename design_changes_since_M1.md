# Design text changed since M1, exactly as on main (614dea6)

## Changes table (rows added since M1)

| Date | Section | Change | Note |
| --- | --- | --- | --- |
| 28 Oct | 8.5, 5.2 | The wait estimate is water-filling over each slot's remaining time r_i (published per slot instead of their sum R): Σ max(0, Ŵ − r_i) = B; still 0 while a slot is free; for c_q = 1 the same as before | CHANGES C4 |
| 28 Oct | 14.4, 18 | Run times are lognormal, fitted to each row's mean and p95; the placeholder p5 factor is 0.75 (was 0.9); measured rows report the fitted p5 next to the measured one | CHANGES C5 (QUESTIONS Q7) |
| 29 Oct | 4.1, 4.2, 6A.3, 15.3, 17 | The radio uplink shares its airtime equally among the cars uploading at once (each advances at R_car ÷ n) instead of one car at a time; W_radio is the extra push time from sharing, still known exactly at t_rx | CHANGES C6 (QUESTIONS Q18) |
| 29 Oct | 18, 14.3 | The car's encode time is a config value, `timing.encode` (M, placeholder 0); the upload starts after it, and B_car does not subtract it again | QUESTIONS Q14 |
| 29 Oct | 15.2 | Every drawn time is rounded to whole nanoseconds and every radio rate to whole bit/s, so one seed gives the same run on every OS | web review of PR #20 |
| 29 Oct | 16.2, 18 | λ_car is divided by the share of task creations with coverage, measured once per map by a positions-only calibration run (tag C); frames made in a gap are reported as uncovered | QUESTIONS Q29 |
| 29 Oct | 16.2, 18 | The hotspot share h (cars on z1's roads) is set by the map calibration run so that z1 = 1.6 × the mean load | QUESTIONS Q30 |
| 29 Oct | 9.3 | The upload timeout T_up starts when the zone's own send has finished; before that only the deadline cap applies, and reaching it drops the attempt with no re-send | CHANGES C7 (QUESTIONS Q31) |
| 29 Oct | 16.2 | S3 shows an overloaded system: under Greedy the fog GPUs saturate (97–100 %); under cloud-only placement the WAN queue grows without limit (M2 test); it no longer says the WAN congests under every method | M3 acceptance runs after C7 |

## C4: section 5.2, zone/{id}/state row

| zone/{id}/state | 0, retained, every 100 ms | Each zone and the cloud → neighbours | zone_id, version (epoch, counter); per slot type with c_q > 0: c, Q, n_hard, n_soft, u (busy slots), B_hard_ms, B_soft_ms, r_ms (the remaining time of each of the c slots, 0 if free); wan_wait_ms and its sample age; load |

## C4: section 8.5 (whole section)

### 8.5 What a node publishes and how the anchor estimates the wait

Per slot type with c_q > 0: c_q, Q_q, n_q^hard, n_q^soft, B_q^hard and B_q^soft (sum of the mean run times of the queued tasks), r_{q,i} (the remaining mean time of the task on each of the c_q slots, 0 for a free slot), u_q (busy slots). The anchor's estimate is 0 while a slot is free (u_q + its own pending sends to q < c_q), and otherwise the water level W at which the class backlog, poured onto the slots in the order they free up, fills them all (CHANGES C4):

```latex
\sum_{i=1}^{c_q} \max(0,\; \hat{W}_q^{hard} - r_{q,i}) = B_q^{hard}, \qquad \sum_{i=1}^{c_q} \max(0,\; \hat{W}_q^{soft} - r_{q,i}) = B_q^{hard} + B_q^{soft}
```

For c_q = 1 this is r + B, exact in the mean; with equal r_{q,i} it equals the pooled (R_q + B) ÷ c_q, which it replaces because one long run (a 720 ms T2 on one GPU slot) would be spread over every slot and the wait overestimated several times over. It treats queued work as divisible, so it can underestimate by up to one task's run per slot. The hard estimate assumes the new task waits behind every queued hard task. The soft one ignores hard tasks that arrive later (optimistic); the status probe covers that (section 9.3). The anchor's own pending sends (R1) are added to the backlog of their class on their q*. The executor's own estimate (8.3) is the same function on its exact state, without pending sends.

## C5: section 14.4 (whole section)

### 14.4 The time table

One row per (task type, hardware class, slot type): mean, p5 and p95 of the run time, the longest step, and the mean dynamic energy per run. Measured by running real YOLO many times (or taken from published benchmarks), with the compressed input sizes beside them. In the main experiments a slot holds a task for a time drawn from this measured distribution (fixed seed); no real YOLO runs. The draw is lognormal, fitted to the row's mean and p95, which both stay exact; its p5 is about 0.75 × the mean for the placeholder rows, and for measured rows the calibration run reports the fitted p5 next to the measured one (CHANGES C5). The quantiles give Ê^low and Ê^high.

## C6: section 4.1, last paragraph

S = the size in bits, B = the pipe's bandwidth, d = a random delay drawn per transfer, t_free = when the pipe is next free. The delay does not hold the pipe: the next transfer can be pushed while this one travels. Waiting costs time, never energy (section 12). A cloud transfer crosses two pipes in series, the zone's own backhaul pipe and the shared WAN pipe. Its packets flow through both at once, so it reaches the WAN pipe d_bh after it starts on the backhaul pipe, and the WAN pipe, the narrowest, sets the push time: T = W_bh + d_bh + W_WAN + S ÷ B_WAN + d_WAN. Fast mode simulates it this way; on the testbed TCP through two links does it by itself. The radio uplink is the one exception: the cars uploading at once share its airtime instead of queueing (4.2, CHANGES C6).

## C6: section 4.2, Radio uplink row

| Radio uplink | car → access point | task input | U(1, 5) ms (18) | each upload pushed at that car's Shannon rate R_car from its distance; while n uploads are in progress each advances at R_car ÷ n | all cars of the zone: one uplink per zone, its airtime shared equally among the cars uploading at once (CHANGES C6) |

## C6: section 6A.3

3. **Anchor deadline.** The whole upload has arrived at t_rx (anchor clock). The anchor sets the task's deadline on its own clock, taking off the upload time: W_radio = the time from the upload's first bit to its last, minus S ÷ R_car (the extra time from sharing the air; known exactly, because the zone runs its uplink; CHANGES C6), S ÷ R_car = its push time alone at the car's rate, δ̂_radio = the radio delay:

```latex
d_A = t_{rx} + B_{car} - \hat{U}_{up}, \qquad \hat{U}_{up} = W_{radio} + S / R_{car} + \hat{\delta}_{radio}
```

## C7: section 9.3, upload timeout

- **Upload timeout:**

```latex
T_{up,k} = \min\big(\kappa_{up}\,(\hat{T}_{in,k} + \hat{\delta}_{reply,k}),\; d_A - t_s + \ell\big), \qquad \ell = 0 \text{ (hard)},\; L_{max} \text{ (soft)}
```

An upload still being sent is not a failure (CHANGES C7): while the zone's own send is going (the input has not yet reached the executor), only the deadline cap d_A + ℓ applies, and reaching it drops the attempt with no re-send. T_up is counted from the end of that send; only a reply missing after it counts as "no reply" (9.1).

## C6: section 15.3, radio paragraph

**The radio uplink:** one pipe per zone whose airtime the cars uploading at once share equally: while n uploads are in progress each advances at that car's current Shannon rate R_car ÷ n (CHANGES C6). On the testbed tc cannot give every virtual car its own rate on one interface, so the pipe runs in software at the zone's intake with the fast-mode code (the car app sends R_car with the upload), and netem adds only the radio's base delay. Coverage is enforced in the car app: a car talks only to the zone covering its simulated position, and to nobody in a gap.

## Rounding: section 15.2, last paragraph

Every random draw comes from seeded generators, one per purpose (arrivals, run times, link delays, mobility), so the same seed gives the same run. Every drawn time (run times, link delays, arrival gaps) is rounded to whole nanoseconds, and every radio rate to whole bit/s, where it is made: log, exp, sin and cos may differ in the last bit between operating systems, and one bit can flip a scheduling decision, so without it the same seed could give different runs on Windows and Linux. No quantity in the design is near 1 ns or 1 bit/s. Fast mode is used for NSGA-II, pretraining, baseline training and sweeps; the testbed for the final evaluation of every method. Training on the testbed alone is impossible: about 240 real-time runs (6 scenarios × 8 methods × 5 seeds) plus training and thousands of NSGA-II evaluations would take weeks.

## Q29, Q30: section 16.2, definitions paragraph

Designed by load, not by number of cars. Two exact definitions: ρ_fog = the city-mean arrival rate per zone × the mix's mean GPU run on F-GPU ÷ c_GPU (the load if every task of the whole mix, boulders included, ran on its anchor's GPU; so S2 and S4–S6 have the same arrival rate and differ only in deadlines, mobility or fog hardware, and in S4–S6 the real fog load is lower because T2 goes to the cloud), and TI_WAN = the cloud-eligible bits per second (soft tasks and boulders) ÷ B_WAN (the WAN load if all of them went to the cloud). One B_WAN, set so that TI_WAN = 0.9 at ρ_fog = 0.9, gives TI_WAN = ρ_fog in S1–S3, because both scale with the arrival rate. The arrival rate per car is divided by the share of task creations made with coverage, measured once per map by a short positions-only calibration run (tag C, QUESTIONS Q29), so that the creations that reach a zone meet ρ_fog; a frame made in a coverage gap is reported as uncovered, apart from the method's drops. The hotspot (40 % of cars through z1) puts z1 at 1.6 × the mean load in every scenario: a share h of the cars stays on the roads between z1's intersections and the rest roam the whole map, with h set by the same calibration run so that z1 holds 40 % of the covered load (tag C, QUESTIONS Q30). With these deadlines a hard task cannot reach the cloud in time (T1: about 140 ms mean round trip against D = 100 ms), so the cloud serves soft tasks and boulders only, which is the reason fog exists (N9).

## Section 16.2, S3 row

| S3 Overload | 1.5 to 2 | Loose | Low | The system is overloaded: under Greedy the fog GPUs saturate (97–100 %); under cloud-only placement the WAN queue grows without limit (M2 test) |

## C6: section 17, radio bullet

- The radio uplink shares its airtime equally among the cars uploading at once, each at its own Shannon rate (fluid processor sharing; real 5G NR and Wi-Fi 6 schedulers share airtime among users within a millisecond). The rates come from free-space path loss (the VFC model), about 540–930 Mbps at 10–300 m, so the radio is not the bottleneck (radio TI about 0.37 in S2), and a 150 KB frame is never held behind a 9 MB T2 clip (CHANGES C6); every option pays the same upload.

## Section 18 rows (C5 time table, Q29 workload, Q30 hotspot, Q14 encode)

| Cars | Hotspot | 40 % of cars routed through z1 (1.6 × the mean load): a share h of the cars stays on z1's roads, h = 0.200 (low mobility) and 0.175 (high) on the VFC grid | R / C | Like Fan et al.'s high-load RSUs: without uneven load, neighbour offloading (C4) is never needed. h from tools/calibrate_map.py so that z1 holds 40 % of the covered load (QUESTIONS Q30) |
| Workload | Arrival rate per car | λ_car = ρ_fog × c_GPU ÷ (mean GPU run of the mix × cars per zone × coverage): S2 = 74 tasks/s per zone, 1.5 per car at 200 cars, 0.3 at 1 000, before the coverage; coverage = the share of task creations with coverage, 0.863 (low mobility) and 0.849 (high) on the VFC grid, at the calibrated hotspot share | C | Sets the scenario load (16.2). Coverage from tools/calibrate_map.py (positions only, 120 s, the smallest fleet); frames made in a gap are reported as uncovered (QUESTIONS Q29) |
| Time table | Mean run, p5, p95, longest step, energy per run | Placeholders, F-GPU GPU slot: T1 8 ms, T3 10 ms, T4 25 ms, T2 720 ms; CPU slot: 90 / 110 / 200 ms, T2 none; cloud GPU about half of the F-GPU time. p5 = 0.75 ×, p95 = 1.3 × the mean, longest step = 0.5 × the mean; run times lognormal, fitted to the mean and p95 (the fit's p5 is 0.748 × the mean) | M | Measured with real YOLO (15.6, Calibrate). Ultralytics' own table (A100 TensorRT: v8n 0.99 ms, v8l 2.39 ms; CPU ONNX: v8n 80.4 ms) puts the placeholders in the right range |
| Timing | Encode time (the car's JPEG encode, 14.3) | 0 s (placeholder) | M | `timing.encode`. Inside L: the upload starts at creation + encode, and B_car = D − time since creation already holds it, so it is not subtracted again (QUESTIONS Q14) |
