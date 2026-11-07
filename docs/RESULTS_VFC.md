# PCNME on the VFC grid: results (5 Nov 2026)

**Setup.** VFC grid: 1.2 km, 4 active fog zones (z5 joins only in the topology experiment), 200 cars, 60 s
episodes, 6 scenarios (final-design 16.2). There are 8 methods (16.1). Each method was evaluated on the same 20
seeds (3000–3019) in every scenario, 960 runs in total. Every number below is the mean over those seeds; the 95 %
intervals are in `data/vfc/results/report.md`. Bold marks the best method in each column. evaluate.py writes
these tables.

**S6 changed on 6 Nov (QUESTIONS Q60).** S6 now has loose deadlines; it was retrained and evaluated alone
(7 Nov), with w* kept at 0.414. Its first version, with tight deadlines, is kept as a stress test:
"S6 tight (archived)".

<!-- tables:begin (written by evaluate.py) -->
**On time (%)**

| method | S1 | S2 | S3 | S4 | S5 | S6 | S6 tight (archived) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PCNME | **99.9** | **99.8** | **98.4** | **94.2** | **93.3** | 92.1 | 56.8 |
| Greedy | **99.9** | **99.8** | 92.3 | 92.6 | 92.2 | **96.9** | 55.4 |
| D3QN + greedy demos | **99.9** | **99.8** | 97.2 | 93.5 | 92.7 | 92.3 | 56.2 |
| D3QN (scratch) | **99.9** | 99.6 | 86.4 | 93.6 | 93.1 | 91.9 | 56.2 |
| VFC PPO | 99.8 | 99.1 | 86.3 | 93.8 | 93.0 | 89.3 | 64.2 |
| Tang & Wong | 99.8 | 98.6 | 80.1 | 93.2 | 92.7 | 79.2 | 61.4 |
| QECO | 99.8 | 98.6 | 78.4 | 93.2 | 92.8 | 80.9 | 59.2 |
| Random | 99.5 | 98.1 | 81.3 | 90.9 | 90.3 | 81.0 | **65.6** |

**p99 latency (ms)**

| method | S1 | S2 | S3 | S4 | S5 | S6 | S6 tight (archived) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PCNME | 167 | 228 | **605** | 228 | 187 | 1076 | 453 |
| Greedy | **157** | 182 | 932 | **167** | **168** | **638** | **261** |
| D3QN + greedy demos | 159 | **177** | 751 | 173 | 172 | 1007 | 290 |
| D3QN (scratch) | 174 | 291 | 1597 | 196 | 209 | 1195 | 331 |
| VFC PPO | 257 | 632 | 1409 | 209 | 209 | 1159 | 811 |
| Tang & Wong | 193 | 546 | 1702 | 194 | 192 | 1556 | 740 |
| QECO | 193 | 540 | 1742 | 186 | 184 | 1432 | 614 |
| Random | 292 | 607 | 1528 | 294 | 294 | 1193 | 853 |

**Energy per task (J)**

| method | S1 | S2 | S3 | S4 | S5 | S6 | S6 tight (archived) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PCNME | 2.17 | **2.21** | **2.36** | 2.12 | 2.11 | 3.22 | 1.72 |
| Greedy | **2.16** | 2.22 | 2.81 | **2.10** | **2.09** | **2.83** | **1.59** |
| D3QN + greedy demos | 2.17 | **2.21** | 2.57 | 2.11 | 2.10 | 3.33 | 1.61 |
| D3QN (scratch) | 2.19 | 2.47 | 3.71 | 2.11 | 2.10 | 3.19 | 1.61 |
| VFC PPO | 2.62 | 3.15 | 3.92 | 2.12 | 2.12 | 4.18 | 2.61 |
| Tang & Wong | 3.04 | 3.25 | 3.97 | 2.13 | 2.11 | 4.37 | 2.17 |
| QECO | 3.04 | 3.23 | 3.96 | 2.11 | 2.10 | 4.19 | 1.95 |
| Random | 3.19 | 3.50 | 3.98 | 2.18 | 2.16 | 4.50 | 2.66 |
<!-- tables:end -->

**S3, the overload scenario (fog load ρ = 1.5).** PCNME is best on all three metrics at once: 98.4 % on time (next
best: D3QN with greedy demonstrations 97.2 %, Greedy 92.3 %, the other learners 78–86 %), the lowest p99 (605 ms
against 751–1742 ms) and the lowest energy (2.36 J per task). The difference from every baseline is statistically
real: a paired t test over the 20 seeds gives p = 0.002 against D3QN with greedy demonstrations and p < 0.001 against
each other method (`paired.md`). The WAN is saturated, with a queue of up to 3.7 s, so every method keeps tasks
off the cloud (0.4 % of attempts for PCNME and Greedy alike). PCNME wins by how it spreads load across the
neighbouring zones. Expert pretraining clearly matters: D3QN from scratch reaches 86.4 % on the same budget.

**S6 with tight deadlines (the stress test, archived).** All methods are at 55–66 % on time with about 43 % of
tasks dropped. Random and VFC PPO come out above PCNME, at much higher energy (2.6 against 1.7 J per task) and p99 (about 830 against 453 ms).
This is not a placement failure (`tools/s6_feasibility.py`, QUESTIONS Q60):
- With empty queues, 0.0 % of the 184 020 logged S6 tasks are infeasible. A hard task from a CPU-only zone reaches a
  GPU neighbour in about 32 ms, against a 50 ms deadline; locally it would take 72.5 ms.
- The limit is capacity against slack. Hard tasks meet a tight deadline only on a GPU slot. T1 and T3 alone use
  3.19 of the 4 GPU slots (ρ 0.80) and have 14–20 ms of slack against a 22.6 ms run, so even an ideal shared queue
  misses 29–36 % of them.
- S6 therefore measures which tasks a method sacrifices. PCNME, with w* = 0.414 weighing energy, keeps soft T4
  tasks on time (97.9 %) and sacrifices hard T1 tasks (44.7 % on time); Random does the opposite (T1 58.3 %, T4 74.0 %).
- Only 1 of S6's 6 NSGA-II windows gave demonstrations (Q59), so PCNME was pretrained on the whole map's demonstrations.

**S6 with loose deadlines (the scenario now).** The change works as intended: every method moves from 55–66 % to
79–97 % on time, so S6 now separates placement quality instead of measuring a capacity ceiling. Here **Greedy is
the best method**: 96.9 % on time, p99 638 ms, 2.83 J per task. PCNME reaches 92.1 % (p < 0.001 against Greedy),
level with both D3QN variants (92.4 % and 91.9 %, no significant difference), and clearly ahead of VFC PPO (89.3 %,
p = 0.004) and of Random, QECO and Tang & Wong (79–81 %, p < 0.001), with lower p99 and energy than all four.
What is known so far:
- It is not the expert data: all 6 S6 windows gave demonstrations (7–9 front plans each; the tight S6 had 1), and
  pretraining agreed with the expert on 62 % of the held-out decisions, as in S3 (59 %).
- The learners were still improving when the budget ran out: their training on-time rate rose to 93–94 % in the
  last 20 episodes (1 M decisions, about 110 episodes); PCNME's from 90.5 % to 93.1 %. Greedy needs no training.
- Greedy's rule, the earliest estimated finish among the options, is close to ideal when nodes differ by a fixed
  factor (a CPU-only zone runs T1 3 times slower): it reads each option's run time directly. The agent sees one run
  estimate per option (48 inputs); the 16.4 ablation "60 inputs: both slot types per option" targets exactly this.
Open: the S6 analysis (where PCNME's decisions differ from Greedy's, by zone and task type; a longer budget; the
60-input ablation), PLAN T6.

**Testbed gap (Greedy, Docker, 40 cars at the same per-zone load as 200).** S1: 96.9 % on time on the testbed
against 99.9 % in fast mode. S2: 96.6 % against 99.8 %, with p99 199 ms against 171 ms. At 200 cars the busiest
zone's Python process is limited by the GIL (48.9 % on time). Splitting a zone over several processes is the fix.

**Open.** PCNME runs in the testbed zones but loses 88 % of the car confirmations (drop_no_ack, while Greedy is
fine on the same build). This is a testbed bug, scheduled after Friday.

**Budget used.**
- NSGA-II: 36 windows (6 per scenario), population 40 × 20 generations, 2 h 40 min, which gave w* = 0.414.
- Training: 1 M decisions per method and scenario, 36 jobs, 35 h 59 min on 11 workers (i7-6850K).
- Evaluation: 960 runs, about 5.5 h (49 CPU-hours).
- Testbed: about 10 runs of 60 s.

**Next.**
1. **S6 analysis:** why Greedy wins on unequal nodes (decisions compared with Greedy's, a longer budget, the
   60-input ablation).
2. **LuST:** traces, then LuST training and evaluation.
3. **Testbed:** fix the PCNME confirmation bug, run the agent's testbed gap, and split zones over processes for
   200 cars.

**See it.** Run `uv run python view.py` and open `#vfc/S3/pcnme/3000/fast/greedy` for PCNME next to Greedy on the
same seed in S3.
