# PCNME on LuST (Luxembourg): results (14 Nov 2026)

**Setup.** LuST (Codeca et al. 2015): a 1.2 km crop of central Luxembourg with SUMO traces replayed at three
times of day (QUESTIONS Q58: 11:00 for S1, 15:00 for S2, S4, S6, the 08:00 peak for S3 and S5), 4 active fog
zones, 200 cars, 60 s episodes, 6 scenarios (final-design 16.2). Every method was trained on LuST itself (w* =
0.344; VFC's is 0.414) and evaluated on the same 20 seeds (3000–3019) in every scenario, 960 runs in total.
Every number below is the mean over those seeds; the 95 % intervals are in `data/lust/results/report.md`, the
paired tests in `data/lust/results/paired.md`. Bold marks the best method in each column. evaluate.py writes these
tables.

<!-- tables:begin (written by evaluate.py) -->
**On time (%)**

| method | S1 | S2 | S3 | S4 | S5 | S6 |
| --- | --- | --- | --- | --- | --- | --- |
| PCNME | 99.8 | **99.8** | 49.4 | 94.5 | 92.2 | 96.4 |
| Greedy | **99.9** | 99.6 | 52.6 | 93.6 | 90.9 | **97.0** |
| D3QN + greedy demos | **99.9** | **99.8** | **53.7** | 94.0 | 92.0 | 96.7 |
| D3QN (scratch) | 99.8 | 99.6 | 49.0 | 94.7 | 92.4 | 92.5 |
| VFC PPO | 99.6 | 98.9 | 44.5 | **94.9** | **92.5** | 94.0 |
| Tang & Wong | 99.7 | 98.7 | 48.8 | 94.2 | 91.8 | 81.2 |
| QECO | 99.7 | 99.1 | 46.3 | 94.1 | 92.0 | 84.8 |
| Random | 99.4 | 98.3 | 43.0 | 92.0 | 89.4 | 84.5 |

**p99 latency (ms)**

| method | S1 | S2 | S3 | S4 | S5 | S6 |
| --- | --- | --- | --- | --- | --- | --- |
| PCNME | 171 | 190 | 1958 | 184 | 191 | 742 |
| Greedy | **162** | 211 | 1950 | **168** | **170** | **603** |
| D3QN + greedy demos | 165 | **181** | 1961 | 172 | 175 | 756 |
| D3QN (scratch) | 186 | 309 | 1964 | 204 | 224 | 1061 |
| VFC PPO | 378 | 833 | 1965 | 204 | 201 | 1233 |
| Tang & Wong | 221 | 491 | **1949** | 209 | 208 | 1391 |
| QECO | 206 | 428 | 1961 | 181 | 195 | 1295 |
| Random | 323 | 573 | 1971 | 285 | 284 | 1103 |

**Energy per task (J)**

| method | S1 | S2 | S3 | S4 | S5 | S6 |
| --- | --- | --- | --- | --- | --- | --- |
| PCNME | 2.24 | **2.22** | 2.90 | 2.13 | 2.08 | **2.66** |
| Greedy | **2.22** | 2.27 | **2.71** | **2.12** | **2.06** | 2.70 |
| D3QN + greedy demos | 2.24 | 2.23 | 2.75 | 2.13 | 2.08 | 2.67 |
| D3QN (scratch) | 2.33 | 2.53 | 2.95 | 2.14 | 2.09 | 3.08 |
| VFC PPO | 3.05 | 3.31 | 3.22 | 2.15 | 2.09 | 4.01 |
| Tang & Wong | 3.24 | 3.23 | 3.08 | 2.17 | 2.10 | 4.32 |
| QECO | 3.23 | 3.01 | 3.15 | 2.13 | 2.09 | 4.05 |
| Random | 3.34 | 3.53 | 3.20 | 2.21 | 2.14 | 4.47 |
<!-- tables:end -->

## Findings

- **S1, S2 (light to medium load):** every method is at its ceiling (98–100 % on time). PCNME and Greedy differ
  by under 0.2 pp; PCNME beats VFC PPO by 0.3–0.9 pp (p < 0.001).
- **S3 (peak hour, 08:00, overloaded):** every method falls to 43–54 % on time. Unlike the VFC grid, where PCNME
  was best in S3 (98.4 % vs Greedy 92.3 %), on LuST Greedy (52.6 %) and D3QN + greedy demos (53.7 %) beat PCNME
  (49.4 %; −3.3 pp vs Greedy, p = 0.005). The peak's real traffic is far denser than the grid's, so S3 here is a
  hard overload where no placement keeps up; the learners still beat VFC PPO, QECO and Random.
- **S4, S5 (tight deadlines):** PCNME beats Greedy by 0.9 pp (S4) and 1.2 pp (S5), both p < 0.001, the same
  direction as on the grid; VFC PPO is level with it (−0.3 pp).
- **S6 (mixed fog):** Greedy and the D3QN methods are level (96–97 %); PCNME −0.7 pp vs Greedy, not significant
  (p = 0.09); PCNME beats VFC PPO by 2.4 pp (p < 0.001).
- **Overall:** the grid's results carry over to real traffic where the network has room (S1, S2, S4, S5, S6);
  the advantage PCNME had in the grid's overload (S3) does not survive LuST's peak hour.
