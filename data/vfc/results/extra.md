# Extra 16.3 metrics: vfc

From the existing evaluation runs (tools/extra_metrics.py, metrics/extra.py); means over the seeds.

## Mobility predictor (10.2), level 2: motion math

Greedy's runs (the cars move the same under every method). One prediction per moving, served car per second. The mean error is dominated by slow or stopped cars (a long predicted exit); the median is the typical error.

| scenario | predictions | next-zone acc@1 % | T_exit error, mean s | T_exit error, median s |
| --- | --- | --- | --- | --- |
| S1 | 22,572 | 82.5 | 9.33 | 3.49 |
| S2 | 22,572 | 82.5 | 9.33 | 3.49 |
| S3 | 22,572 | 82.5 | 9.33 | 3.49 |
| S4 | 22,572 | 82.5 | 9.33 | 3.49 |
| S5 | 48,179 | 75.0 | 9.81 | 3.67 |
| S6 | 22,572 | 82.5 | 9.33 | 3.49 |

## Delivery after a zone change (on time %)

Tasks whose car's serving zone at creation differs from the one at close (the last row: how many such tasks a run has, Greedy's runs; few, so these shares are noisy).

| method | S1 | S2 | S3 | S4 | S5 | S6 |
| --- | --- | --- | --- | --- | --- | --- |
| PCNME | 100.0 | 100.0 | 80.2 | 86.1 | 86.7 | 95.8 |
| Greedy | 100.0 | 97.4 | 86.1 | 81.9 | 83.3 | 100.0 |
| D3QN + greedy demos | 100.0 | 100.0 | 87.5 | 87.5 | 85.0 | 93.0 |
| D3QN (scratch) | 100.0 | 97.4 | 72.6 | 87.5 | 86.7 | 77.1 |
| VFC PPO | 100.0 | 98.2 | 82.1 | 86.1 | 81.7 | 88.0 |
| Tang & Wong | 100.0 | 96.4 | 83.4 | 84.6 | 70.6 | 76.5 |
| QECO | 100.0 | 100.0 | 61.4 | 84.7 | 86.7 | 83.9 |
| Random | 100.0 | 94.6 | 73.1 | 80.6 | 79.8 | 95.1 |
| tasks per run | 0 | 1 | 3 | 1 | 2 | 1 |

## Decisions until the on-time rate is stable (≥ 95 % over the last 1000 tasks, to the end)

Median over the seeds that get there; in brackets how many of them (– : none).

| method | S1 | S2 | S3 | S4 | S5 | S6 |
| --- | --- | --- | --- | --- | --- | --- |
| PCNME | 999 (20/20) | 998 (20/20) | 1,001 (17/20) | 8,856 (5/20) | 8,729 (3/20) | 8,527 (8/20) |
| Greedy | 999 (20/20) | 999 (20/20) | 8,955 (9/20) | – (0/20) | 9,243 (1/20) | 3,907 (19/20) |
| D3QN + greedy demos | 999 (20/20) | 998 (20/20) | 1,003 (14/20) | 9,020 (5/20) | 9,056 (2/20) | 8,835 (2/20) |
| D3QN (scratch) | 999 (20/20) | 999 (20/20) | 14,841 (3/20) | 8,586 (4/20) | 9,123 (4/20) | 9,058 (4/20) |
| VFC PPO | 999 (20/20) | 999 (20/20) | 14,095 (2/20) | 8,460 (2/20) | 9,254 (1/20) | 8,494 (5/20) |
| Tang & Wong | 1,000 (20/20) | 1,004 (19/20) | 14,384 (1/20) | 8,820 (1/20) | 9,175 (1/20) | – (0/20) |
| QECO | 999 (20/20) | 1,001 (19/20) | 14,624 (1/20) | 8,819 (1/20) | 9,269 (4/20) | – (0/20) |
| Random | 1,001 (20/20) | 1,008 (18/20) | 13,938 (1/20) | – (0/20) | – (0/20) | – (0/20) |

## NSGA-II fronts (13.2): hypervolume

Per scenario, the reference point is 1.1 x the worst f1 and f2 of its fronts and greedy plans; the ratio is the front's hypervolume over the greedy plan's (> 1: the front dominates more).

| scenario | windows | front points (mean) | hypervolume (mean) | ratio to greedy (mean) |
| --- | --- | --- | --- | --- |
| S1 | 6 | 24 | 0.0104 | 1.048 |
| S2 | 6 | 19 | 0.0183 | 1.260 |
| S3 | 6 | 8 | 0.0648 | 1.603 |
| S4 | 6 | 2 | 0.5336 | 4.261 |
| S5 | 6 | 5 | 0.4476 | 2.624 |
| S6 | 6 | 8 | 0.1543 | 4.053 |
