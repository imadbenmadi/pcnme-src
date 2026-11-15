# Extra 16.3 metrics: lust

From the existing evaluation runs (tools/extra_metrics.py, metrics/extra.py); means over the seeds.

## Mobility predictor (10.2), level 2: motion math

Greedy's runs (the cars move the same under every method). One prediction per moving, served car per second. The mean error is dominated by slow or stopped cars (a long predicted exit); the median is the typical error.

| scenario | predictions | next-zone acc@1 % | T_exit error, mean s | T_exit error, median s |
| --- | --- | --- | --- | --- |
| S1 | 34,242 | 83.9 | 83.41 | 6.43 |
| S2 | 28,759 | 85.8 | 89.04 | 6.62 |
| S3 | 92,889 | 57.9 | 112.60 | 6.21 |
| S4 | 28,759 | 85.8 | 89.04 | 6.62 |
| S5 | 92,889 | 57.9 | 112.60 | 6.21 |
| S6 | 28,759 | 85.8 | 89.04 | 6.62 |

## Delivery after a zone change (on time %)

Tasks whose car's serving zone at creation differs from the one at close (the last row: how many such tasks a run has, Greedy's runs; few, so these shares are noisy).

| method | S1 | S2 | S3 | S4 | S5 | S6 |
| --- | --- | --- | --- | --- | --- | --- |
| PCNME | 97.1 | 95.8 | 23.3 | 86.8 | 76.5 | 95.1 |
| Greedy | 100.0 | 98.0 | 22.8 | 92.9 | 87.1 | 96.6 |
| D3QN + greedy demos | 100.0 | 100.0 | 23.7 | 91.7 | 84.2 | 97.5 |
| D3QN (scratch) | 100.0 | 96.7 | 21.1 | 89.2 | 80.2 | 82.0 |
| VFC PPO | 93.1 | 88.7 | 20.7 | 90.3 | 85.5 | 79.7 |
| Tang & Wong | 100.0 | 94.6 | 24.5 | 84.7 | 84.2 | 71.9 |
| QECO | 100.0 | 96.5 | 22.0 | 95.9 | 91.1 | 78.8 |
| Random | 94.7 | 94.2 | 20.8 | 85.7 | 73.8 | 71.4 |
| tasks per run | 1 | 2 | 33 | 2 | 3 | 4 |

## Decisions until the on-time rate is stable (≥ 95 % over the last 1000 tasks, to the end)

Median over the seeds that get there; in brackets how many of them (– : none).

| method | S1 | S2 | S3 | S4 | S5 | S6 |
| --- | --- | --- | --- | --- | --- | --- |
| PCNME | 998 (20/20) | 998 (20/20) | – (0/20) | 1,004 (10/20) | 7,938 (4/20) | 1,002 (13/20) |
| Greedy | 999 (20/20) | 998 (19/20) | – (0/20) | 1,005 (8/20) | 8,298 (3/20) | 1,000 (16/20) |
| D3QN + greedy demos | 998 (20/20) | 998 (20/20) | – (0/20) | 1,006 (10/20) | 8,837 (4/20) | 1,000 (14/20) |
| D3QN (scratch) | 999 (20/20) | 999 (20/20) | – (0/20) | 1,018 (11/20) | 9,250 (5/20) | 3,766 (10/20) |
| VFC PPO | 999 (20/20) | 999 (20/20) | – (0/20) | 1,729 (12/20) | 9,493 (5/20) | 3,033 (12/20) |
| Tang & Wong | 999 (20/20) | 1,002 (20/20) | – (0/20) | 1,024 (9/20) | 8,677 (4/20) | 5,425 (4/20) |
| QECO | 999 (20/20) | 1,000 (19/20) | – (0/20) | 1,053 (8/20) | 8,951 (4/20) | 5,145 (4/20) |
| Random | 1,000 (20/20) | 1,006 (20/20) | – (0/20) | 4,030 (6/20) | – (0/20) | 5,536 (6/20) |

## NSGA-II fronts (13.2): hypervolume

Per scenario, the reference point is 1.1 x the worst f1 and f2 of its fronts and greedy plans; the ratio is the front's hypervolume over the greedy plan's (> 1: the front dominates more).

| scenario | windows | front points (mean) | hypervolume (mean) | ratio to greedy (mean) |
| --- | --- | --- | --- | --- |
| S1 | 6 | 22 | 0.0154 | 1.288 |
| S2 | 6 | 17 | 0.0218 | 1.334 |
| S3 | 6 | 4 | 0.7262 | 2.408 |
| S4 | 6 | 2 | 0.3681 | 6.701 |
| S5 | 6 | 1 | 0.5720 | 4.422 |
| S6 | 6 | 7 | 0.7729 | 3.317 |
