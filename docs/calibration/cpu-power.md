# CPU slot power (estimate, not measured)

No RAPL counter is readable on Windows, so the CPU slot's dynamic power is estimated (QUESTIONS Q52 e):

- CPU: Intel Core i7-6850K, 6 cores / 12 threads, TDP 140 W.
- Idle package power: assumed 15 W (typical for an idle Broadwell-E desktop; not measured here).
- One CPU slot = 4 of the 12 hardware threads (config `hardware.F-GPU`, `c_cpu: 2`, 4 cores each).

p_cpu_slot = (TDP − idle) × threads per slot ÷ threads = (140 − 15) × 4 ÷ 12 ≈ **41.7 W**

TDP is an upper bound on sustained package power, so this is a high estimate. It goes into the config as the
CPU slot's `p_dyn_w` (tag M-estimate); the CPU rows carry no `energy_j`, so their power is the slot's.
