# The first full NSGA-II run on the desktop (WSL)

One S2 window at the section 18 budget (population 100 × 100 generations = 10 000 evaluations) on 12 workers.
Run it in WSL, not in Windows: the warm-up of each window is then simulated once and forked for every
evaluation (about 2.7× faster, the same results; Windows has no `fork` and falls back to the plain path).

## 1. Once: a Linux checkout inside WSL

Keep the repository on the WSL file system (`~/`), not on `/mnt/f/`: the Windows drive is several times slower
from WSL.

```bash
sudo apt-get update && sudo apt-get install -y git python3 python3-pip
```

```bash
python3 -m pip install --user "uv==0.13.6"
```

```bash
git clone https://github.com/imadbenmadi/pcnme-src.git ~/pcnme-src
```

```bash
cd ~/pcnme-src && ~/.local/bin/uv sync --locked --python 3.12 --extra dev --extra analysis --extra nsga
```

Check it (about a minute; every test must pass):

```bash
cd ~/pcnme-src && .venv/bin/pytest -q -n auto
```

## 2. The run: S2, window 0 (seed 1100), 100 × 100, 12 workers

One window cannot use window-level parallelism, so its population is spread over the 12 workers
(`--mode population`); each worker keeps its own warm-up.

```bash
cd ~/pcnme-src && .venv/bin/python tools/nsga_runs.py --scenario S2 --windows 1 --population 100 --generations 100 --workers 12 --mode population | tee results/nsga-S2-w0-full.txt
```

Expected time: 10 000 evaluations × about 1.4 s ÷ 12 ≈ 20 min with this cloud box's per-core speed (measure:
the last lines print the core-seconds per evaluation and the scenario estimate). The front is written to
`results/nsga/S2-w0.json`.

For all 20 S2 windows later (window-level parallelism, one window per process):

```bash
cd ~/pcnme-src && .venv/bin/python tools/nsga_runs.py --scenario S2 --windows 20 --population 100 --generations 100 --workers 12 | tee results/nsga-S2-full.txt
```

## 3. Reading the output

```
S2 window 0 (seed 1100): 566 genes
  greedy f1 0.4379 f2 1.1542 (drops 6)
  NSGA-II 10000 evaluations in ... s
  front: N points, M distinct
    f1 ...  f2 ...  w_p ...
  knee (f1, f2): s ..., w~ ..., w ...
```

- **Front check (final-design 16.5): `M distinct` must be at least 5.** If it is below 5 on the full budget,
  the latency–energy conflict is too weak: stop and report it before any training (the energy model is
  revisited, not the code).
- **Greedy must not dominate any front point** (CHANGES C9 guarantees it; the greedy line and the front can be
  compared directly: a front point with both f1 and f2 above greedy's would be a bug).
- **The knee's w** is one window's weight; w* is the median over the windows (printed when several run).
- Then the demonstrations, which also re-check that every front plan replays to its objectives exactly:

```bash
cd ~/pcnme-src && .venv/bin/python tools/make_demos.py
```

Send back `results/nsga-S2-w0-full.txt` and `results/nsga/S2-w0.json`.
