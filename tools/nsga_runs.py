"""NSGA-II runs on full windows: fronts, knees, w*, and the compute estimate (PLAN M5.2-M5.3, 16.5).

For each window of a scenario: discover it (greedy), run NSGA-II with the given budget, report the front (its
distinct points: 16.5 asks for at least 5 in S2), the knee and its w, every point's w_p, and the median w*
over the windows. Each front is written to <out>/<scenario>-w<k>.json (genomes, objectives, w_p) for the
demonstrations (M5.4). The windows run in parallel, one per process (--mode windows), each from its warm-up
by fork on Linux and WSL; neither changes a result. The estimate scales the measured core-seconds per
evaluation, every overhead included, to the section 18 budget.

    python tools/nsga_runs.py --scenario S2 --windows 2 --population 10 --generations 5 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

from pcnme.config import DEFAULT_CONFIG_PATH, config_from_mapping, load_yaml
from pcnme.expert.knee import knee, w_star, weights
from pcnme.expert.nsga import run_nsga, run_windows
from pcnme.expert.window import FORK, WindowSpec, discover, window_seed


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="S2")
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--population", type=int, default=10)
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument(
        "--out", type=Path, default=Path(os.environ.get("PCNME_DATA_DIR", ".")) / "results" / "nsga"
    )
    ap.add_argument(
        "--mode",
        choices=("windows", "population"),
        default="windows",
        help="parallel over windows (one per process, the default) or over each window's population",
    )
    args = ap.parse_args(argv)
    source = load_yaml(DEFAULT_CONFIG_PATH)
    cfg = config_from_mapping(source)
    args.out.mkdir(parents=True, exist_ok=True)
    specs = [WindowSpec(args.scenario, window_seed(cfg, args.scenario, k), k) for k in range(args.windows)]
    print(f"warm-up reuse by fork: {'on' if FORK else 'off (no fork on this OS)'}")
    t0 = time.perf_counter()
    if args.mode == "windows":
        fronts = run_windows(
            source, specs, population=args.population, generations=args.generations, workers=args.workers
        )
    else:
        fronts = [
            run_nsga(
                source,
                discover(cfg, s),
                population=args.population,
                generations=args.generations,
                workers=args.workers,
            )
            for s in specs
        ]
    wall = time.perf_counter() - t0
    knees = []
    for spec, front in zip(specs, fronts, strict=True):
        window = front.window
        kn = knee(front.F)
        knees.append(kn)
        wp = weights(front.F)
        print(f"\n{args.scenario} window {spec.index} (seed {spec.seed}): {len(window.tasks)} genes")
        print(f"  greedy f1 {window.greedy.f1:.4f} f2 {window.greedy.f2:.4f} (drops {window.greedy.drops})")
        print(f"  NSGA-II {front.evaluations} evaluations in {front.seconds:.0f} s")
        print(f"  front: {len(front.F)} points, {front.distinct} distinct")
        for f1, f2 in front.F:
            print(f"    f1 {f1:.4f}  f2 {f2:.4f}  w_p {wp.get((f1, f2), float('nan')):.4f}")
        if kn is not None:
            at = f"({kn.point[0]:.4f}, {kn.point[1]:.4f})"
            print(f"  knee {at}: s {kn.slope:.4f}, w~ {kn.w_tilde:.4f}, w {kn.w:.4f}")
        (args.out / f"{args.scenario}-w{spec.index}.json").write_text(
            json.dumps(
                {
                    "scenario": args.scenario,
                    "index": spec.index,
                    "seed": spec.seed,
                    "population": args.population,
                    "generations": args.generations,
                    "genomes": front.genomes,
                    "F": front.F,
                    "w_p": [wp.get(p) for p in front.F],
                    "knee": None if kn is None else {"point": kn.point, "w": kn.w},
                },
                indent=1,
            ),
            encoding="utf-8",
        )
    if any(k is not None for k in knees):
        print(f"\nw* (median knee weight over {args.windows} windows) = {w_star(knees):.4f}")
    evals = sum(f.evaluations for f in fronts)
    used = min(args.workers, args.windows) if args.mode == "windows" else args.workers
    core_s = wall * used / evals  # measured core-seconds per evaluation, every overhead included
    full = cfg.nsga2.population * cfg.nsga2.generations * cfg.nsga2.windows_per_scenario
    print(f"\n{evals} evaluations in {wall:.0f} s on {used} processes: {core_s:.2f} core-s per evaluation")
    for cores in (4, 12):
        hours = full * core_s / cores / 3600
        print(f"section 18 budget ({full} evaluations per scenario) on {cores} cores: about {hours:.1f} h")


if __name__ == "__main__":
    main()
