"""Small NSGA-II runs on full windows: fronts, knees, w*, and the compute estimate (PLAN M5.2-M5.3, 16.5).

For each window of a scenario: discover it (greedy), time one evaluation, run NSGA-II with the given budget,
report the front (its distinct points: 16.5 asks for at least 5 in S2), the knee and its w, every point's w_p,
and the median w* over the windows. Each front is written to <out>/<scenario>-w<k>.json (genomes, objectives,
w_p) for the demonstrations (M5.4). The estimate scales the measured evaluation time to the section 18 budget.

    python tools/nsga_runs.py --scenario S2 --windows 2 --population 10 --generations 5 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

from pcnme.config import DEFAULT_CONFIG_PATH, config_from_mapping, load_yaml
from pcnme.expert.knee import knee, w_star, weights
from pcnme.expert.nsga import run_nsga
from pcnme.expert.window import WindowSpec, discover, evaluate, window_seed


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
    args = ap.parse_args(argv)
    source = load_yaml(DEFAULT_CONFIG_PATH)
    cfg = config_from_mapping(source)
    args.out.mkdir(parents=True, exist_ok=True)
    knees, eval_s = [], []
    for k in range(args.windows):
        spec = WindowSpec(args.scenario, window_seed(cfg, args.scenario, k), k)
        window = discover(cfg, spec)
        t = time.perf_counter()
        evaluate(cfg, window, window.greedy_genome)
        eval_s.append(time.perf_counter() - t)
        front = run_nsga(
            source, window, population=args.population, generations=args.generations, workers=args.workers
        )
        kn = knee(front.F)
        knees.append(kn)
        wp = weights(front.F)
        print(f"\n{args.scenario} window {k} (seed {spec.seed}): {len(window.tasks)} genes")
        print(f"  greedy f1 {window.greedy.f1:.4f} f2 {window.greedy.f2:.4f} (drops {window.greedy.drops})")
        cost = f"NSGA-II {front.evaluations} evaluations in {front.seconds:.0f} s"
        print(f"  one evaluation {eval_s[-1]:.2f} s; {cost}")
        print(f"  front: {len(front.F)} points, {front.distinct} distinct")
        for f1, f2 in front.F:
            print(f"    f1 {f1:.4f}  f2 {f2:.4f}  w_p {wp.get((f1, f2), float('nan')):.4f}")
        if kn is not None:
            at = f"({kn.point[0]:.4f}, {kn.point[1]:.4f})"
            print(f"  knee {at}: s {kn.slope:.4f}, w~ {kn.w_tilde:.4f}, w {kn.w:.4f}")
        (args.out / f"{args.scenario}-w{k}.json").write_text(
            json.dumps(
                {
                    "scenario": args.scenario,
                    "index": k,
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
    per_eval = statistics.mean(eval_s)
    full = cfg.nsga2.population * cfg.nsga2.generations * cfg.nsga2.windows_per_scenario
    print(f"\nmean wall time of one evaluation: {per_eval:.2f} s (one process)")
    for cores in (4, 12):
        hours = full * per_eval / cores / 3600
        print(f"section 18 budget ({full} evaluations per scenario) on {cores} cores: about {hours:.0f} h")


if __name__ == "__main__":
    main()
