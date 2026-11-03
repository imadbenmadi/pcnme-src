"""The first fast-mode results (docs/results-first.md): train and evaluate every method on S1-S6 with a
reduced, equal training budget, N runs at a time. Restartable: a job whose last evaluation seed already has a
run directory is skipped.

    python tools/first_results.py --scenarios S1,S2,S3 --methods greedy,random,d3qn-scratch --parallel 4
    python tools/first_results.py --scenarios S1 --methods pcnme --demos results/demos-first

Every job is `pcnme run --scenario S --method M --seeds 3000-(3000+seeds-1) --budget B --out OUT`; pcnme gets
its scenario's demonstrations (<demos>/<S>-w*.parquet), d3qn-greedy plans greedy demonstrations on `windows`
windows. Logs: <out>/logs/<S>-<M>.log.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("greedy", "random", "d3qn-scratch", "d3qn-greedy", "vfc-ppo", "tang-wong", "qeco", "pcnme")


def job(args: argparse.Namespace, scenario: str, method: str) -> list[str] | None:
    last = args.out / f"{scenario}-{method}-{3000 + args.seeds - 1}-fast"
    if last.exists():
        return None  # done
    cmd = [sys.executable, "-m", "pcnme.experiments.cli", "run", "--scenario", scenario, "--method", method]
    cmd += ["--seeds", f"3000-{3000 + args.seeds - 1}", "--out", str(args.out)]
    if method not in ("greedy", "random"):
        cmd += ["--budget", str(args.budget)]
    if method == "d3qn-greedy":
        cmd += ["--windows", str(args.windows)]
    if method == "pcnme":
        demos = sorted(args.demos.glob(f"{scenario}-w*.parquet"))
        if not demos:
            print(f"skip {scenario} pcnme: no demonstrations in {args.demos}", flush=True)
            return None
        cmd += ["--demos", *map(str, demos)]
    return cmd


def main() -> None:
    data = Path(os.environ.get("PCNME_DATA_DIR", ".")).resolve() / "results"
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenarios", default="S1,S2,S3")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budget", type=int, default=1_000_000, help="training decisions, every learning method")
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--out", type=Path, default=data / "first")
    ap.add_argument("--demos", type=Path, default=data / "demos-first")
    args = ap.parse_args()
    (args.out / "logs").mkdir(parents=True, exist_ok=True)
    queue = [
        (s, m, c)
        for s in args.scenarios.split(",")
        for m in args.methods.split(",")
        if (c := job(args, s, m)) is not None
    ]
    running: list[tuple[str, subprocess.Popen[bytes], float]] = []
    while queue or running:
        while queue and len(running) < args.parallel:
            s, m, cmd = queue.pop(0)
            log = (args.out / "logs" / f"{s}-{m}.log").open("wb")
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            running.append((f"{s}-{m}", p, time.monotonic()))
            print(f"start {s}-{m}", flush=True)
        time.sleep(10)
        for name, p, t0 in list(running):
            if p.poll() is not None:
                running.remove((name, p, t0))
                mins = (time.monotonic() - t0) / 60
                print(f"end {name}: exit {p.returncode} after {mins:.0f} min", flush=True)


if __name__ == "__main__":
    main()
