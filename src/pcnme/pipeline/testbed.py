"""run_testbed.py: the Docker testbed (final-design 15.4; PLAN M9), a thin entry point over
pcnme.testbed.runner. It stops with what is missing until PLAN M9.3 (a scale the containers hold) and M9.4
(the fast-mode vs testbed gap) are done; --force runs the runner anyway (development only). No Docker load
while train.py or evaluate.py runs (PLAN M9: paused during training).
"""

from __future__ import annotations

import argparse
import os

MISSING = """The testbed is not ready for experiments yet (PLAN M9):
  - M9.3: at 200 cars the zone containers cannot hold the load (654 of 9 121 tasks on time vs about 100 % in
    fast mode). Next: a reduced scale, 40-60 cars (--containers 2 or 3), the same per-zone load; one
    timeboxed performance session if it still lags.
  - M9.4: the fast-mode vs testbed gap at that scale (tools/testbed_gap.py), Greedy first, then the agent
    (Q54: CPU-only torch in the zone image).
  - Topology runs (Q57), three separate 120 s runs with the event at 40 s: zone-5 joins; zones 3 and 4
    killed; the broker restarts. Low priority.
Paused until training finishes: no Docker load runs while train.py or evaluate.py runs."""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--maps", nargs="+")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--scenario", default="S2")
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--method", default="greedy", choices=("greedy", "random"))
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--containers", type=int, default=2, help="car containers of 20 cars (M9.3: 2-3)")
    ap.add_argument("--smoke", action="store_true", help="check the entry point only (no Docker)")
    ap.add_argument("--force", action="store_true", help="run the runner although M9.3/M9.4 are open")
    args = ap.parse_args(argv)
    maps = ["vfc"] if args.all else args.maps
    if maps != ["vfc"]:
        print("The testbed runs the VFC grid only (the car containers walk it); LuST is not planned there.")
        return 1
    if not args.force:
        print(MISSING)
        return 0 if args.smoke else 1
    from pathlib import Path

    from pcnme.config import load_config
    from pcnme.paths import data_dir
    from pcnme.testbed.runner import run

    os.environ.setdefault("PCNME_DATA_DIR", str(data_dir()))
    out = run(
        load_config(),
        args.scenario,
        args.seed,
        args.method,
        args.seconds,
        data_dir() / "vfc",
        args.containers,
    )
    print(Path(out))
    return 0
