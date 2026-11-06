"""run_testbed.py: the Docker testbed (final-design 15.4; PLAN M9), a thin entry point over
pcnme.testbed.runner. It prints the open items (PLAN M9.3, M9.4); --force runs one testbed run into
data/vfc/testbed/results/. No Docker load
while train.py or evaluate.py runs (PLAN M9: paused during training).
"""

from __future__ import annotations

import argparse
import os

from pcnme.pipeline.evaluate import METHODS

MISSING = """The testbed runs (PLAN M9); open items:
  - The agent: PCNME runs in the zone containers but loses 88 % of the car confirms (drop_no_ack; Greedy is
    fine on the same build). Fix it, then the M9.4 gap for PCNME at 40 cars.
  - Scale: Greedy holds at 40 cars (S2 96.6 % on time vs 99.8 % in fast mode); at 200 cars the busiest zone
    is bound by Python's GIL. More needs a zone split over processes.
  - Topology runs (Q57, low priority): zone-5 joins; zones 3 and 4 killed; the broker restarts.
Run one anyway: --force (or python -m pcnme.testbed.runner).
No Docker load while train.py or evaluate.py runs."""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--maps", nargs="+")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--scenario", default="S2")
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--method", default="greedy", choices=METHODS)
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
    from pcnme.paths import MapPaths, data_dir
    from pcnme.testbed.runner import run

    os.environ.setdefault("PCNME_DATA_DIR", str(data_dir()))
    out = run(
        load_config(),
        args.scenario,
        args.seed,
        args.method,
        args.seconds,
        MapPaths("vfc").testbed,
        args.containers,
    )
    print(Path(out))
    return 0
