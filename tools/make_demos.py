"""Demonstrations from saved NSGA-II fronts (PLAN M5.4; final-design 13.4).

Reads the fronts tools/nsga_runs.py wrote (<nsga>/<scenario>-w<k>.json), replays every distinct front plan
with its own w_p (13.4.4), checks each replay gives its saved objectives exactly, and writes one Parquet file per
window to <out>/<scenario>-w<k>.parquet.

    python tools/make_demos.py [--nsga results/nsga] [--out results/demos]
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from pcnme.config import DEFAULT_CONFIG_PATH, config_from_mapping, load_yaml
from pcnme.expert.demos import demonstrate_front, write
from pcnme.expert.window import WindowSpec, discover


def main(argv: Sequence[str] | None = None) -> None:
    root = Path(os.environ.get("PCNME_DATA_DIR", ".")) / "results"
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nsga", type=Path, default=root / "nsga")
    ap.add_argument("--out", type=Path, default=root / "demos")
    args = ap.parse_args(argv)
    cfg = config_from_mapping(load_yaml(DEFAULT_CONFIG_PATH))
    total = experts = 0
    for path in sorted(args.nsga.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        window = discover(cfg, WindowSpec(data["scenario"], data["seed"], data["index"]))
        F = [tuple(p) for p in data["F"]]
        demos = demonstrate_front(cfg, window, data["genomes"], F)  # type: ignore[arg-type]
        for d in demos:
            if (d.objectives.f1, d.objectives.f2) != F[d.plan]:
                raise SystemExit(f"{path.name} plan {d.plan}: the replay does not reproduce its objectives")
        if not demos:
            print(f"{path.name}: fewer than 2 distinct front points: no weight, no demonstrations")
            continue
        out = write(args.out / f"{path.stem}.parquet", demos)
        n = sum(len(d.transitions) for d in demos)
        e = sum(t.expert for d in demos for t in d.transitions)
        total, experts = total + n, experts + e
        ws = ", ".join(f"{d.w:.3f}{' (knee)' if d.is_knee else ''}" for d in demos)
        print(f"{path.name}: {len(demos)} plans (w {ws}), {n} transitions, {e} expert ({e / n:.1%}) -> {out}")
    if total:
        print(f"\nall: {total} transitions, {experts} expert ({experts / total:.1%})")


if __name__ == "__main__":
    main()
