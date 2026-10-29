"""The 48 inputs over every decision of Greedy runs (PLAN M4 acceptance; final-design 11.1).

Per input: min, mean and max over all decisions, the share of decisions where it is 0 and where it is at its
cap, and a flag when it is always 0 or always at its cap. Default: S2, 60 s, training seeds 2000-2002.

    python tools/input_ranges.py [--scenario S2] [--seconds 60] [--seeds 2000 2001 2002]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from pcnme.agent.inputs import EXIT_CAP, INPUT_NAMES, TIME_CAP
from pcnme.config import Config, load_config
from pcnme.experiments.city import City

UNIT = ("input_size", "D_norm", "exit", "w")  # in [0, 1]; energies, flags and the one-hot too


def cap(name: str) -> float:
    """The largest value an input can take (11.1)."""
    if name == "exit":
        return EXIT_CAP
    if name in UNIT or name.endswith(("_energy", "_present")) or name.startswith("next_") or name == "hard":
        return 1.0
    return TIME_CAP


@dataclass(frozen=True, slots=True)
class Row:
    name: str
    lo: float
    mean: float
    hi: float
    zero: float  # share of decisions where it is 0
    at_cap: float  # share where it is at its cap

    @property
    def flag(self) -> str:
        if self.hi == 0.0:
            return "always 0"
        if self.lo == cap(self.name):
            return "always at cap"
        return ""


def decisions(cfg: Config, scenario: str, seconds: float, seeds: Sequence[int]) -> list[tuple[float, ...]]:
    out: list[tuple[float, ...]] = []
    for seed in seeds:
        city = City(cfg, scenario, "greedy", seed).run(seconds)
        out += [d.inputs for z in city.zones.values() for d in z.decisions]
    return out


def table(xs: Sequence[tuple[float, ...]]) -> list[Row]:
    n = len(xs)
    rows = []
    for i, name in enumerate(INPUT_NAMES):
        col = [x[i] for x in xs]
        c = cap(name)
        rows.append(
            Row(
                name,
                min(col),
                sum(col) / n,
                max(col),
                sum(v == 0.0 for v in col) / n,
                sum(v == c for v in col) / n,
            )
        )
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="S2")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2000, 2001, 2002])
    args = ap.parse_args(argv)
    xs = decisions(load_config(), args.scenario, args.seconds, args.seeds)
    print(f"{args.scenario}, Greedy, {args.seconds:g} s, seeds {args.seeds}: {len(xs)} decisions\n")
    print("| # | input | min | mean | max | = 0 | at cap | flag |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for i, r in enumerate(table(xs), 1):
        stats = f"{r.lo:.4g} | {r.mean:.4g} | {r.hi:.4g} | {r.zero:.1%} | {r.at_cap:.1%}"
        print(f"| {i} | {r.name} | {stats} | {r.flag} |")


if __name__ == "__main__":
    main()
