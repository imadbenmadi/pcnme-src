"""PLAN M9.4 (final-design 16.3): the same scenario, method and seed in fast mode and on the testbed; the gap.

Run the testbed first (python -m pcnme.testbed.runner ...), then:
    python tools/testbed_gap.py --scenario S2 --method greedy --seed 3000 --seconds 60

It runs the fast-mode city with the testbed's fleet (200 cars: 10 containers x 20) on the same evaluation
seed, writes it next to the testbed run under data/vfc/testbed/results/, and prints one Markdown table of the
16.3 metrics for both modes and their difference. Numbers come from the Parquet files only (CLAUDE.md).
"""

from __future__ import annotations

import argparse
from typing import Any

from pcnme.config import load_config
from pcnme.experiments.city import City
from pcnme.experiments.runner import deployed, method_config
from pcnme.metrics.stats import run_metrics
from pcnme.paths import MapPaths
from pcnme.pipeline import checkpoint
from pcnme.pipeline.evaluate import METHODS

ROWS = (
    ("on_time", "on-time share", "{:.3f}"),
    ("late", "late share", "{:.3f}"),
    ("dropped", "dropped share", "{:.3f}"),
    ("failed", "failed share", "{:.3f}"),
    ("latency_mean_s", "latency mean (s)", "{:.4f}"),
    ("latency_p99_s", "latency p99 (s)", "{:.4f}"),
    ("latency_worst1_s", "latency worst 1 % (s)", "{:.4f}"),
    ("energy_per_task_j", "energy per task (J)", "{:.3f}"),
    ("refusal_rate", "refusal rate", "{:.3f}"),
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenario", default="S2")
    p.add_argument("--method", default="greedy", choices=METHODS)
    p.add_argument("--seed", type=int, default=3000)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--cars", type=int, default=200)
    a = p.parse_args()
    cfg = method_config(load_config(), a.method)
    results = MapPaths("vfc").testbed / "results"
    testbed = results / f"{a.scenario}-{a.method}-{a.seed}-testbed"
    if not (testbed / "tasks.parquet").exists():
        raise SystemExit(f"no testbed run at {testbed}: run python -m pcnme.testbed.runner first")
    factory: Any = a.method
    if a.method not in ("greedy", "random"):  # deployed exactly as in the zone container (Q42, Q54)
        trained = checkpoint.load(MapPaths("vfc").train(a.scenario, a.method) / "trained.pt")
        factory = deployed(cfg, trained, a.seed)
    city = City(cfg, a.scenario, factory, a.seed, n_cars=a.cars, usage="evaluation", method_name=a.method)
    fast = city.run(a.seconds).write(results, {"method": a.method})
    m_fast = run_metrics(fast, cfg.fast_mode.early_decisions).metrics
    m_bed = run_metrics(testbed, cfg.fast_mode.early_decisions).metrics
    lines = [f"{a.scenario}, {a.method}, seed {a.seed}, {a.seconds:g} s, {a.cars} cars", ""]
    lines += ["| metric | fast mode | testbed | testbed - fast |", "| --- | --- | --- | --- |"]
    for key, name, fmt in ROWS:
        f, t = m_fast[key], m_bed[key]
        lines.append(f"| {name} | {fmt.format(f)} | {fmt.format(t)} | {fmt.format(t - f)} |")
    out = results.parent / f"gap-{a.scenario}-{a.method}-{a.seed}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines) + f"\n\nwritten to {out}")


if __name__ == "__main__":
    main()
