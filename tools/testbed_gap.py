"""PLAN M9.4 (final-design 16.3): the same scenario, method and seed in fast mode and on the testbed; the gap.

Run the testbed first (python -m pcnme.testbed.runner ...), then:
    python tools/testbed_gap.py --scenario S2 --method greedy --seed 3000 --seconds 60

It runs the fast-mode city with the testbed's fleet (200 cars: 10 containers x 20) on the same evaluation
seed, writes it next to the testbed run under $PCNME_DATA_DIR/results/, and prints one Markdown table of the
16.3 metrics for both modes and their difference. Numbers come from the Parquet files only (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pcnme.config import load_config
from pcnme.experiments.city import City
from pcnme.metrics.stats import run_metrics

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
    p.add_argument("--method", default="greedy", choices=("greedy", "random"))
    p.add_argument("--seed", type=int, default=3000)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--cars", type=int, default=200)
    a = p.parse_args()
    cfg = load_config()
    results = Path(os.environ.get("PCNME_DATA_DIR", ".")).resolve() / "results"
    testbed = results / f"{a.scenario}-{a.method}-{a.seed}-testbed"
    if not (testbed / "tasks.parquet").exists():
        raise SystemExit(f"no testbed run at {testbed}: run python -m pcnme.testbed.runner first")
    city = City(cfg, a.scenario, a.method, a.seed, n_cars=a.cars, usage="evaluation").run(a.seconds)
    fast = city.write(results, {"method": a.method})
    m_fast = run_metrics(fast, cfg.fast_mode.early_decisions).metrics
    m_bed = run_metrics(testbed, cfg.fast_mode.early_decisions).metrics
    print(f"{a.scenario}, {a.method}, seed {a.seed}, {a.seconds:g} s, {a.cars} cars\n")
    print("| metric | fast mode | testbed | testbed - fast |")
    print("| --- | --- | --- | --- |")
    for key, name, fmt in ROWS:
        f, t = m_fast[key], m_bed[key]
        print(f"| {name} | {fmt.format(f)} | {fmt.format(t)} | {fmt.format(t - f)} |")


if __name__ == "__main__":
    main()
