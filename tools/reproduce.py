"""PLAN Q3 / Q7: are the reported results reproducible? Re-runs chosen evaluation runs with today's code,
exactly as evaluate.py runs them (pipeline/evaluate.py _eval_job), into a scratch folder, and compares every
16.3 metric with the row summary.csv holds for that run. Nothing in data/<map>/results is touched.

    uv run python tools/reproduce.py --maps vfc lust --methods pcnme greedy --seed 3000

Prints one line per run and writes data/<map>/results/reproduce.md. A run is "identical" when every metric is
equal to the stored one (fast mode is deterministic: same config and seed, bit-identical logs).
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


def rerun(args: tuple[str, str, str, int, str]) -> dict[str, Any]:
    from pcnme.config import config_from_mapping
    from pcnme.experiments.city import City
    from pcnme.experiments.runner import RunSpec, method_config, placement
    from pcnme.maps import map_source
    from pcnme.metrics.stats import METRICS, run_metrics
    from pcnme.paths import MapPaths
    from pcnme.pipeline import checkpoint
    from pcnme.pipeline.train import with_w_star

    map_name, scenario, method, seed, out = args
    models = MapPaths(map_name)
    w = json.loads(models.w_star.read_text(encoding="utf-8"))["w_star"]
    source = with_w_star(map_source(map_name), w)
    cfg = method_config(config_from_mapping(source), method)
    trained = None
    if method not in ("greedy", "random"):
        trained = checkpoint.load(models.train(scenario, method) / "trained.pt")
    spec = RunSpec(scenario, method, (seed,))
    factory, _ = placement(source, cfg, spec, trained, seed)
    city = City(cfg, scenario, factory, seed, None, usage="evaluation", method_name=method)
    city = city.run(cfg.fast_mode.episode_length)
    d = city.write(Path(out), {"method": method, "usage": "evaluation", "map": map_name})
    m = run_metrics(d, cfg.fast_mode.early_decisions).metrics
    return {
        "map": map_name,
        "scenario": scenario,
        "method": method,
        "seed": seed,
        **{k: m[k] for k in METRICS},
    }


def stored(map_name: str) -> dict[tuple[str, str, int], dict[str, str]]:
    from pcnme.paths import MapPaths

    with (MapPaths(map_name).results / "summary.csv").open(encoding="utf-8", newline="") as f:
        return {(r["scenario"], r["method"], int(r["seed"])): r for r in csv.DictReader(f)}


def compare(new: dict[str, Any], old: dict[str, str]) -> list[str]:
    """The metrics that differ (as stored text vs today's value)."""
    from pcnme.metrics.stats import METRICS

    out = []
    for k in METRICS:
        a, b = new[k], old.get(k, "")
        if a is None and b in ("", "None"):
            continue
        try:
            if float(b) != float(a):
                out.append(f"{k}: {b} -> {a}")
        except (TypeError, ValueError):
            if str(a) != b:
                out.append(f"{k}: {b} -> {a}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--maps", nargs="+", default=["vfc", "lust"])
    ap.add_argument("--methods", nargs="+", default=["pcnme", "greedy"])
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from pcnme.maps import map_config
    from pcnme.paths import MapPaths

    with tempfile.TemporaryDirectory(prefix="pcnme-reproduce-") as tmp:
        for map_name in a.maps:
            scenarios = list(map_config(map_name).scenarios)
            jobs = [(map_name, s, m, a.seed, tmp) for s in scenarios for m in a.methods]
            with ProcessPoolExecutor(a.workers) as pool:
                rows = list(pool.map(rerun, jobs))
            old = stored(map_name)
            lines = [f"# Reproduction check: {map_name} (PLAN Q3)", "",
                     f"Seed {a.seed}, rerun with today's code and compared metric by metric with "
                     "summary.csv.", "",
                     "| scenario | method | result |", "| --- | --- | --- |"]  # fmt: skip
            same = 0
            for r in rows:
                key = (r["scenario"], r["method"], r["seed"])
                diff = compare(r, old[key]) if key in old else ["no stored row"]
                same += not diff
                verdict = "identical" if not diff else "; ".join(diff[:4]) + (" ..." if len(diff) > 4 else "")
                lines.append(f"| {r['scenario']} | {r['method']} | {verdict} |")
                print(f"{map_name} {r['scenario']} {r['method']}: {verdict}")
            lines += ["", f"{same} of {len(rows)} runs identical."]
            (MapPaths(map_name).results / "reproduce.md").write_bytes(("\n".join(lines) + "\n").encode())
            print(f"{map_name}: {same}/{len(rows)} identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
