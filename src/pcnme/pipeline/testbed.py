"""run_testbed.py: every method on the Docker testbed next to its fast-mode twin (final-design 15.4,
16.3; PLAN M9.4).

For each seed, scenario and method: one testbed run (pcnme.testbed.runner: the real containers and network,
`--containers` car containers of 20 cars, the scenario's per-zone load) and its fast-mode twin (the same
scenario, method, seed and fleet in the simulator, the trained methods deployed as in the zones). Each
finished run appends one row per mode to data/vfc/testbed/summary.csv (every 16.3 metric, wall time, commit,
config hash, setup), and gap.md / gap.csv give, per scenario and method, the means over the seeds both modes
ran and the difference.

Resumable: a (scenario, method, seed) finished in both modes is skipped; rows of another commit, config or
setup are refused (move the file away to start over). Seed-major: every scenario and method of the first
seed, then the next seed, so a stopped batch still covers everything. Never while train.py or evaluate.py
runs.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pcnme.pipeline.evaluate import METHODS, append_row

GAP_METRICS = ("on_time", "late", "dropped", "failed", "latency_mean_s", "latency_p99_s", "energy_per_task_j")
MODES = ("testbed", "fast")
MINUTES_PER_RUN = 3.2  # measured 7 Nov (40 cars, 60 s): containers up, run, drain, collect, the fast twin
Key = tuple[str, str, int, str]  # scenario, method, seed, mode


def plan_runs(
    scenarios: Sequence[str], methods: Sequence[str], seeds: Sequence[int], done: set[Key]
) -> list[tuple[str, str, int]]:
    """What is left, seed-major; a run missing in either mode runs again in both."""
    return [
        (s, m, seed)
        for seed in seeds
        for s in scenarios
        for m in methods
        if not all((s, m, seed, mode) in done for mode in MODES)
    ]


def read_done(path: Path, ident: Mapping[str, str]) -> set[Key]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    keys = ("commit", "config_hash", "setup")
    mine = tuple(ident.get(k, "") for k in keys)
    other = {tuple(r.get(k, "") for k in keys) for r in rows} - {mine}
    if other:
        raise RuntimeError(
            f"{path} has rows of another config or commit or setup {sorted(other)}; this batch is {mine}. "
            "Move the file away to start a new batch."
        )
    return {(r["scenario"], r["method"], int(r["seed"]), r["mode"]) for r in rows}


def gap_rows(rows: Iterable[Mapping[str, str]]) -> list[dict[str, Any]]:
    """Per scenario and method: the means over the seeds that ran in both modes, and testbed - fast."""
    by: dict[tuple[str, str], dict[int, dict[str, Mapping[str, str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in rows:
        by[(r["scenario"], r["method"])][int(r["seed"])][r["mode"]] = r
    out = []
    order = {m: i for i, m in enumerate(METHODS)}
    for (s, m), seeds in sorted(by.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 99))):
        both = [v for v in seeds.values() if all(mode in v for mode in MODES)]
        if not both:
            continue
        g: dict[str, Any] = {"scenario": s, "method": m, "seeds": len(both)}
        for k in GAP_METRICS:
            fast = statistics.mean(float(v["fast"][k]) for v in both)
            bed = statistics.mean(float(v["testbed"][k]) for v in both)
            g[f"{k}_fast"], g[f"{k}_testbed"], g[f"{k}_gap"] = fast, bed, bed - fast
        out.append(g)
    return out


def gap_table(rows: Sequence[Mapping[str, Any]]) -> str:
    from pcnme.metrics.compare import NAMES

    head = [
        "# Fast mode vs testbed (PLAN M9.4)",
        "",
        "Means over the seeds that ran in both modes; gap = testbed - fast mode. Percentages of tasks.",
        "",
        "| scenario | method | seeds | on time fast | on time testbed | gap (pp) | p99 fast ms "
        "| p99 testbed ms | J/task fast | J/task testbed | dropped testbed | failed testbed |",
        "| --- " * 12 + "|",
    ]
    lines = []
    for r in rows:
        cells = [
            r["scenario"],
            NAMES.get(r["method"], r["method"]),
            str(r["seeds"]),
            f"{100 * r['on_time_fast']:.1f}",
            f"{100 * r['on_time_testbed']:.1f}",
            f"{100 * r['on_time_gap']:.1f}",
            f"{1000 * r['latency_p99_s_fast']:.0f}",
            f"{1000 * r['latency_p99_s_testbed']:.0f}",
            f"{r['energy_per_task_j_fast']:.2f}",
            f"{r['energy_per_task_j_testbed']:.2f}",
            f"{100 * r['dropped_testbed']:.1f}",
            f"{100 * r['failed_testbed']:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(head + lines) + "\n"


def write_gap(folder: Path) -> None:
    summary = folder / "summary.csv"
    if not summary.exists():
        return
    with summary.open(encoding="utf-8", newline="") as f:
        rows = gap_rows(csv.DictReader(f))
    if not rows:
        return
    (folder / "gap.md").write_bytes(gap_table(rows).encode())
    with (folder / "gap.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fast_twin(scenario: str, method: str, seed: int, seconds: float, cars: int, results: Path) -> Path:
    """The same scenario, method, seed and fleet in fast mode; trained methods deployed as in the zones."""
    from pcnme.config import load_config
    from pcnme.experiments.city import City
    from pcnme.experiments.runner import deployed, method_config
    from pcnme.paths import MapPaths
    from pcnme.pipeline import checkpoint

    cfg = method_config(load_config(), method)
    factory: Any = method
    if method not in ("greedy", "random"):  # Q42, Q54
        trained = checkpoint.load(MapPaths("vfc").train(scenario, method) / "trained.pt")
        factory = deployed(cfg, trained, seed)
    city = City(cfg, scenario, factory, seed, n_cars=cars, usage="evaluation", method_name=method)
    return city.run(seconds).write(results, {"method": method})


def _row(
    run: Path, mode: str, s: str, m: str, seed: int, wall: float, ident: Mapping[str, str]
) -> dict[str, Any]:
    from pcnme.config import load_config
    from pcnme.metrics.stats import METRICS, run_metrics

    metrics = run_metrics(run, load_config().fast_mode.early_decisions).metrics
    return {"map": "vfc", "scenario": s, "method": m, "seed": seed, "mode": mode,
            **{k: metrics[k] for k in METRICS}, "wall_s": round(wall, 1), "commit": ident["commit"],
            "config_hash": ident["config_hash"], "setup": ident["setup"], "run_dir": run.name}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--maps", nargs="+")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--scenarios", nargs="+", default=None, help="default: S1-S6")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--seeds", type=int, default=5, help="evaluation seeds per scenario and method")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--containers", type=int, default=2, help="car containers of 20 cars (M9.3: 40 cars)")
    ap.add_argument("--yes", action="store_true", help="start without waiting for Enter")
    ap.add_argument("--smoke", action="store_true", help="print the plan only (no Docker)")
    args = ap.parse_args(argv)
    if (["vfc"] if args.all else args.maps) != ["vfc"]:
        print(
            "The testbed runs the VFC grid (the car containers walk it); LuST's testbed is planned (PLAN R3)."
        )
        return 1
    from pcnme.config import load_config
    from pcnme.maps import config_hash, map_source
    from pcnme.paths import MapPaths, data_dir
    from pcnme.pipeline import runinfo

    cfg = load_config()
    folder = MapPaths("vfc").testbed
    first = cfg.fast_mode.seed_split["evaluation"].start
    seeds = list(range(first, first + args.seeds))
    scenarios = args.scenarios or list(cfg.scenarios)
    cars = 20 * args.containers
    ident = {
        "commit": runinfo.commit(),
        "config_hash": config_hash(map_source("vfc", measured=False)),
        "setup": f"{cars} cars, {args.seconds:g} s",  # a batch never mixes fleets or run lengths
    }
    try:
        done = read_done(folder / "summary.csv", ident)
    except RuntimeError as e:
        print(e)
        return 2
    todo = plan_runs(scenarios, args.methods, seeds, done)
    print(f"PCNME run_testbed.py   commit {ident['commit']}\n")
    print(f"  {len(scenarios)} scenarios x {len(args.methods)} methods x {len(seeds)} seeds, {cars} cars, "
          f"{args.seconds:g} s: {len(todo)} runs to do ({len(done) // 2} done before); about "
          f"{len(todo) * MINUTES_PER_RUN / 60:.1f} h. Results: {folder}")  # fmt: skip
    if args.smoke or not todo:
        write_gap(folder)
        return 0
    if not args.yes:
        try:
            input("\nPress Enter to start (Ctrl+C to stop) ")
        except (EOFError, KeyboardInterrupt):
            print("\nnot started.")
            return 1
    from pcnme.testbed.runner import run

    os.environ.setdefault("PCNME_DATA_DIR", str(data_dir()))
    summary = folder / "summary.csv"
    t_start = time.monotonic()
    for i, (s, m, seed) in enumerate(todo, 1):
        t0 = time.monotonic()
        bed = run(cfg, s, seed, m, args.seconds, folder, args.containers)
        t1 = time.monotonic()
        twin = fast_twin(s, m, seed, args.seconds, cars, folder / "results")
        t2 = time.monotonic()
        append_row(summary, _row(bed, "testbed", s, m, seed, t1 - t0, ident))
        append_row(summary, _row(twin, "fast", s, m, seed, t2 - t1, ident))
        write_gap(folder)
        left = (time.monotonic() - t_start) / i * (len(todo) - i)
        print(f"  {i}/{len(todo)}  {s} {m} seed {seed}: {(t2 - t0) / 60:.1f} min; {left / 3600:.1f} h left")
    print(f"\ndone: {summary}; the gap table: {folder / 'gap.md'}")
    return 0
