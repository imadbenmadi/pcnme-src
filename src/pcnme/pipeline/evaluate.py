"""evaluate.py: every scenario x method x evaluation seed on a trained map (final-design 16.1, 16.3).

Learning methods are deployed from data/<map>/train/<scenario>-<method>/trained.pt (they keep learning
online, Q42); Greedy and Random run as they are. Each finished run appends one row to
data/<map>/results/summary.csv (scenario, method, seed, every 16.3 metric, wall time, commit, config hash);
its Parquet logs go to results/runs/<run_id>/. At the end: report.md / report.csv (mean +- 95 % interval over
the seeds), one figure per scenario, and the training curves. Resumable: a (scenario, method, seed) already
in summary.csv is skipped; rows of another config or commit are refused, never mixed.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from pcnme.paths import MapPaths

METHODS = ("pcnme", "d3qn-scratch", "d3qn-greedy", "vfc-ppo", "tang-wong", "qeco", "greedy", "random")
FIELDS_HEAD = ["map", "scenario", "method", "seed"]
FIELDS_TAIL = ["wall_s", "commit", "config_hash", "run_dir"]


def _eval_job(
    map_name: str,
    scenario: str,
    method: str,
    seeds: tuple[int, ...],
    seconds: float | None,
    n_cars: int | None,
    ident: dict[str, str],
    report: Any,
) -> int:
    from pcnme.config import config_from_mapping
    from pcnme.experiments.city import City
    from pcnme.experiments.runner import RunSpec, method_config, placement, report_constants, training_info
    from pcnme.maps import map_source
    from pcnme.metrics.stats import METRICS, run_metrics
    from pcnme.pipeline import checkpoint
    from pcnme.pipeline.train import with_w_star

    paths = MapPaths(map_name)
    w = json.loads(paths.w_star.read_text(encoding="utf-8"))["w_star"]
    source = with_w_star(map_source(map_name), w)
    cfg = method_config(config_from_mapping(source), method)
    trained = None
    if method not in ("greedy", "random"):
        f = paths.train(scenario, method) / "trained.pt"
        if not f.exists():
            raise RuntimeError(f"{method} on {scenario} is not trained ({f}): run train.py first")
        trained = checkpoint.load(f)
    spec = RunSpec(scenario, method, seeds, seconds=seconds, n_cars=n_cars)
    length = cfg.fast_mode.episode_length if seconds is None else seconds
    summary = paths.results / "summary.csv"
    report(status="running", phase="episodes", done=0, total=len(seeds), unit="seeds")
    for i, seed in enumerate(seeds):
        t0 = time.monotonic()
        extra = {
            "method": method,
            "usage": "evaluation",
            "reward_w": cfg.reward.w,
            "map": map_name,
            "training": training_info(trained),
            **report_constants(cfg, scenario),
        }
        factory, _ = placement(source, cfg, spec, trained, seed)
        city = City(cfg, scenario, factory, seed, n_cars, usage="evaluation", method_name=method).run(length)
        extra["broker_messages"] = city.bus.published
        d = city.write(paths.runs, extra)
        m = run_metrics(d, cfg.fast_mode.early_decisions).metrics
        row = {
            "map": map_name,
            "scenario": scenario,
            "method": method,
            "seed": seed,
            **{k: m[k] for k in METRICS},
            "wall_s": round(time.monotonic() - t0, 1),
            "commit": ident["commit"],
            "config_hash": ident["config_hash"],
            "run_dir": d.name,
        }
        append_row(summary, row)
        report(
            done=i + 1,
            numbers={
                "seed": seed,
                "on time": round(m["on_time"], 4),
                "mean ê": round(m["mean_e_hat"], 3),
                "p99 ms": round(1e3 * m["latency_p99_s"], 1),
            },
        )
    report(status="done")
    return len(seeds)


def append_row(path: Path, row: dict[str, Any]) -> None:
    """One finished run. The lock file keeps parallel jobs from interleaving their lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    for _ in range(600):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            time.sleep(0.05)
    else:
        raise TimeoutError(f"{lock} stays locked: delete it if no evaluate.py runs")
    try:
        new = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
    finally:
        os.close(fd)
        os.unlink(lock)


def done_rows(path: Path, ident: dict[str, str]) -> set[tuple[str, str, int]]:
    """(scenario, method, seed) already finished; refuses rows of another config or commit."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    other = {(r["commit"], r["config_hash"]) for r in rows} - {(ident["commit"], ident["config_hash"])}
    if other:
        raise RuntimeError(
            f"{path} has rows of another config or commit {sorted(other)}; this run is "
            f"{(ident['commit'], ident['config_hash'])}. Refusing to mix them: move the file away (or use "
            "--fresh) to start a new summary."
        )
    return {(r["scenario"], r["method"], int(r["seed"])) for r in rows}


def write_outputs(screen: Any, paths: MapPaths, early: int) -> None:
    from pcnme.metrics.report import report

    if paths.runs.exists():
        files = report([paths.runs], paths.results, early)
        screen.say(f"tables and figures: {', '.join(p.name for p in files[:4])} ... in {paths.results}")
    _training_curves(paths)


def _training_curves(paths: MapPaths) -> None:
    """One figure per scenario: on-time share and mean ê per training episode, every learning method."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by: dict[str, dict[str, list[dict[str, str]]]] = {}
    for f in sorted((paths.root / "train").glob("*/episodes.csv")):
        scenario, method = f.parent.name.split("-", 1)
        with f.open(encoding="utf-8", newline="") as fh:
            by.setdefault(scenario, {})[method] = list(csv.DictReader(fh))
    for scenario, methods in by.items():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
        for method, rows in sorted(methods.items()):
            x = [int(r["episode"]) for r in rows]
            a1.plot(x, [100 * float(r["on_time"]) for r in rows], label=method)
            a2.plot(x, [float(r["mean_e_hat"]) for r in rows], label=method)
        a1.set(xlabel="training episode", ylabel="on time %", title=f"{scenario}: on-time share in training")
        a2.set(xlabel="training episode", ylabel="mean ê", title=f"{scenario}: mean ê in training")
        a1.legend(fontsize=7)
        fig.tight_layout()
        paths.results.mkdir(parents=True, exist_ok=True)
        fig.savefig(paths.results / f"training-{scenario}.png", dpi=130)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    from pcnme.pipeline.train import archive, parse, smoke_env

    args = parse(argv, evaluate=True)
    if args.smoke:
        smoke_env()
    from pcnme.config import DEFAULT_CONFIG_PATH, load_yaml
    from pcnme.maps import config_hash, map_config, map_source
    from pcnme.pipeline import runinfo
    from pcnme.pipeline.screen import Screen, run_jobs

    base = load_yaml(DEFAULT_CONFIG_PATH)
    names = list(base["maps"]) if args.all else args.maps
    todo: list[tuple[str, list[tuple[str, tuple[Any, ...]]], dict[str, str]]] = []
    print(f"PCNME evaluate.py   commit {runinfo.commit()}\n")
    for name in names:
        paths = MapPaths(name)
        if name not in base["maps"] or not paths.w_star.exists():
            print(f"  {name}: not trained yet (no {paths.w_star}): skipped")
            continue
        if args.fresh and (dest := archive_results(paths)):
            print(f"  {name}: earlier results moved to {dest}")
        cfg = map_config(name)
        ident = {"commit": runinfo.commit(), "config_hash": config_hash(map_source(name, measured=False))}
        try:
            finished = done_rows(paths.results / "summary.csv", ident)
        except RuntimeError as e:
            print(f"  {name}: {e}")
            return 2
        n_seeds = 2 if args.smoke else cfg.fast_mode.eval_seeds
        first = cfg.fast_mode.seed_split["evaluation"].start
        seconds, n_cars = (3.0, 40) if args.smoke else (None, None)
        scenarios = args.scenarios or (["S1"] if args.smoke else list(cfg.scenarios))
        jobs = []
        for s in scenarios:
            for m in METHODS:
                seeds = tuple(k for k in range(first, first + n_seeds) if (s, m, k) not in finished)
                if seeds:
                    jobs.append((f"{s} {m}", (name, s, m, seeds, seconds, n_cars, ident)))
        runs = sum(len(a[3]) for _, a in jobs)
        print(
            f"  {name}: {len(scenarios)} scenarios x {len(METHODS)} methods x {n_seeds} seeds; "
            f"{runs} runs to do ({len(finished)} done before); "
            f"about {runs * 1.5 / max(args.workers, 1) / 60:.1f} h "
            f"on {args.workers} workers"
        )
        todo.append((name, jobs, ident))
    if not todo:
        return 1
    if not args.yes:
        try:
            input("\nPress Enter to start (Ctrl+C to stop) ")
        except (EOFError, KeyboardInterrupt):
            print("\nnot started.")
            return 1
    for name, jobs, _ in todo:
        paths = MapPaths(name)
        with Screen(f"evaluate.py  map {name}{'  (smoke)' if args.smoke else ''}") as screen:
            screen.stage("evaluation: scenario x method x seed (16.3)", [n for n, _ in jobs])
            res = run_jobs(screen, _eval_job, jobs, args.workers)
            failed = {k: v for k, v in res.items() if isinstance(v, Exception)}
            write_outputs(screen, paths, map_config(name).fast_mode.early_decisions)
        if failed:
            print(f"{name}: {len(failed)} jobs failed: {sorted(failed)}")
            return 3
        print(
            f"\n{name}: done. summary: {paths.results / 'summary.csv'}; tables and figures in {paths.results}"
        )
    _ = archive
    return 0


def archive_results(paths: MapPaths) -> Path | None:
    if not paths.results.exists():
        return None
    dest = paths.root / "_archive" / f"results-{time.strftime('%Y%m%d-%H%M%S')}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    paths.results.rename(dest)
    return dest
