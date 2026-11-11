"""run_testbed.py: every method on the Docker testbed next to its fast-mode twin (final-design 15.4,
16.3; PLAN M9.4).

For each seed, scenario and method: one testbed run (pcnme.testbed.runner: the real containers and network,
`--containers` car containers of 20 cars, the scenario's per-zone load) and its fast-mode twin (the same
scenario, method, seed and fleet in the simulator, the trained methods deployed as in the zones). Each
finished run appends one row per mode to data/vfc/testbed/summary.csv (every 16.3 metric, wall time, commit,
config hash, setup), and gap.md / gap.csv give, per scenario and method, the means over the seeds both modes
ran and the difference.

Resumable: a (scenario, method, seed) finished in both modes is skipped; rows of another config or setup are
refused (move the file away to start over). Each row keeps the commit it ran on: a fix to the testbed does not
void the cells it did not touch; the cells it did are archived first (`--archive-frozen`, `--redo`).
Seed-major: every scenario and method of the first seed, then the next seed, so a stopped batch still covers
everything.
Never while train.py or evaluate.py runs.

Before a batch, a pre-flight run per method (S2, testbed.preflight_s) must pass `health()`: every zone wrote
its rows (none crashed), every learning zone made online updates through a live trainer (Q62), and at most
testbed.preflight_max_no_ack of the tasks ended without an ack. During the batch a crashed zone or a learner
without updates stops it.
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
    keys = ("config_hash", "setup")  # the commit is kept per row (a testbed fix voids only what it touched)
    mine = tuple(ident.get(k, "") for k in keys)
    other = {tuple(r.get(k, "") for k in keys) for r in rows} - {mine}
    if other:
        raise RuntimeError(
            f"{path} has rows of another config or setup {sorted(other)}; this batch is {mine}. "
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


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def archive(folder: Path, pick: Any, label: str, slug: str) -> int:
    """Move the summary rows `pick(row)` selects, and their run folders, to archive/<slug>/ with a `label`
    column (kept for the record, out of the gap table). Returns the rows moved."""
    import shutil

    rows = _read(folder / "summary.csv")
    moved = [r for r in rows if pick(r)]
    if not moved:
        return 0
    dest = folder / "archive" / slug
    _write(dest / "summary.csv", _read(dest / "summary.csv") + [{**r, "label": label} for r in moved])
    for r in moved:
        src = folder / "results" / r.get("run_dir", "")
        if r.get("run_dir") and src.exists():
            (dest / "results").mkdir(parents=True, exist_ok=True)
            target = dest / "results" / src.name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(src), str(target))
    _write(folder / "summary.csv", [r for r in rows if not pick(r)])
    write_gap(folder)
    return len(moved)


def health(run_dir: Path, method: str, max_no_ack: float, zones: int = 4) -> list[str]:
    """What is wrong with a testbed run, in words (empty: healthy). A zone that wrote no rows crashed; a
    learning method's zone needs online updates and a live trainer (Q62); at most `max_no_ack` of the tasks
    may end without an ack."""
    import json

    from pcnme.methods.learners import LEARNING

    problems = []
    wrote = {f.stem for f in (run_dir / "zones").glob("*.json")}
    counts = {f.stem.removeprefix("counts-"): json.loads(f.read_text(encoding="utf-8"))
              for f in run_dir.glob("counts-*.json")}  # fmt: skip
    if len(wrote) < zones:
        problems.append(f"{zones - len(wrote)} zone(s) wrote no rows (crashed): only {sorted(wrote)}")
    log = run_dir / "containers.log"
    if log.exists():
        raised = [
            e for e in raised_in(log.read_text(encoding="utf-8", errors="replace")) if e not in HARMLESS
        ]
        if raised:
            problems.append(f"a container raised: {raised[-1]}")
    if method in LEARNING:
        for z, c in sorted(counts.items()):
            if c.get("agent updates", 0) <= 0:
                problems.append(f"{z}: no online updates (agent not connected to its trainer?)")
            elif c.get("trainer alive", 0) != 1:
                problems.append(f"{z}: its trainer process died")
    tasks = _task_count(run_dir)
    no_ack = sum(c.get("no ack by the end (drop)", 0) for c in counts.values())
    if tasks and no_ack / tasks > max_no_ack:
        problems.append(f"{no_ack / tasks:.1%} of the tasks ended without an ack (limit {max_no_ack:.0%})")
    return problems


# exceptions that are not a fault of the run: a client that hung up mid-request (a car's request timed out
# under load; the server logs it, nothing in the zone is touched)
HARMLESS = frozenset({"starlette.requests.ClientDisconnect"})


def raised_in(log: str) -> list[str]:
    """The exception of every traceback in a compose log ("<container> | ..." lines): its last line's type."""
    out = []
    rows = [tuple(x.split("|", 1)) if "|" in x else ("", x) for x in log.splitlines()]
    for i, (who, x) in enumerate(rows):
        if not x.strip().startswith("Traceback (most recent call last)"):
            continue
        indent = len(x) - len(x.lstrip())
        for other, y in rows[i + 1 :]:  # the same container's exception line, at the Traceback line's indent
            if other == who and y.strip() and len(y) - len(y.lstrip()) == indent:
                out.append(y.strip().split(":", 1)[0])  # "pkg.Type: message" -> "pkg.Type"
                break
    return out


def _task_count(run_dir: Path) -> int:
    f = run_dir / "tasks.parquet"
    if not f.exists():
        return 0
    import pyarrow.parquet as pq

    return int(pq.read_metadata(f).num_rows)


def preflight(cfg: Any, methods: Sequence[str], folder: Path, containers: int) -> dict[str, list[str]]:
    """One short S2 run per method (testbed.preflight_s, the first evaluation seed); its problems."""
    from pcnme.testbed.runner import run

    seed = cfg.fast_mode.seed_split["evaluation"].start
    out = {}
    for m in methods:
        d = run(cfg, "S2", seed, m, cfg.testbed.preflight_s, folder / "preflight", containers,
                models=folder.parent / "train")  # fmt: skip
        out[m] = health(d, m, cfg.testbed.preflight_max_no_ack)
        print(f"  pre-flight {m}: {'ok' if not out[m] else '; '.join(out[m])}")
    return out


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
    ap.add_argument("--archive-frozen", action="store_true",
                    help="move every learning method's rows to archive/frozen-bug/ (the testbed agents ran "
                         "frozen before 11 Nov) and stop")  # fmt: skip
    ap.add_argument("--redo", action="store_true",
                    help="run the chosen cells again; their old rows go to archive/superseded/")  # fmt: skip
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
    if args.archive_frozen:
        from pcnme.methods.learners import LEARNING

        n = archive(folder, lambda r: r["method"] in LEARNING, "frozen (bug)", "frozen-bug")
        print(
            f"{n} rows of the learning methods moved to {folder / 'archive' / 'frozen-bug'} (frozen (bug))."
        )
        return 0
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
    if args.redo and not args.smoke:
        chosen = {(s, m, seed) for s in scenarios for m in args.methods for seed in seeds}
        n = archive(folder, lambda r: (r["scenario"], r["method"], int(r["seed"])) in chosen, "superseded",
                    "superseded")  # fmt: skip
        print(f"--redo: {n} earlier rows of these cells moved to {folder / 'archive' / 'superseded'}")
        done = read_done(folder / "summary.csv", ident)
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
    methods = list(dict.fromkeys(m for _, m, _ in todo))
    print(f"\nPre-flight: {len(methods)} short check runs ({cfg.testbed.preflight_s:g} s each)")
    failed = {m: p for m, p in preflight(cfg, methods, folder, args.containers).items() if p}
    if failed:
        print("\nNot started: the pre-flight check failed (data/vfc/testbed/preflight/ has the runs).")
        return 3
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
        problems = health(bed, m, 1.0)  # under load many tasks may end without an ack: not a stop
        if problems:
            print(f"  STOPPED after {s} {m} seed {seed}: {'; '.join(problems)}")
            return 4
        left = (time.monotonic() - t_start) / i * (len(todo) - i)
        print(f"  {i}/{len(todo)}  {s} {m} seed {seed}: {(t2 - t0) / 60:.1f} min; {left / 3600:.1f} h left")
    print(f"\ndone: {summary}; the gap table: {folder / 'gap.md'}")
    return 0
