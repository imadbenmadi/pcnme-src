"""The 16.3 metrics the evaluation did not compute, from the existing runs (src/pcnme/metrics/extra.py; PLAN
"Missing metrics"). Nothing is rerun.

    uv run python tools/extra_metrics.py --maps vfc lust

Writes data/<map>/results/extra.md and extra-*.csv:
- predictor: next-zone acc@1 and T_exit error per scenario (Greedy's runs: mobility does not depend on the
  method)
- zone change: delivery and on-time share of the tasks whose car changed zone, per scenario and method
- stable: decisions until the on-time rate is stable, per scenario and method (median over the seeds that get
  there)
- hypervolume: per scenario, the NSGA-II fronts' hypervolume and its ratio to the greedy plan's
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


def _columns(run: Path, table: str, cols: list[str]) -> dict[str, list[Any]]:
    import pyarrow.parquet as pq

    out: dict[str, list[Any]] = pq.read_table(run / f"{table}.parquet", columns=cols).to_pydict()
    return out


def one_run(args: tuple[str, str, bool]) -> dict[str, Any]:
    """Every per-run metric of one fast-mode run directory."""
    from pcnme.maps import map_config
    from pcnme.metrics.extra import decisions_until_stable, predictor_errors, zone_change_delivery

    run_dir, map_name, predict = args
    run = Path(run_dir)
    scenario, method, seed = (
        run.name.split("-")[0],
        "-".join(run.name.split("-")[1:-2]),
        int(run.name.split("-")[-2]),
    )
    pos = _columns(
        run, "positions", ["time_s", "car_id", "x_m", "y_m", "speed_mps", "heading_x", "heading_y", "zone"]
    )
    tasks = _columns(run, "tasks", ["task_id", "time_s", "outcome", "latency_s", "deadline_s"])
    dec = _columns(run, "decisions", ["time_s"])["time_s"]
    row: dict[str, Any] = {"scenario": scenario, "method": method, "seed": seed}
    zc = zone_change_delivery(tasks, pos)
    row.update({"zc_tasks": zc["tasks"], "zc_delivered": zc["delivered"], "zc_on_time": zc["on_time"]})
    row["decisions_until_stable"] = decisions_until_stable(tasks, dec)
    if predict:
        lay = map_config(map_name).layout
        centres = {z: tuple(lay.zone_positions_m[z]) for z in lay.active_zones}
        row.update({f"pred_{k}": v for k, v in predictor_errors(pos, centres, lay.zone_radius_m).items()})
    return row


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def _fmt(x: float | None, pct: bool = False, digits: int = 2) -> str:
    if x is None:
        return "–"
    return f"{100 * x:.1f}" if pct else f"{x:.{digits}f}"


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "| --- " * len(head) + "|"] + [
        "| " + " | ".join(r) + " |" for r in rows
    ]


def _stable_cell(rows: list[dict[str, Any]]) -> str:
    vals = [r["decisions_until_stable"] for r in rows if r["decisions_until_stable"] is not None]
    return f"{int(statistics.median(vals)):,} ({len(vals)}/{len(rows)})" if vals else f"– (0/{len(rows)})"


def report(map_name: str, workers: int) -> Path:
    from pcnme.metrics.compare import NAMES
    from pcnme.metrics.extra import STABLE_LEVEL, STABLE_WINDOW, front_hypervolumes
    from pcnme.paths import MapPaths

    paths = MapPaths(map_name)
    runs = sorted(
        d for d in paths.runs.iterdir() if d.name.endswith("-fast") and (d / "positions.parquet").exists()
    )
    jobs = [(str(d), map_name, "-greedy-" in d.name) for d in runs]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(one_run, jobs, chunksize=4))
    out = paths.results
    with (out / "extra-runs.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["scenario"], r["method"])].append(r)
    scs = sorted({r["scenario"] for r in rows})
    order = ["pcnme", "greedy", "d3qn-greedy", "d3qn-scratch", "vfc-ppo", "tang-wong", "qeco", "random"]
    methods = [m for m in order if any(r["method"] == m for r in rows)]
    name = {m: NAMES.get(m, m) for m in methods}

    def col(s: str, m: str, k: str) -> list[Any]:
        return [r[k] for r in by.get((s, m), [])]

    lines = [
        f"# Extra 16.3 metrics: {map_name}",
        "",
        "From the existing evaluation runs (tools/extra_metrics.py, metrics/extra.py); means over the seeds.",
        "",
        "## Mobility predictor (10.2), level 2: motion math",
        "",
        "Greedy's runs (the cars move the same under every method). One prediction per moving, served car "
        "per second. The mean error is dominated by slow or stopped cars (a long predicted exit); the median "
        "is the typical error.",
        "",
    ]
    pred = [
        [
            s,
            f"{sum(col(s, 'greedy', 'pred_predictions')):,}",
            _fmt(_mean(col(s, "greedy", "pred_acc1")), True),
            _fmt(_mean(col(s, "greedy", "pred_t_exit_mae_s"))),
            _fmt(_mean(col(s, "greedy", "pred_t_exit_median_s"))),
        ]
        for s in scs
    ]
    lines += _table(
        ["scenario", "predictions", "next-zone acc@1 %", "T_exit error, mean s", "T_exit error, median s"],
        pred,
    )
    lines += [
        "",
        "## Delivery after a zone change (on time %)",
        "",
        "Tasks whose car's serving zone at creation differs from the one at close (the last row: how many "
        "such tasks a run has, Greedy's runs; few, so these shares are noisy).",
        "",
    ]
    zc = [[name[m]] + [_fmt(_mean(col(s, m, "zc_on_time")), True) for s in scs] for m in methods]
    zc.append(["tasks per run"] + [f"{_mean(col(s, 'greedy', 'zc_tasks')) or 0:.0f}" for s in scs])
    lines += _table(["method", *scs], zc)
    lines += [
        "",
        f"## Decisions until the on-time rate is stable (≥ {100 * STABLE_LEVEL:.0f} % over the last "
        f"{STABLE_WINDOW} tasks, to the end)",
        "",
        "Median over the seeds that get there; in brackets how many of them (– : none).",
        "",
    ]
    lines += _table(
        ["method", *scs], [[name[m]] + [_stable_cell(by.get((s, m), [])) for s in scs] for m in methods]
    )
    windows = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(paths.nsga.glob("S*-w*.json"))]
    hv = front_hypervolumes(windows)
    with (out / "extra-hypervolume.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hv[0]))
        w.writeheader()
        w.writerows(hv)
    lines += [
        "",
        "## NSGA-II fronts (13.2): hypervolume",
        "",
        "Per scenario, the reference point is 1.1 x the worst f1 and f2 of its fronts and greedy plans; the "
        "ratio is the front's hypervolume over the greedy plan's (> 1: the front dominates more).",
        "",
    ]
    hvr = []
    for s in sorted({r["scenario"] for r in hv}):
        ws = [r for r in hv if r["scenario"] == s]
        hvr.append(
            [
                s,
                str(len(ws)),
                f"{_mean([r['points'] for r in ws]) or 0:.0f}",
                _fmt(_mean([r["hv"] for r in ws]), digits=4),
                _fmt(_mean([r["hv_ratio"] for r in ws]), digits=3),
            ]
        )
    lines += _table(
        ["scenario", "windows", "front points (mean)", "hypervolume (mean)", "ratio to greedy (mean)"], hvr
    )
    md = out / "extra.md"
    md.write_bytes(("\n".join(lines) + "\n").encode())
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--maps", nargs="+", default=["vfc", "lust"])
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    for m in a.maps:
        print(f"{m}: {report(m, a.workers)}")
    return 0


if __name__ == "__main__":  # the guard keeps spawned workers from rerunning main
    raise SystemExit(main())
