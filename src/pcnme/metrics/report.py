"""`pcnme report`: the results tables and plots from run directories (final-design 16.3; PLAN M7).

Reads only the Parquet logs and run.json of each run (metrics/stats.py) and writes to the output directory:

    report.csv          one row per scenario, method and metric: mean, the 95 % interval, n and the seeds
    report.md           per scenario, the main metrics as mean ± half-width, and the knee-window table
    windows.csv         every run on a knee-reference window: (f1, f2), tasks found, distance to the front
    <scenario>.png      on-time share, mean latency and energy per task by method, with 95 % intervals
    <scenario>-load.png each method's mean busy share per zone (16.3: the per-zone load chart)
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from pcnme.metrics.stats import (
    METRICS,
    Interval,
    Run,
    WindowPoint,
    read_table,
    run_metrics,
    summarise,
    window_points,
    zone_loads,
)

TABLE = (  # the report.md columns: metric, header, scale, digits
    ("on_time", "on time %", 100.0, 1),
    ("late", "late %", 100.0, 1),
    ("dropped", "dropped %", 100.0, 1),
    ("failed", "failed %", 100.0, 2),
    ("latency_mean_s", "latency ms", 1000.0, 1),
    ("latency_p99_s", "p99 ms", 1000.0, 1),
    ("latency_worst1_s", "worst 1 % ms", 1000.0, 1),
    ("energy_per_task_j", "J / task", 1.0, 3),
    ("refusal_rate", "refused %", 100.0, 2),
    ("wasted_share", "wasted %", 100.0, 2),
    ("jain_load", "Jain", 1.0, 3),
    ("early_on_time", "early on time %", 100.0, 1),
)
PLOTS = (("on_time", "on-time share", 1.0), ("latency_mean_s", "mean latency (ms)", 1000.0),
         ("energy_per_task_j", "energy per task (J)", 1.0))  # fmt: skip
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")  # categorical order, zones z1-z5
INK, MUTED = "#0b0b0b", "#52514e"


def run_dirs(paths: Sequence[Path]) -> list[Path]:
    """The run directories in `paths`: each path is a run (it holds run.json) or a directory of runs; the
    training directories (<scenario>-<method>-train) hold no run.json and are skipped."""
    out: list[Path] = []
    for p in paths:
        if (p / "run.json").exists():
            out.append(p)
        else:
            out += sorted(d for d in p.iterdir() if (d / "run.json").exists())
    if not out:
        raise ValueError(f"no run directories (with run.json) in {', '.join(map(str, paths))}")
    return sorted(set(out))


def fmt(ci: Interval, scale: float, digits: int) -> str:
    if ci.n == 0 or math.isnan(ci.mean):
        return "–"
    m = f"{ci.mean * scale:.{digits}f}"
    return m if math.isnan(ci.half) else f"{m} ± {ci.half * scale:.{digits}f}"


def write_csv(path: Path, summary: dict[tuple[str, str], dict[str, Interval]], runs: Sequence[Run]) -> None:
    seeds: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in runs:
        seeds[(r.scenario, r.method)].append(r.seed)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "method", "metric", "mean", "ci_low", "ci_high", "half", "n", "seeds"])
        for (sc, me), metrics in summary.items():
            for k in METRICS:
                ci = metrics[k]
                s = " ".join(map(str, sorted(seeds[(sc, me)])))
                w.writerow([sc, me, k, ci.mean, ci.low, ci.high, ci.half, ci.n, s])


def write_markdown(
    path: Path, summary: dict[tuple[str, str], dict[str, Interval]], windows: Sequence[WindowPoint]
) -> None:
    lines = ["# Results", "", "Mean ± 95 % Student t interval over the seeds (final-design 16.3).", ""]
    for sc in sorted({s for s, _ in summary}):
        lines += [f"## {sc}", "", "| method | n | " + " | ".join(h for _, h, _, _ in TABLE) + " |"]
        lines.append("| --- " * (len(TABLE) + 2) + "|")
        for (s, me), m in summary.items():
            if s == sc:
                n = m["on_time"].n
                cells = [fmt(m[k], scale, dg) for k, _, scale, dg in TABLE]
                lines.append(f"| {me} | {n} | " + " | ".join(cells) + " |")
        lines.append("")
    if windows:
        lines += [
            "## Knee-reference windows",
            "",
            "| scenario | seed | method | f1 | f2 | tasks | distance |",
        ]
        lines.append("| --- " * 7 + "|")
        for p in windows:
            lines.append(
                f"| {p.scenario} | {p.seed} | {p.method} | {p.f1:.4f} | {p.f2:.4f} | {p.tasks} "
                f"| {p.distance:.4f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _axes(ax: object) -> None:
    ax.spines[["top", "right"]].set_visible(False)  # type: ignore[attr-defined]
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)  # type: ignore[attr-defined]
    ax.set_axisbelow(True)  # type: ignore[attr-defined]
    ax.tick_params(colors=MUTED, labelsize=8)  # type: ignore[attr-defined]


def plot_scenario(path: Path, scenario: str, summary: dict[tuple[str, str], dict[str, Interval]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [me for s, me in summary if s == scenario]
    fig, axes = plt.subplots(1, len(PLOTS), figsize=(4 * len(PLOTS), 3.4), constrained_layout=True)
    for ax, (k, title, scale) in zip(axes, PLOTS, strict=True):
        cis = [summary[(scenario, me)][k] for me in methods]
        means = [ci.mean * scale for ci in cis]
        halves = [0.0 if math.isnan(ci.half) else ci.half * scale for ci in cis]
        ax.bar(methods, means, color=SERIES[0], width=0.6, yerr=halves, ecolor=MUTED, capsize=3)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.tick_params(axis="x", rotation=45)
        _axes(ax)
    fig.suptitle(f"{scenario}: mean and 95 % interval over seeds", color=INK, fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loads(path: Path, scenario: str, loads: dict[str, list[dict[str, float]]]) -> None:
    """Each method's mean busy share per zone, averaged over its seeds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    methods = sorted(loads)
    zones = sorted({z for per_seed in loads.values() for d in per_seed for z in d})
    width = 0.8 / max(len(zones), 1)
    fig, ax = plt.subplots(figsize=(1.2 * len(methods) + 2, 3.4), constrained_layout=True)
    x = np.arange(len(methods))
    for j, z in enumerate(zones):
        y = [float(np.mean([d.get(z, 0.0) for d in loads[me]])) for me in methods]
        ax.bar(x + (j - (len(zones) - 1) / 2) * width, y, width * 0.9, color=SERIES[j % len(SERIES)], label=z)
    ax.set_xticks(x, methods, rotation=45)
    ax.set_ylabel("mean busy share", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncols=len(zones), loc="lower left", bbox_to_anchor=(0.0, 1.0))
    fig.suptitle(f"{scenario}: mean busy share per zone during the episode", color=INK, fontsize=10)
    _axes(ax)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(paths: Sequence[Path], out: Path, early_decisions: int, plots: bool = True) -> list[Path]:
    """Every output file of the report; the numbers come from the Parquet logs only."""
    dirs = run_dirs(paths)
    runs = [run_metrics(d, early_decisions) for d in dirs]
    summary = summarise(runs)
    windows = window_points(dirs)
    out.mkdir(parents=True, exist_ok=True)
    files = [out / "report.csv", out / "report.md"]
    write_csv(files[0], summary, runs)
    write_markdown(files[1], summary, windows)
    if windows:
        files.append(out / "windows.csv")
        with files[-1].open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scenario", "seed", "method", "f1", "f2", "tasks", "distance"])
            for p in windows:
                w.writerow([p.scenario, p.seed, p.method, p.f1, p.f2, p.tasks, p.distance])
    if plots:
        for sc in sorted({s for s, _ in summary}):
            files.append(out / f"{sc}.png")
            plot_scenario(files[-1], sc, summary)
            loads: dict[str, list[dict[str, float]]] = defaultdict(list)
            for r in runs:
                if r.scenario == sc:
                    loads[r.method].append(zone_loads(read_table(r.path, "zones"), r.seconds))
            files.append(out / f"{sc}-load.png")
            plot_loads(files[-1], sc, loads)
    return files
