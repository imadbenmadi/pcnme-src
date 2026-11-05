"""Comparisons from summary.csv rows (final-design 16.3; PLAN T3c).

    paired.csv / .md      PCNME vs each baseline per scenario and metric: the mean paired difference over
                          the seeds both ran, the paired Student t statistic and its two-sided p value
    across-<metric>.png   every method's mean over the scenarios, one figure per key metric
    transfer.csv / .md, transfer-<metric>.png   models trained on one map, evaluated on it and another

The paired test uses the same Student t as the 95 % intervals (metrics/stats.py): the evaluation seeds are
shared by all methods, so each seed is one pair.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pcnme.metrics.stats import t_cdf

KEY = (  # metric, label, scale
    ("on_time", "on time %", 100.0),
    ("latency_p99_s", "p99 latency ms", 1000.0),
    ("mean_e_hat", "mean ê", 1.0),
    ("energy_per_task_j", "J / task", 1.0),
)
FIGURES = ("on_time", "latency_p99_s", "mean_e_hat")
Row = Mapping[str, str]


@dataclass(frozen=True)
class Paired:
    diff: float  # mean of a − b
    t: float
    p: float  # two-sided; NaN with fewer than two pairs
    n: int


def paired(a: Sequence[float], b: Sequence[float]) -> Paired:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    if n < 2:
        return Paired(float(d.mean()) if n else math.nan, math.nan, math.nan, n)
    mean, sd = float(d.mean()), float(np.std(d, ddof=1))
    if sd == 0.0:
        return Paired(mean, math.copysign(math.inf, mean) if mean else 0.0, 0.0 if mean else 1.0, n)
    t = mean / (sd / math.sqrt(n))
    return Paired(mean, t, 2.0 * (1.0 - t_cdf(abs(t), n - 1)), n)


def _by(rows: Sequence[Row], metric: str) -> dict[tuple[str, str], dict[int, float]]:
    out: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in rows:
        out[(r["scenario"], r["method"])][int(r["seed"])] = float(r[metric])
    return out


def paired_rows(rows: Sequence[Row], ours: str, metrics: Sequence[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for metric in metrics:
        by = _by(rows, metric)
        for sc, me in sorted(by):
            if me == ours or (sc, ours) not in by:
                continue
            mine, theirs = by[(sc, ours)], by[(sc, me)]
            seeds = sorted(set(mine) & set(theirs))
            r = paired([mine[s] for s in seeds], [theirs[s] for s in seeds])
            out.append({"scenario": sc, "metric": metric, "baseline": me, "ours": ours,
                        "ours_mean": float(np.mean([mine[s] for s in seeds])) if seeds else math.nan,
                        "baseline_mean": float(np.mean([theirs[s] for s in seeds])) if seeds else math.nan,
                        "diff": r.diff, "t": r.t, "p": r.p, "n": r.n})  # fmt: skip
    return out


def across_scenarios(rows: Sequence[Row], metric: str) -> dict[str, dict[str, float]]:
    """method -> scenario -> mean over the seeds."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (sc, me), seeds in sorted(_by(rows, metric).items()):
        out[me][sc] = float(np.mean(list(seeds.values())))
    return dict(sorted(out.items()))


def transfer_rows(
    home: Sequence[Row], other: Sequence[Row], metrics: Sequence[str], home_name: str, other_name: str
) -> list[dict[str, Any]]:
    """Per scenario, method and metric: the mean on the home map and on the other map."""
    out: list[dict[str, Any]] = []
    for metric in metrics:
        h, o = across_scenarios(home, metric), across_scenarios(other, metric)
        for me in sorted(set(h) & set(o)):
            for sc in sorted(set(h[me]) & set(o[me])):
                out.append({"scenario": sc, "method": me, "metric": metric,
                            home_name: h[me][sc], other_name: o[me][sc]})  # fmt: skip
    return out


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["empty"])
        w.writeheader()
        w.writerows(rows)
    return path


def _num(v: float, scale: float = 1.0, digits: int = 2) -> str:
    return "–" if math.isnan(v) else f"{v * scale:.{digits}f}"


def _paired_md(path: Path, rows: Sequence[Mapping[str, Any]], ours: str) -> Path:
    lines = [f"# {ours} vs each baseline (paired t test over the shared seeds)", "",
             f"diff = {ours} − baseline (mean over the seeds); p < 0.05: not seed noise.", ""]  # fmt: skip
    for metric, label, scale in KEY:
        sub = [r for r in rows if r["metric"] == metric]
        if not sub:
            continue
        lines += [f"## {label}", "", f"| scenario | baseline | {ours} | baseline | diff | p | n |"]
        lines.append("| --- " * 7 + "|")
        for r in sub:
            p = "–" if math.isnan(r["p"]) else ("< 0.001" if r["p"] < 0.001 else f"{r['p']:.3f}")
            means = f"{_num(r['ours_mean'], scale)} | {_num(r['baseline_mean'], scale)}"
            lines.append(
                f"| {r['scenario']} | {r['baseline']} | {means} | {_num(r['diff'], scale)} | {p} | {r['n']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _plot_lines(
    path: Path, series: Mapping[str, Mapping[str, float]], label: str, scale: float, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = sorted({s for v in series.values() for s in v})
    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    for me, v in series.items():
        ax.plot(scenarios, [v.get(s, math.nan) * scale for s in scenarios], marker="o",
                linewidth=2.4 if me == "pcnme" else 1.2, label=me)  # fmt: skip
    ax.set(xlabel="scenario", ylabel=label, title=title)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    ax.legend(fontsize=7, frameon=False, ncols=2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_all(rows: Sequence[Row], out: Path, ours: str = "pcnme") -> list[Path]:
    """paired.csv, paired.md and across-<metric>.png from one map's summary rows."""
    out.mkdir(parents=True, exist_ok=True)
    pr = paired_rows(rows, ours, [k for k, _, _ in KEY])
    files = [_csv(out / "paired.csv", pr), _paired_md(out / "paired.md", pr, ours)]
    for metric, label, scale in KEY:
        if metric in FIGURES:
            files.append(out / f"across-{metric}.png")
            _plot_lines(files[-1], across_scenarios(rows, metric), label, scale, f"{label} by scenario")
    return files


def write_transfer(
    home: Sequence[Row], other: Sequence[Row], out: Path, home_name: str, other_name: str
) -> list[Path]:
    """transfer.csv / .md and one figure per key metric: each method on its training map and on the other."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    tr = transfer_rows(home, other, [k for k, _, _ in KEY], home_name, other_name)
    files = [_csv(out / "transfer.csv", tr)]
    lines = [
        f"# Trained on {home_name}, evaluated on {home_name} and on {other_name} (no {other_name} training)",
        "",
    ]
    for metric, label, scale in KEY:
        sub = [r for r in tr if r["metric"] == metric]
        if not sub:
            continue
        lines += [f"## {label}", "", f"| scenario | method | {home_name} | {other_name} | change |"]
        lines.append("| --- " * 5 + "|")
        for r in sub:
            h, o = r[home_name], r[other_name]
            lines.append(f"| {r['scenario']} | {r['method']} | {_num(h, scale)} | {_num(o, scale)} | "
                         f"{_num(o - h, scale)} |")  # fmt: skip
        lines.append("")
        if metric in FIGURES:
            methods = sorted({r["method"] for r in sub})
            h_mean = [
                float(np.mean([r[home_name] for r in sub if r["method"] == m])) * scale for m in methods
            ]
            o_mean = [
                float(np.mean([r[other_name] for r in sub if r["method"] == m])) * scale for m in methods
            ]
            x = np.arange(len(methods))
            fig, ax = plt.subplots(figsize=(1.0 * len(methods) + 2, 3.6), constrained_layout=True)
            ax.bar(x - 0.2, h_mean, 0.4, label=f"{home_name} (trained here)")
            ax.bar(x + 0.2, o_mean, 0.4, label=f"{other_name} (zero-shot)")
            ax.set_xticks(x, methods, rotation=45)
            ax.set(ylabel=label, title=f"{label}: mean over the scenarios")
            ax.legend(fontsize=8, frameon=False)
            files.append(out / f"transfer-{metric}.png")
            fig.savefig(files[-1], dpi=150)
            plt.close(fig)
    files.append(out / "transfer.md")
    files[-1].write_text("\n".join(lines), encoding="utf-8")
    return files
