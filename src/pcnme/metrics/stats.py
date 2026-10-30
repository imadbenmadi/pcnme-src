"""Metrics and confidence intervals from the Parquet logs only (final-design 16.3; docs/METRICS.md; PLAN M7).

Per run (one scenario, method and evaluation seed), from tasks, attempts, zones, decisions and events:

    on_time, late, dropped, failed   shares of the covered tasks: the tasks a zone held (tasks.parquet) plus
                                     the frames a car gave up after a cut upload (events car_drop, dropped)
    uncovered                        share of all tasks made in a coverage gap and never held (Q29: apart)
    latency_mean_s, latency_p99_s,   over the delivered tasks (on time or late); worst 1 % = the mean of the
    latency_worst1_s                 ceil(1 %) largest latencies
    energy_per_task_j                dynamic energy per covered task, every attempt (12.5)
    energy_dynamic_j, energy_total_j the run's dynamic energy; + idle power of the nodes over the episode (12)
    refusal_rate                     attempts refused (full or late) / attempts
    wasted_share                     energy of attempts that did not deliver / dynamic energy
    jain_load                        Jain's index over the zones' mean busy share during the episode
    early_on_time                    on-time share of the tasks of the first N decisions (Zhang et al.)

"failed" = a task lost to a failure (drop_lost, drop_no_ack: C14, 9.1-9.5); every other drop_* is
"dropped" (QUESTIONS Q46 stand-in). Across seeds: mean ± the 95 % Student t interval with n − 1 degrees of
freedom (16.3: t = 2.78 for 5 seeds, 2.09 for 20).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FAILED = frozenset({"drop_lost", "drop_no_ack"})  # Q46: lost to a failure (C14)
METRICS = (
    "on_time",
    "late",
    "dropped",
    "failed",
    "uncovered",
    "latency_mean_s",
    "latency_p99_s",
    "latency_worst1_s",
    "energy_per_task_j",
    "energy_dynamic_j",
    "energy_total_j",
    "refusal_rate",
    "wasted_share",
    "jain_load",
    "early_on_time",
)


# --- Student t (16.3) -------------------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """The continued fraction of the regularised incomplete beta function (modified Lentz)."""
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        for aa in (
            m * (b - m) * x / ((qam + m2) * (a + m2)),
            -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)),
        ):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """I_x(a, b), the regularised incomplete beta function."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t >= 0 else tail


def t_quantile(p: float, df: int) -> float:
    """The Student t quantile, by bisection on the CDF."""
    if not 0.0 < p < 1.0 or df < 1:
        raise ValueError(f"t quantile needs 0 < p < 1 and df >= 1, got p {p}, df {df}")
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if t_cdf(mid, df) < p else (lo, mid)
    return 0.5 * (lo + hi)


@dataclass(frozen=True, slots=True)
class Interval:
    """Mean ± half over n seeds (95 % Student t, n − 1 degrees of freedom); half is NaN for one seed."""

    mean: float
    half: float
    n: int

    @property
    def low(self) -> float:
        return self.mean - self.half

    @property
    def high(self) -> float:
        return self.mean + self.half


def interval(values: Sequence[float], level: float = 0.95) -> Interval:
    x = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    n = len(x)
    if n == 0:
        return Interval(math.nan, math.nan, 0)
    if n == 1:
        return Interval(float(x[0]), math.nan, 1)
    s = float(np.std(x, ddof=1))
    return Interval(float(x.mean()), t_quantile(0.5 + level / 2.0, n - 1) * s / math.sqrt(n), n)


def jain(loads: Sequence[float]) -> float:
    """Jain's fairness index (Σx)² / (n Σx²): 1 when equal, 1/n when one zone holds everything."""
    x = np.asarray(loads, dtype=float)
    sq = float((x * x).sum())
    return float(x.sum() ** 2 / (len(x) * sq)) if len(x) and sq > 0 else math.nan


# --- one run ------------------------------------------------------------------------------------------------


def read_table(d: Path, table: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    p = d / f"{table}.parquet"
    return pq.read_table(p).to_pylist() if p.exists() else []


def classify(outcome: str) -> str:
    if outcome in ("on_time", "late"):
        return outcome
    if outcome in FAILED:
        return "failed"
    if outcome.startswith("drop_"):
        return "dropped"
    raise ValueError(f"unknown task outcome {outcome!r}")


def worst_share(x: Sequence[float], share: float = 0.01) -> float:
    """The mean of the ceil(share × n) largest values."""
    if not x:
        return math.nan
    k = max(1, math.ceil(share * len(x)))
    return float(np.mean(sorted(x)[-k:]))


@dataclass(frozen=True, slots=True)
class Run:
    """One run directory's tags and its metrics."""

    path: Path
    scenario: str
    method: str
    seed: int
    seconds: float  # the episode (arrivals); the drain after it is not part of the load
    metrics: dict[str, float]
    variant: str = ""  # an ablation or a sweep point (16.4); empty for the design

    @property
    def label(self) -> str:
        """The method as the tables show it: with its variant in brackets."""
        return f"{self.method} [{self.variant}]" if self.variant else self.method


def run_metrics(d: Path, early_decisions: int) -> Run:
    info = json.loads((d / "run.json").read_text(encoding="utf-8"))
    tasks = read_table(d, "tasks")
    events = read_table(d, "events")
    kinds = [classify(t["outcome"]) for t in tasks]
    gave_up = sum(e["kind"] == "car_drop" for e in events)
    uncovered = sum(e["kind"] == "uncovered" for e in events)
    covered = len(tasks) + gave_up
    m: dict[str, float] = {}
    for k in ("on_time", "late", "failed"):
        m[k] = kinds.count(k) / covered if covered else math.nan
    m["dropped"] = (kinds.count("dropped") + gave_up) / covered if covered else math.nan
    m["uncovered"] = uncovered / (covered + uncovered) if covered + uncovered else math.nan
    lat = [t["latency_s"] for t, k in zip(tasks, kinds, strict=True) if k in ("on_time", "late")]
    m["latency_mean_s"] = float(np.mean(lat)) if lat else math.nan
    m["latency_p99_s"] = float(np.percentile(lat, 99)) if lat else math.nan
    m["latency_worst1_s"] = worst_share(lat)
    dyn = sum(t["energy_j"] for t in tasks)
    m["energy_dynamic_j"] = dyn
    m["energy_per_task_j"] = dyn / covered if covered else math.nan
    m["energy_total_j"] = dyn + info.get("idle_w", math.nan) * info["seconds"]
    wasted = sum(t["energy_wasted_j"] for t in tasks)
    m["wasted_share"] = wasted / dyn if dyn > 0 else math.nan
    attempts = read_table(d, "attempts")
    refused = sum(str(a["result"]).startswith("refused_") for a in attempts)
    m["refusal_rate"] = refused / len(attempts) if attempts else math.nan
    m["jain_load"] = jain(list(zone_loads(read_table(d, "zones"), info["seconds"]).values()))
    m["early_on_time"] = early_on_time(read_table(d, "decisions"), tasks, early_decisions)
    method = info.get("method", tasks[0]["method"] if tasks else "")
    return Run(d, info["scenario"], method, info["seed"], info["seconds"], m, info.get("variant") or "")


def zone_loads(zones: Iterable[Mapping[str, Any]], until: float = math.inf) -> dict[str, float]:
    """Each zone's mean busy share over its state periods before `until` (the episode's end: the drain after
    it, 30 s with no arrivals, is left out) and its slot types (16.3: the per-zone load chart)."""
    acc: dict[str, list[float]] = defaultdict(list)
    for r in zones:
        if r["time_s"] < until:
            acc[r["zone"]].append(r["rho"])
    return {z: float(np.mean(v)) for z, v in sorted(acc.items())}


def early_on_time(
    decisions: Sequence[Mapping[str, Any]], tasks: Sequence[Mapping[str, Any]], n: int
) -> float:
    """16.3 (Zhang et al.): the on-time share of the tasks whose first decision is among the first n
    decisions after deployment (every zone, in time order)."""
    first = sorted((d["time_s"], d["anchor"], d["task_id"]) for d in decisions if d["kind"] == "first")[:n]
    ids = {tid for _, _, tid in first}
    outcome = {t["task_id"]: t["outcome"] for t in tasks}
    picked = [outcome[i] for i in ids if i in outcome]
    return sum(o == "on_time" for o in picked) / len(picked) if picked else math.nan


# --- across seeds -------------------------------------------------------------------------------------------


def summarise(runs: Sequence[Run]) -> dict[tuple[str, str], dict[str, Interval]]:
    """(scenario, method label) -> metric -> mean ± CI over its seeds; a seed counted twice is refused."""
    groups: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for r in runs:
        groups[(r.scenario, r.label)].append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        seeds = [r.seed for r in rs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{key}: a seed appears twice ({sorted(seeds)})")
        out[key] = {k: interval([r.metrics[k] for r in rs]) for k in METRICS}
    return out


# --- the knee reference's windows (16.3; QUESTIONS Q45) ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class WindowPoint:
    """A run's (f1, f2) over a knee-reference window's tasks (13.2.4) and its distance to that front."""

    scenario: str
    method: str
    seed: int
    f1: float
    f2: float
    tasks: int  # window tasks found in this run (a task anchored elsewhere here is not counted)
    distance: float  # to the nearest front point, on axes normalised by the front's ranges


def window_objectives(
    tasks: Iterable[Mapping[str, Any]], keys: set[tuple[str, str]], objective: Mapping[str, Any]
) -> tuple[float, float, int]:
    """13.2.4 from the task rows: c^time and ê per window task (a drop: C_drop and ē), averaged."""
    w3, c_drop, e_bar = objective["w3"], objective["c_drop"], objective["e_bar"]
    e_ref = objective["e_ref"]
    f1 = f2 = 0.0
    n = 0
    for t in tasks:
        if (t["anchor"], t["task_id"]) not in keys:
            continue
        n += 1
        if t["outcome"] in ("on_time", "late"):
            lat, d = t["latency_s"], t["deadline_s"]
            f1 += lat / d + w3 * max(0.0, lat - d) / d
            f2 += min(t["energy_j"] / e_ref[t["type"]], e_bar)
        else:
            f1 += c_drop
            f2 += e_bar
    return (f1 / n, f2 / n, n) if n else (math.nan, math.nan, 0)


def front_distance(point: tuple[float, float], front: Sequence[Sequence[float]]) -> float:
    f1 = [p[0] for p in front]
    f2 = [p[1] for p in front]
    r1 = (max(f1) - min(f1)) or 1.0
    r2 = (max(f2) - min(f2)) or 1.0
    return min(math.hypot((point[0] - a) / r1, (point[1] - b) / r2) for a, b in zip(f1, f2, strict=True))


def window_points(dirs: Sequence[Path]) -> list[WindowPoint]:
    """Every run on the same scenario and seed as a knee-reference run, put on that run's window."""
    infos = {d: json.loads((d / "run.json").read_text(encoding="utf-8")) for d in dirs}
    out = []
    for kd, ki in infos.items():
        if not (kd / "window.json").exists():
            continue
        window = json.loads((kd / "window.json").read_text(encoding="utf-8"))
        keys = {(a, t) for a, t in window["tasks"]}
        for d, i in infos.items():
            if (i["scenario"], i["seed"]) != (ki["scenario"], ki["seed"]):
                continue
            f1, f2, n = window_objectives(read_table(d, "tasks"), keys, i["objective"])
            dist = front_distance((f1, f2), window["front"]) if n else math.nan
            label = f"{i['method']} [{i['variant']}]" if i.get("variant") else i["method"]
            out.append(WindowPoint(i["scenario"], label, i["seed"], f1, f2, n, dist))
    return sorted(out, key=lambda p: (p.scenario, p.seed, p.method))
