"""The 16.3 metrics the evaluation runs did not compute (PLAN "Missing metrics"; docs/METRICS.md), from logs
that already exist, so nothing is rerun:

- next-zone acc@1 and T_exit error (10.2): the level-2 predictor (mobility/predictor.py) replayed over the
  logged car positions and compared with where each car really went. The predictor and the mobility do not
  depend on the placement method, so one method's runs per scenario and seed are enough.
- delivery success after a zone change: the tasks whose car's serving zone changed between the task's creation
  and its close; the share delivered (on time or late) and the share on time.
- decisions until the on-time rate is stable: decisions made until the on-time share over the last
  STABLE_WINDOW tasks reaches STABLE_LEVEL and stays there for the rest of the run (QUESTIONS Q63 stand-in).
- the hypervolume of the NSGA-II fronts (13.2): per window, the area dominated by the front up to a reference
  point; the reference is 1.1 x the worst f1 and f2 of the scenario's fronts and greedy plans (Q63 stand-in),
  so windows of one scenario are comparable.

All functions take plain columns (dicts of lists, as pyarrow's to_pydict gives) so they are cheap to test.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pcnme.mobility.predictor import exit_time, next_zone

Point = tuple[float, float]
STABLE_WINDOW = 1000  # Q63 stand-in: tasks in the rolling on-time share
STABLE_LEVEL = 0.95  # METRICS.md: "until the on-time rate reaches 95 %"
SAMPLE_EVERY_S = 1.0  # one prediction per car per second (GPS is every 0.1 s; neighbours are near-identical)
REF_MARGIN = 1.1  # Q63 stand-in: the hypervolume reference point, 10 % past the worst point


# ---- next-zone acc@1 and T_exit error ----------------------------------------------------------------------


def predictor_errors(
    pos: Mapping[str, Sequence[Any]],
    centres: Mapping[str, Point],
    radius: float,
    every: float = SAMPLE_EVERY_S,
) -> dict[str, float | int | None]:
    """Replay the predictor on `pos` (positions.parquet columns: time_s, car_id, x_m, y_m, speed_mps,
    heading_x, heading_y, zone). A sample is a moving car served by a zone, every `every` seconds. acc@1 over
    the samples whose car later reached another zone; T_exit error over those whose car left its zone in the
    run."""
    tracks: dict[str, list[tuple[float, float, float, float, float, float, str]]] = {}
    for t, c, x, y, v, hx, hy, z in zip(
        pos["time_s"], pos["car_id"], pos["x_m"], pos["y_m"], pos["speed_mps"], pos["heading_x"],
        pos["heading_y"], pos["zone"], strict=True,
    ):  # fmt: skip
        tracks.setdefault(c, []).append((t, x, y, v, hx, hy, z or ""))
    hits = tried = 0
    errors: list[float] = []
    for rows in tracks.values():
        rows.sort()
        times = [r[0] for r in rows]
        # for each index: the first later index whose serving zone differs (the car left its zone there)
        n = len(rows)
        exit_at: list[int | None] = [None] * n
        later: int | None = None
        for i in range(n - 1, -1, -1):
            if i + 1 < n and rows[i + 1][6] != rows[i][6]:
                later = i + 1
            exit_at[i] = later
        last_sample = -math.inf
        for i, (t, x, y, v, hx, hy, z) in enumerate(rows):
            if not z or z not in centres or v <= 0 or t - last_sample < every - 1e-9:
                continue
            last_sample = t
            vel = (hx * v, hy * v)
            pred = next_zone((x, y), vel, z, centres, radius)
            actual = _first_other(rows, i, z)
            if pred is not None and actual is not None:
                tried += 1
                hits += int(pred == actual)
            j = exit_at[i]
            t_pred = exit_time((x, y), vel, centres[z], radius)
            if j is not None and math.isfinite(t_pred):
                errors.append(abs(t_pred - (times[j] - t)))
    errors.sort()
    return {
        "acc1": hits / tried if tried else None,
        "predictions": tried,
        "t_exit_mae_s": sum(errors) / len(errors) if errors else None,
        "t_exit_median_s": errors[len(errors) // 2] if errors else None,
        "exits": len(errors),
    }


def _first_other(
    rows: Sequence[tuple[float, float, float, float, float, float, str]], i: int, z: str
) -> str | None:
    """The first served zone other than z after index i (a gap in between is skipped)."""
    for r in rows[i + 1 :]:
        if r[6] and r[6] != z:
            return r[6]
    return None


# ---- delivery success after a zone change ------------------------------------------------------------------


def zone_change_delivery(
    tasks: Mapping[str, Sequence[Any]], pos: Mapping[str, Sequence[Any]]
) -> dict[str, float | int | None]:
    """The tasks whose car's serving zone at creation differs from the one at close (created + latency, or +
    deadline for a task never delivered): how many, and their delivered and on-time shares."""
    zone_at: dict[str, tuple[list[float], list[str]]] = {}
    order = sorted(range(len(pos["time_s"])), key=lambda k: (pos["car_id"][k], pos["time_s"][k]))
    for k in order:
        ts, zs = zone_at.setdefault(pos["car_id"][k], ([], []))
        ts.append(pos["time_s"][k])
        zs.append(pos["zone"][k] or "")

    def zone(car: str, t: float) -> str:
        ts, zs = zone_at.get(car, ([], []))
        i = bisect.bisect_right(ts, t) - 1
        return zs[i] if i >= 0 else ""

    n = delivered = on_time = 0
    for tid, created, outcome, lat, dl in zip(
        tasks["task_id"],
        tasks["time_s"],
        tasks["outcome"],
        tasks["latency_s"],
        tasks["deadline_s"],
        strict=True,
    ):
        car = str(tid).split(".")[0]
        start = zone(car, created)
        end = zone(car, created + (lat if lat is not None else dl))
        if not start or not end or start == end:
            continue
        n += 1
        delivered += int(outcome in ("on_time", "late"))
        on_time += int(outcome == "on_time")
    return {"tasks": n, "delivered": delivered / n if n else None, "on_time": on_time / n if n else None}


# ---- decisions until the on-time rate is stable ------------------------------------------------------------


def decisions_until_stable(
    tasks: Mapping[str, Sequence[Any]],
    decision_times: Iterable[float],
    window: int = STABLE_WINDOW,
    level: float = STABLE_LEVEL,
) -> int | None:
    """Decisions made until the on-time share of the last `window` tasks (by creation) reaches `level` and
    stays at or above it for the rest of the run; None if it never does (or fewer tasks than one window)."""
    order = sorted(range(len(tasks["time_s"])), key=lambda k: tasks["time_s"][k])
    ok = [int(tasks["outcome"][k] == "on_time") for k in order]
    times = [tasks["time_s"][k] for k in order]
    if len(ok) < window:
        return None
    run = sum(ok[:window])
    good = [run / window >= level]
    for j in range(window, len(ok)):
        run += ok[j] - ok[j - window]
        good.append(run / window >= level)
    if not good[-1]:
        return None
    first = len(good) - 1
    while first > 0 and good[first - 1]:
        first -= 1
    t_stable = times[first + window - 1]  # the window ending here is the first of the good tail
    dts = sorted(decision_times)
    return bisect.bisect_right(dts, t_stable)


# ---- hypervolume of the NSGA-II fronts ---------------------------------------------------------------------


def hypervolume_2d(points: Iterable[Sequence[float]], ref: Sequence[float]) -> float:
    """The area dominated by `points` (both objectives minimised) and bounded by `ref`: a sweep over f1, each
    non-dominated point adding the strip up to the next one's f1."""
    pts = sorted((float(p[0]), float(p[1])) for p in points if p[0] < ref[0] and p[1] < ref[1])
    front: list[Point] = []
    for p in pts:
        if not front or p[1] < front[-1][1]:
            front.append(p)
    area = 0.0
    for i, (f1, f2) in enumerate(front):
        right = front[i + 1][0] if i + 1 < len(front) else ref[0]
        area += (right - f1) * (ref[1] - f2)
    return area


def front_hypervolumes(windows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per NSGA-II window (data/<map>/nsga/<S>-w<k>.json, as loaded): its front's hypervolume, the greedy
    plan's, and their ratio. The reference point is per scenario (REF_MARGIN x the worst f1 and f2 seen)."""
    by_scenario: dict[str, list[Mapping[str, Any]]] = {}
    for w in windows:
        by_scenario.setdefault(str(w["scenario"]), []).append(w)
    out = []
    for s, ws in sorted(by_scenario.items()):
        f1s = [p[0] for w in ws for p in w["F"]] + [w["greedy"][0] for w in ws]
        f2s = [p[1] for w in ws for p in w["F"]] + [w["greedy"][1] for w in ws]
        ref = (REF_MARGIN * max(f1s), REF_MARGIN * max(f2s))
        for w in sorted(ws, key=lambda w: int(w["index"])):
            hv = hypervolume_2d(w["F"], ref)
            hv_g = hypervolume_2d([(w["greedy"][0], w["greedy"][1])], ref)
            out.append({"scenario": s, "window": int(w["index"]), "points": len(w["F"]), "hv": hv,
                        "hv_greedy": hv_g, "hv_ratio": hv / hv_g if hv_g else None,
                        "ref_f1": ref[0], "ref_f2": ref[1]})  # fmt: skip
    return out
