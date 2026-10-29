"""The knee point and the weight w (final-design 13.3; PLAN M5.3). THIS TOUCHES THE AGENT: w is input 48 and
the time weight of every reward (11.4).

Normalise the final front: f̃_i = (f_i - f_i^min) / R_i with R_i = f_i^max - f_i^min. The knee is the point
with the smallest f̃1 + f̃2 (the farthest from the line through the two extremes). The local slope from its two
neighbours on the front sorted by f̃1 (one-sided at an end):

    s = (f̃2(p-) - f̃2(p+)) / (f̃1(p+) - f̃1(p-)),    w̃ = s / (1 + s)

and back through the ranges to the raw per-task costs the reward uses:

    w = (w̃ / R1) / (w̃ / R1 + (1 - w̃) / R2),    w1 = w, w2 = 1 - w

Every front point p gets its own w_p by the same formulas (conditioning, 13.3). w* = the median knee weight
over the training windows. Engineering choices: duplicate points count once; a tie on f̃1 + f̃2 goes to the
point with the smaller f̃1 (the faster plan); a front with fewer than 2 distinct points has no knee weight.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Knee:
    point: Point  # (f1, f2) of the knee
    index: int  # among the front's distinct points sorted by f1
    slope: float  # s
    w_tilde: float  # on normalised axes
    w: float  # on the raw costs: w1 = w, w2 = 1 - w
    ranges: tuple[float, float]  # R1, R2


def distinct_sorted(points: Sequence[Point]) -> list[Point]:
    return sorted(set(points))


def _normalised(pts: list[Point]) -> tuple[list[Point], float, float]:
    f1 = [p[0] for p in pts]
    f2 = [p[1] for p in pts]
    r1, r2 = max(f1) - min(f1), max(f2) - min(f2)
    if r1 <= 0 or r2 <= 0:
        raise ValueError(f"a front needs a range on both objectives, got R1 = {r1}, R2 = {r2}")
    return [((a - min(f1)) / r1, (b - min(f2)) / r2) for a, b in pts], r1, r2


def weight_at(pts: list[Point], i: int) -> tuple[float, float, float, float, float]:
    """(s, w̃, w, R1, R2) at the i-th of the distinct points sorted by f1 (13.3)."""
    n, r1, r2 = _normalised(pts)
    lo, hi = max(i - 1, 0), min(i + 1, len(n) - 1)  # one-sided at an end
    s = (n[lo][1] - n[hi][1]) / (n[hi][0] - n[lo][0])
    wt = s / (1 + s)
    w = (wt / r1) / (wt / r1 + (1 - wt) / r2)
    return s, wt, w, r1, r2


def knee(points: Sequence[Point]) -> Knee | None:
    """The knee of a final front and its weight; None with fewer than 2 distinct points."""
    pts = distinct_sorted(points)
    if len(pts) < 2:
        return None
    n, _, _ = _normalised(pts)
    i = min(range(len(n)), key=lambda k: (n[k][0] + n[k][1], n[k][0]))
    s, wt, w, r1, r2 = weight_at(pts, i)
    return Knee(pts[i], i, s, wt, w, (r1, r2))


def weights(points: Sequence[Point]) -> dict[Point, float]:
    """w_p for every distinct front point (13.3: training conditioned on w)."""
    pts = distinct_sorted(points)
    if len(pts) < 2:
        return {}
    return {p: weight_at(pts, i)[2] for i, p in enumerate(pts)}


def w_star(knees: Sequence[Knee | None]) -> float:
    """13.3: the deployment value w*, the median of the knee weights over the windows that have one."""
    ws = [k.w for k in knees if k is not None]
    if not ws:
        raise ValueError("no window has a knee weight")
    return statistics.median(ws)
