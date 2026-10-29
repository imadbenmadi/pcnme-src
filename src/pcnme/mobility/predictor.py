"""The mobility predictor (final-design 10.2, 7.4 C3 and C8; PLAN M3.9; QUESTIONS Q27).

Level 2 (motion math + road grid), where PCNME starts:
1. Exit time: when the car, going straight on from p with velocity v, crosses its zone's coverage circle
   (centre c, radius R_z): T_exit = (-b + sqrt(b^2 + |v|^2 (R_z^2 - |p - c|^2))) / |v|^2, b = (p - c) . v;
   infinity when stopped (C8). Inside the circle every heading leaves it.
2. Next zone: going straight, the zone whose circle the path enters next after leaving this one; if it
   enters none (a gap or the map edge), the nearest other centre to the exit point (Q27). At an
   intersection only the 2-3 roads on offer are possible: their next zones are the candidates, straight on
   first (Q27).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

Point = tuple[float, float]


def exit_time(p: Point, v: Point, c: Point, radius: float) -> float:
    """T_exit (10.2.1), seconds; infinity when stopped. For a car outside the circle it is the time until it
    leaves the circle's line of travel, 0 if it is moving away (it has already left)."""
    vv = v[0] * v[0] + v[1] * v[1]
    if vv == 0.0:
        return math.inf
    dx, dy = p[0] - c[0], p[1] - c[1]
    b = dx * v[0] + dy * v[1]
    disc = b * b + vv * (radius * radius - (dx * dx + dy * dy))
    if disc < 0:
        return 0.0  # outside, and the line misses the circle
    return max(0.0, (-b + math.sqrt(disc)) / vv)


def _enter_time(p: Point, v: Point, c: Point, radius: float) -> float | None:
    """When the straight path from p enters circle c (None if it never does from outside)."""
    vv = v[0] * v[0] + v[1] * v[1]
    dx, dy = p[0] - c[0], p[1] - c[1]
    b = dx * v[0] + dy * v[1]
    disc = b * b + vv * (radius * radius - (dx * dx + dy * dy))
    if vv == 0.0 or disc < 0:
        return None
    t = (-b - math.sqrt(disc)) / vv
    return t if t >= 0 else None


def next_zone(p: Point, v: Point, zone: str, centres: Mapping[str, Point], radius: float) -> str | None:
    """10.2.2, going straight: the first other circle the path enters after leaving `zone`'s; else the
    nearest other centre to the exit point (Q27). None when stopped."""
    t_exit = exit_time(p, v, centres[zone], radius)
    if math.isinf(t_exit):
        return None
    out = (p[0] + v[0] * t_exit, p[1] + v[1] * t_exit)
    entering = []
    for z, c in centres.items():
        if z == zone:
            continue
        if math.dist(out, c) <= radius:  # already inside it where it leaves this one (the overlap)
            entering.append((0.0, z))
            continue
        t = _enter_time(out, v, c, radius)
        if t is not None:
            entering.append((t, z))
    if entering:
        return min(entering)[1]
    return min((math.dist(out, c), z) for z, c in centres.items() if z != zone)[1]


def candidates(
    p: Point,
    speed: float,
    roads: Sequence[Point],
    heading: Point,
    zone: str,
    centres: Mapping[str, Point],
    radius: float,
) -> list[str]:
    """10.2.2 at an intersection: the next zones of the roads on offer (unit directions), straight on first,
    each once (Q27)."""
    order = sorted(roads, key=lambda d: (d != heading, d))
    out: list[str] = []
    for d in order:
        z = next_zone(p, (d[0] * speed, d[1] * speed), zone, centres, radius)
        if z is not None and z not in out:
            out.append(z)
    return out
