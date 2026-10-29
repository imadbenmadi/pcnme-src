"""The VFC grid and a car walking on it (final-design 10, 16.2, 18; PLAN M3.8; QUESTIONS Q24-Q26).

Roads run every block_m (200 m, VFC) in x and y over the area; a car moves along a road at its speed and, at
each intersection, turns at random among the roads that stay on the map, as VFC does (a U-turn included).
The complex grid adds a traffic light at every intersection (half its cycle green per axis) and random
stops (STAND-INS, Q24). Every draw comes from the car's own sequential "mobility" stream, in order (Q4):
its start, its speed, each turn, each stop; times are whole nanoseconds (15.2).

A car may be confined to a set of intersections (the hotspot cars of z1, Q25): it then turns only onto
roads whose far end is in the set.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from pcnme.config import Grid
from pcnme.sim.clock import EventClock
from pcnme.sim.quantize import to_ns

Node = tuple[int, int]  # an intersection, in blocks
DIRECTIONS: tuple[Node, ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))  # east, north, west, south


@dataclass(frozen=True, slots=True)
class RoadGrid:
    """The intersections (0..nx) x (0..ny) of a grid with roads every `block` metres."""

    block: float
    nx: int
    ny: int

    @classmethod
    def from_area(cls, area: tuple[float, float], block: float) -> RoadGrid:
        return cls(block, round(area[0] / block), round(area[1] / block))

    def nodes(self) -> list[Node]:
        return [(i, j) for i in range(self.nx + 1) for j in range(self.ny + 1)]

    def xy(self, node: Node) -> tuple[float, float]:
        return node[0] * self.block, node[1] * self.block

    def neighbours(self, node: Node) -> list[Node]:
        i, j = node
        return [
            (i + di, j + dj) for di, dj in DIRECTIONS if 0 <= i + di <= self.nx and 0 <= j + dj <= self.ny
        ]

    def within(self, centre: tuple[float, float], radius: float) -> frozenset[Node]:
        """The intersections inside a circle (for a confined car, Q25)."""
        return frozenset(n for n in self.nodes() if math.dist(self.xy(n), centre) <= radius)


class Walker:
    """One car on the grid: where it is at any time, and its velocity (10.1: position, speed, heading)."""

    def __init__(
        self,
        clock: EventClock,
        grid: RoadGrid,
        rng: np.random.Generator,
        speeds: Sequence[float],
        cfg: Grid,
        allowed: frozenset[Node] | None = None,
        on_turn: Callable[[], None] | None = None,
    ) -> None:
        self._clock = clock
        self._grid = grid
        self._rng = rng
        self._cfg = cfg
        nodes = sorted(allowed) if allowed is not None else grid.nodes()
        self._allowed = frozenset(nodes)
        self.node: Node = nodes[int(rng.integers(len(nodes)))]  # STAND-IN (Q26): a random intersection
        self.speed = float(speeds[int(rng.integers(len(speeds)))])  # one speed per car (Q26, VFC)
        self._on_turn = on_turn
        self._from: Node = self.node
        self._to: Node = self.node
        self._depart = clock.now  # leaves _from at this time
        self._arrive = clock.now  # reaches _to at this time
        self._choose()

    # --- where the car is -----------------------------------------------------------------------------------

    def position(self, t: float | None = None) -> tuple[float, float]:
        t = self._clock.now if t is None else t
        (x0, y0), (x1, y1) = self._grid.xy(self._from), self._grid.xy(self._to)
        if t <= self._depart or self._arrive <= self._depart:
            return x0, y0
        f = min(1.0, (t - self._depart) / (self._arrive - self._depart))
        return x0 + f * (x1 - x0), y0 + f * (y1 - y0)

    def velocity(self, t: float | None = None) -> tuple[float, float]:
        """(vx, vy) in m/s: zero while waiting at an intersection (10.2: T_exit is infinite when stopped)."""
        t = self._clock.now if t is None else t
        if not self._depart < t < self._arrive:
            return 0.0, 0.0
        dx, dy = self._to[0] - self._from[0], self._to[1] - self._from[1]
        return dx * self.speed, dy * self.speed

    @property
    def heading(self) -> Node:
        return self._to[0] - self._from[0], self._to[1] - self._from[1]

    # --- moving ---------------------------------------------------------------------------------------------

    def _choose(self) -> None:
        """At an intersection: pick the next road (VFC: at random among those on the map), wait for the light
        and a random stop on the complex grid, then drive the block."""
        here = self._to
        options = [n for n in self._grid.neighbours(here) if n in self._allowed]
        nxt = options[int(self._rng.integers(len(options)))]
        wait = 0.0
        now = self._clock.now
        if self._cfg.complex:
            wait += self._light_wait(now, here, nxt)
            if float(self._rng.random()) < self._cfg.stop_probability:
                lo, hi = self._cfg.stop_s
                wait += lo + float(self._rng.random()) * (hi - lo)
        self._from, self._to = here, nxt
        self._depart = to_ns(now + wait)
        self._arrive = to_ns(self._depart + self._grid.block / self.speed)
        self._clock.schedule(self._arrive, self._turn)

    def _light_wait(self, now: float, here: Node, nxt: Node) -> float:
        """STAND-IN (Q24): x-axis roads are green for the first half of each cycle, y-axis the second."""
        cycle = self._cfg.light_cycle_s
        phase = now % cycle
        along_x = nxt[1] == here[1]
        green = phase < cycle / 2 if along_x else phase >= cycle / 2
        if green:
            return 0.0
        return (cycle - phase) if along_x else (cycle / 2 - phase)

    def _turn(self) -> None:
        self.node = self._to
        self._choose()
        if self._on_turn is not None:
            self._on_turn()
