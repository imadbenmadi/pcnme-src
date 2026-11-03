"""Replayed SUMO traces with the grid's interface (final-design 10, 16.2, 18; PLAN LuST): a trace map's cars
move as recorded vehicles did, instead of walking the VFC grid.

A trace is one time of day: every vehicle's positions inside the crop, every fcd_period_s, in the crop's own
coordinates (0..crop_m on both axes, the layout's area). The fleet stays the configured size, because
lambda_car is set per car (18): car i replays slot i, a sequence of recorded vehicles that never overlap in
time; when one leaves the crop the next one to enter takes the slot. Between two vehicles the slot is outside
the area, a coverage gap like a grid car's (C12). Each episode replays a slice of the trace chosen by its
seed, so every seed sees other traffic and the same seed the same traffic.

Trace file (data/<map>/traces/<time_of_day>.npz, mobility/lust.py writes it): t (s from the trace start),
vehicle (index), x, y (m), sorted by vehicle then time.
"""

from __future__ import annotations

import functools
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

OUTSIDE = (-1.0e6, -1.0e6)  # a slot between two vehicles: no zone covers it


@dataclass(frozen=True)
class Track:
    """One vehicle's samples inside the crop."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray

    @property
    def start(self) -> float:
        return float(self.t[0])

    @property
    def end(self) -> float:
        return float(self.t[-1])


@functools.lru_cache(maxsize=4)
def load_tracks(path: Path) -> tuple[Track, ...]:
    """Every vehicle's track, in order of entry (cached: training replays the same file every episode)."""
    d = np.load(path)
    t, v, x, y = d["t"], d["vehicle"], d["x"], d["y"]
    cuts = np.flatnonzero(np.diff(v)) + 1
    tracks = [
        Track(tt, xx, yy)
        for tt, xx, yy in zip(np.split(t, cuts), np.split(x, cuts), np.split(y, cuts), strict=True)
        if len(tt) >= 2
    ]
    return tuple(sorted(tracks, key=lambda k: k.start))


def slots(tracks: tuple[Track, ...], begin: float, end: float, n: int) -> list[list[Track]]:
    """n slots of non-overlapping tracks over [begin, end]: each track that is in the crop during the window
    goes to the slot free the earliest (interval scheduling, in order of entry); tracks no slot is free for
    are left out."""
    out: list[list[Track]] = [[] for _ in range(n)]
    free = [(begin, i) for i in range(n)]  # (time the slot is free, slot)
    heapq.heapify(free)
    for k in tracks:
        if k.end < begin or k.start > end:
            continue
        at, i = free[0]
        if at > max(k.start, begin):
            continue  # every slot is busy when this vehicle enters
        heapq.heapreplace(free, (k.end, i))
        out[i].append(k)
    return out


class TraceMover:
    """One car on a trace map: the grid Walker's interface (position, velocity at a time; clock time by
    default), replaying its slot from the episode's offset into the trace."""

    def __init__(self, clock: Any, slot: list[Track], offset: float) -> None:
        self._clock = clock
        self._slot = slot
        self._offset = offset
        self._k = 0  # the slot's current track (time only moves forward)

    def _track(self, t: float) -> Track | None:
        while self._k < len(self._slot) and self._slot[self._k].end < t:
            self._k += 1
        if self._k < len(self._slot) and self._slot[self._k].start <= t:
            return self._slot[self._k]
        return None

    def position(self, t: float | None = None) -> tuple[float, float]:
        at = self._offset + (self._clock.now if t is None else t)
        k = self._track(at)
        if k is None:
            return OUTSIDE
        return float(np.interp(at, k.t, k.x)), float(np.interp(at, k.t, k.y))

    def velocity(self, t: float | None = None) -> tuple[float, float]:
        at = self._offset + (self._clock.now if t is None else t)
        k = self._track(at)
        if k is None:
            return (0.0, 0.0)
        i = int(np.clip(np.searchsorted(k.t, at), 1, len(k.t) - 1))
        dt = float(k.t[i] - k.t[i - 1])
        if dt <= 0:
            return (0.0, 0.0)
        return float(k.x[i] - k.x[i - 1]) / dt, float(k.y[i] - k.y[i - 1]) / dt


def trace_file(traces: Path, time_of_day: str) -> Path:
    return traces / f"{time_of_day}.npz"


def episode_movers(
    clock: Any, path: Path, fleet: int, span: float, rng: np.random.Generator
) -> list[TraceMover]:
    """The fleet's movers for one episode: a slice of `span` seconds at a seed-drawn offset."""
    tracks = load_tracks(path)
    length = max(k.end for k in tracks)
    offset = float(rng.uniform(0.0, max(0.0, length - span)))
    return [TraceMover(clock, s, offset) for s in slots(tracks, offset, offset + span, fleet)]
