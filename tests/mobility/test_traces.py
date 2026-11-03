"""Trace maps (PLAN LuST; final-design 10, 16.2): recorded vehicles replayed with the grid's interface."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from pcnme.mobility.traces import OUTSIDE, TraceMover, episode_movers, load_tracks, slots
from pcnme.sim.clock import EventClock


def write_trace(path: Path, tracks: list[tuple[float, float, float]]) -> Path:
    """Vehicles moving along x at 10 m/s: (start, end, y), sampled every 0.1 s."""
    t, v, x, y = [], [], [], []
    for i, (a, b, yy) in enumerate(tracks):
        ts = np.arange(a, b + 1e-9, 0.1)
        t += list(ts)
        v += [i] * len(ts)
        x += list(10.0 * (ts - a))
        y += [yy] * len(ts)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"t": np.array(t, np.float32), "vehicle": np.array(v, np.int32)}
    np.savez_compressed(path, **arrays, x=np.array(x, np.float32), y=np.array(y, np.float32))
    load_tracks.cache_clear()
    return path


def test_a_slot_holds_vehicles_that_never_overlap(tmp_path: Path) -> None:
    p = write_trace(tmp_path / "t.npz", [(0, 10, 100), (5, 20, 200), (12, 30, 300), (25, 40, 400)])
    out = slots(load_tracks(p), 0.0, 40.0, 2)
    for s in out:
        for a, b in itertools.pairwise(s):
            assert a.end <= b.start
    assert sum(len(s) for s in out) == 4


def test_a_mover_interpolates_and_is_outside_between_vehicles(tmp_path: Path) -> None:
    p = write_trace(tmp_path / "t.npz", [(0, 10, 100), (20, 30, 300)])
    clock = EventClock()
    (slot,) = slots(load_tracks(p), 0.0, 30.0, 1)
    m = TraceMover(clock, slot, offset=0.0)
    assert m.position(5.05) == pytest.approx((50.5, 100.0), abs=1e-3)
    assert m.velocity(5.05) == pytest.approx((10.0, 0.0), abs=1e-3)
    assert m.position(15.0) == OUTSIDE  # a gap between two vehicles
    assert m.position(25.0) == pytest.approx((50.0, 300.0), abs=1e-3)


def test_the_same_seed_replays_the_same_slice(tmp_path: Path) -> None:
    p = write_trace(tmp_path / "t.npz", [(float(k), float(k) + 30, 10.0 * k) for k in range(60)])

    def first_positions(seed: int) -> list[tuple[float, float]]:
        ms = episode_movers(EventClock(), p, 5, 10.0, np.random.default_rng(seed))
        return [m.position(1.0) for m in ms]

    assert first_positions(1) == first_positions(1)
    assert first_positions(1) != first_positions(2)


def test_a_city_runs_on_a_trace_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The map's scenario trace drives the cars; tasks are made and closed as on the grid."""
    from pcnme.experiments.city import City
    from pcnme.maps import map_config

    monkeypatch.setenv("PCNME_DATA_DIR", str(tmp_path))
    rows = [(float(k % 50), float(k % 50) + 200.0, 50.0 + 11.0 * (k % 100)) for k in range(300)]
    cfg = map_config("lust")
    write_trace(tmp_path / "lust" / "traces" / f"{cfg.map.scenario_time['S1']}.npz", rows)
    city = City(cfg, "S1", "greedy", 2000, n_cars=40, logging=False).run(5.0)
    assert city.records and any(r.outcome == "on_time" for r in city.records)
