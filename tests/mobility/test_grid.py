"""The VFC grid and the walker (PLAN M3.8; final-design 10, 16.2, 18; QUESTIONS Q24-Q26)."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pcnme.config import Config, load_config
from pcnme.mobility.grid import RoadGrid, Walker
from pcnme.sim.clock import EventClock
from pcnme.sim.rng import RandomStreams


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


def walker(cfg: Config, car: str = "c01", seed: int = 2000, **kw: object) -> tuple[EventClock, Walker]:
    clock = EventClock()
    grid = RoadGrid.from_area(cfg.layout.area_m, cfg.cars.grid.block_m)
    rng = RandomStreams.from_config(cfg, seed, "training").generator("mobility", car)
    w = Walker(
        clock, grid, rng, kw.pop("speeds", cfg.cars.speed_low_mps), kw.pop("grid", cfg.cars.grid), **kw
    )  # type: ignore[arg-type]
    return clock, w


def test_the_grid_has_roads_every_200_m_over_the_area(cfg: Config) -> None:
    g = RoadGrid.from_area(cfg.layout.area_m, cfg.cars.grid.block_m)
    assert (g.nx, g.ny, len(g.nodes())) == (6, 6, 49)
    assert g.xy((3, 2)) == (600.0, 400.0)
    assert sorted(g.neighbours((0, 0))) == [(0, 1), (1, 0)]  # a corner: the map edge stops two roads
    assert len(g.neighbours((3, 3))) == 4


def test_a_car_drives_along_a_road_at_its_speed(cfg: Config) -> None:
    clock, w = walker(cfg)
    assert w.speed in cfg.cars.speed_low_mps
    start = w.position()
    clock.run_until(1.0)
    moved = math.dist(start, w.position())
    assert moved == pytest.approx(w.speed * 1.0, rel=1e-9)
    vx, vy = w.velocity()
    assert math.hypot(vx, vy) == pytest.approx(w.speed) and 0 in (vx, vy)  # along x or y


def test_it_stays_on_roads_and_on_the_map_and_turns_at_intersections(cfg: Config) -> None:
    clock, w = walker(cfg, speeds=(11.11,))
    for k in range(1, 4001):
        clock.run_until(k * 0.5)
        x, y = w.position()
        assert 0 <= x <= 1200 and 0 <= y <= 1200
        assert min(x % 200, 200 - x % 200) < 1e-6 or min(y % 200, 200 - y % 200) < 1e-6  # on a road


def test_turns_are_uniform_among_the_roads_that_stay_on_the_map(cfg: Config) -> None:
    # VFC: at an inner intersection each of the 4 roads (a U-turn included) is taken a quarter of the time.
    clock, w = walker(cfg, speeds=(40.0,), seed=2001)
    seen = {d: 0 for d in ((1, 0), (0, 1), (-1, 0), (0, -1))}
    for _ in range(20000):
        clock.run_until(w._arrive)
        if 1 <= w._from[0] <= 5 and 1 <= w._from[1] <= 5:
            seen[w.heading] += 1
    n = sum(seen.values())
    assert all(abs(v / n - 0.25) < 0.02 for v in seen.values())


def test_the_same_seed_gives_the_same_trajectory_and_cars_differ(cfg: Config) -> None:
    def path(car: str) -> list[tuple[float, float]]:
        clock, w = walker(cfg, car=car)
        out = []
        for k in range(1, 200):
            clock.run_until(k * 1.0)
            out.append(w.position())
        return out

    assert path("c01") == path("c01") and path("c01") != path("c02")


def test_a_confined_car_never_leaves_its_intersections(cfg: Config) -> None:
    # Q25: a hotspot car stays on the roads between z1's intersections (those inside its 350 m circle).
    g = RoadGrid.from_area(cfg.layout.area_m, cfg.cars.grid.block_m)
    z1 = g.within(cfg.layout.zone_positions_m["z1"], cfg.layout.zone_radius_m)
    assert (0, 0) not in z1 and (1, 1) in z1 and (3, 1) in z1 and (3, 3) not in z1
    clock, w = walker(cfg, speeds=(11.11,), allowed=z1)
    for k in range(1, 2001):
        clock.run_until(k * 1.0)
        assert math.dist(w.position(), (300.0, 300.0)) <= 350.0 + 1e-6


def test_the_complex_grid_waits_at_red_lights_and_random_stops(cfg: Config) -> None:
    # Q24 stand-ins: lights with a 60 s cycle (x green the first half), stops with p = 0.1 for U(5, 20) s.
    complex_grid = replace(cfg.cars.grid, complex=True)
    clock, w = walker(cfg, grid=complex_grid, speeds=(10.0,))
    waits = []
    for _ in range(2000):
        clock.run_until(w._arrive)
        waits.append(w._depart - clock.now)
        if w._depart > clock.now:
            assert w.velocity((clock.now + w._depart) / 2) == (0.0, 0.0)  # stopped: speed 0 (C8)
    waits_arr = np.array(waits)
    assert (waits_arr == 0).mean() > 0.2 and waits_arr.max() <= 60.0 + 20.0
    simple_clock, simple = walker(cfg, speeds=(10.0,))
    simple_clock.run_until(simple._arrive)
    assert simple._depart == simple_clock.now  # the simple grid never waits


def test_red_lights_alone_hold_a_car_up_to_half_a_cycle(cfg: Config) -> None:
    # Q24 stand-in with stops off: a car arriving on red waits until its axis turns green, at most 30 s.
    lights_only = replace(cfg.cars.grid, complex=True, stop_probability=0.0)
    clock, w = walker(cfg, grid=lights_only, speeds=(10.0,))
    waits = []
    for _ in range(500):
        clock.run_until(w._arrive)
        waits.append(w._depart - clock.now)
    assert any(x > 0 for x in waits) and max(waits) <= 30.0 + 1e-9
