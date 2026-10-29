"""The city with moving cars (PLAN M3.8; final-design 9.6, 9.7, 10, 18; QUESTIONS Q24-Q26). Scripted cars
move on known paths, so attach, cut and delivery times are by hand."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import pytest

from pcnme.config import Config, load_config
from pcnme.experiments.city import City, choose_zone, hotspot_fraction
from pcnme.ids import TaskId
from tests.zone.harness import Fixed, fixed_delay_config

CENTRES = {"z1": (300.0, 300.0), "z2": (900.0, 300.0), "z3": (300.0, 900.0), "z4": (900.0, 900.0)}


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


class Line:
    """A scripted car: from a to b at a constant speed, starting at t0, then parked at b."""

    def __init__(self, a: tuple[float, float], b: tuple[float, float], speed: float, t0: float = 0.0) -> None:
        self.a, self.b, self.v, self.t0 = a, b, speed, t0
        self.length = math.dist(a, b)

    def _f(self, t: float) -> float:
        if self.length == 0:
            return 0.0
        return min(1.0, max(0.0, (t - self.t0) * self.v / self.length))

    def position(self, t: float | None = None) -> tuple[float, float]:
        f = self._f(t or 0.0)
        return self.a[0] + f * (self.b[0] - self.a[0]), self.a[1] + f * (self.b[1] - self.a[1])

    def velocity(self, t: float | None = None) -> tuple[float, float]:
        f = self._f(t or 0.0)
        if f in (0.0, 1.0):
            return 0.0, 0.0
        return (
            (self.b[0] - self.a[0]) / self.length * self.v,
            (self.b[1] - self.a[1]) / self.length * self.v,
        )


# --- choosing the access point (C12, C13, 18) ---------------------------------------------------------------


def test_the_nearest_access_point_in_range_or_a_gap() -> None:
    assert choose_zone((100.0, 300.0), None, CENTRES, 350.0, 20.0) == "z1"
    assert choose_zone((600.0, 600.0), None, CENTRES, 350.0, 20.0) is None  # the centre: a gap until z5
    assert choose_zone((0.0, 0.0), None, CENTRES, 350.0, 20.0) is None  # a corner (424 m)


def test_the_current_access_point_is_kept_unless_another_is_20_m_closer() -> None:
    # In the 100 m overlap of z1 and z2 (x 550-650): at x = 609 z2 is 18 m closer: keep z1; at 611, 22 m.
    assert choose_zone((609.0, 300.0), "z1", CENTRES, 350.0, 20.0) == "z1"
    assert choose_zone((611.0, 300.0), "z1", CENTRES, 350.0, 20.0) == "z2"
    assert choose_zone((660.0, 300.0), "z1", CENTRES, 350.0, 20.0) == "z2"  # z1 out of range: switch at once


def test_the_hotspot_share_puts_40_percent_of_covered_cars_in_z1(cfg: Config) -> None:
    from pcnme.mobility.grid import RoadGrid

    g = RoadGrid.from_area(cfg.layout.area_m, cfg.cars.grid.block_m)
    h = hotspot_fraction(g, CENTRES, 350.0, 0.4)
    assert 0.1 < h < 0.2  # Q25: about 14 % of the cars stay on z1's roads
    assert hotspot_fraction(g, CENTRES, 350.0, 0.25) == pytest.approx(0.0, abs=0.02)  # no hotspot: uniform


# --- cars that move -----------------------------------------------------------------------------------------


def city_of(cars: dict[str, Line], **kw: object) -> City:
    cfg = fixed_delay_config(**kw)
    paths = dict(cars)
    c = City(
        cfg, "S2", lambda cfg, s, z: Fixed("here"), 2000, n_cars=len(paths), mover=lambda car, hot: paths[car]
    )
    return c


def test_a_car_driving_from_z1_to_z2_is_handed_over_and_gps_follows_it() -> None:
    # 10 m/s from x = 300 to 900 at y = 300: z2 is 20 m closer at x = 610 (31 s); GPS every 100 ms goes to
    # the serving zone, which publishes it to the car's anchors; z1 drops the car on z2's attach (9.6).
    c = city_of({"c000": Line((300.0, 300.0), (900.0, 300.0), 10.0)})
    app = c.apps[0]
    c.clock.run_until(30.9)
    assert app.serving == "z1" and "c000" in c.zones["z1"]._cars
    c.clock.run_until(31.15)
    assert app.serving == "z2"
    c.clock.run_until(31.3)
    assert "c000" not in c.zones["z1"]._cars and "c000" in c.zones["z2"]._cars


def test_no_gps_for_t_leave_ends_the_serving_role() -> None:
    # Into the centre gap: from (300, 300) towards (600, 600) at 10 m/s, 350 m from z1 (its range) at 35.0 s
    # (in rounding, the 35.0 s tick may already be out); z1's last GPS is from 34.9 or 35.0 s, and it drops
    # the car T_leave = 300 ms later, at 35.2 or 35.3 s.
    c = city_of({"c000": Line((300.0, 300.0), (600.0, 600.0), 10.0)})
    c.clock.run_until(35.15)
    assert c.apps[0].serving is None and "c000" in c.zones["z1"]._cars
    c.clock.run_until(35.35)
    assert "c000" not in c.zones["z1"]._cars


def test_a_frame_made_in_a_gap_is_held_then_uploaded_with_the_same_id_or_given_up() -> None:
    # A car parked in the centre gap creates tasks: each hard frame is given up once its budget is below
    # the fastest upload + run + return (9.7.3); soft ones wait until D + L_max, then are given up too.
    c = city_of({"c000": Line((600.0, 600.0), (600.0, 600.0), 10.0)})
    c.run(20.0)
    assert c.records == [] and len(c.car_drops) == len(c.created) > 0
    for d in c.car_drops:
        limit = 0.0 if d.task.cls == "hard" else d.task.l_max
        assert d.at <= d.task.created + d.task.deadline + limit + 0.1 + 1e-9


def test_an_upload_cut_by_a_handover_goes_to_the_next_zone_with_the_same_task_id() -> None:
    # 9.7.2. A T2 clip (72 Mbit, 133 ms of air at 300 m) starts uploading to z1 at 10.0 s; the car (110 m/s)
    # is at x = 611 at the 10.1 s tick: handed to z2 mid-upload. The upload is cut, nothing is left in z1, and
    # z2 anchors the task under the same id.
    c = city_of({"c000": Line((600.0, 300.0), (720.0, 300.0), 110.0, t0=10.0)})
    app = c.apps[0]
    c.clock.run_until(10.0)
    assert app.serving == "z1"
    from pcnme.car.car import Task

    t = c.cfg.task_types["T2"]
    task = Task(
        TaskId("c000", 0, 99),
        "c000",
        "T2",
        t.cls,
        t.deadline["loose"],
        t.l_max["loose"],
        t.input_bits,
        t.result_bits,
        c.clock.now,
    )  # type: ignore[index]
    app.car._tasks[task.task_id] = task
    c._task(app, task)
    c.clock.run_until(10.12)
    assert app.serving == "z2"
    c.clock.run_until(20.0)
    assert ("z1", task.task_id) not in c.zones["z1"].ledger
    (rec,) = (r for r in c.records if r.task.task_id == task.task_id)
    assert rec.anchor == "z2"


def test_a_result_follows_the_car_to_its_new_zone() -> None:
    # C7, 10.3: the car drives right at 10 m/s from x = 500 and is handed to z2 at the 11.1 s tick (x = 611).
    # A T4 made at 11.09 s is anchored and run at z1 (25 ms); its result comes after the handover: z2,
    # subscribed on attach, delivers it, and the ack goes from z2 to z1.
    c = city_of({"c000": Line((500.0, 300.0), (700.0, 300.0), 10.0)})
    app = c.apps[0]
    c.clock.run_until(11.09)
    from pcnme.car.car import Task

    t = c.cfg.task_types["T4"]
    task = Task(
        TaskId("c000", 0, 99),
        "c000",
        "T4",
        t.cls,
        t.deadline["loose"],
        t.l_max["loose"],
        t.input_bits,
        t.result_bits,
        c.clock.now,
    )  # type: ignore[index]
    app.car._tasks[task.task_id] = task
    c._task(app, task)
    c.clock.run_until(12.0)
    (rec,) = (r for r in c.records if r.task.task_id == task.task_id)
    assert (rec.anchor, rec.executor, rec.outcome) == ("z1", "z1", "on_time")
    assert app.serving == "z2" and rec.deliveries >= 1


# --- the whole city -----------------------------------------------------------------------------------------


def test_every_task_closes_once_or_is_given_up_by_the_car(cfg: Config) -> None:
    c = City(cfg, "S2", "greedy", 2000).run(5.0)
    closed = Counter(r.task.task_id for r in c.records)
    assert all(n == 1 for n in closed.values())
    given_up = {d.task.task_id for d in c.car_drops}
    assert set(c.created) == set(closed) | given_up and not (set(closed) & given_up)
    assert all(r.outcome is not None for r in c.records)


def test_the_city_run_is_deterministic(cfg: Config) -> None:
    def digest(seed: int) -> list[tuple[object, ...]]:
        c = City(cfg, "S1", "greedy", seed).run(3.0)
        return sorted((r.task.task_id, r.anchor, r.executor, r.outcome, r.closed_at) for r in c.records)  # type: ignore[type-var]

    assert digest(2000) == digest(2000) and digest(2000) != digest(2001)


def test_an_upload_uses_the_rate_at_the_cars_distance_now() -> None:
    # 4.2: pushed at the car's current Shannon rate: attached at 300 m, it uploads at 10 m from z1.
    from pcnme.car.car import Task
    from pcnme.net.radio import shannon_rate

    c = city_of({"c000": Line((0.0, 300.0), (290.0, 300.0), 29.0)})
    app = c.apps[0]
    c.clock.run_until(10.05)  # parked at x = 290, 10 m from z1, since 10 s
    t = c.cfg.task_types["T1"]
    task = Task(
        TaskId("c000", 0, 99),
        "c000",
        "T1",
        t.cls,
        t.deadline["loose"],
        t.l_max["loose"],
        t.input_bits,
        t.result_bits,
        c.clock.now,
    )  # type: ignore[index]
    app.car._tasks[task.task_id] = task
    c._task(app, task)
    c.clock.run_until(11.0)
    (rec,) = (r for r in c.records if r.task.task_id == task.task_id)
    assert rec.rate == shannon_rate(10.0, c.cfg.radio)


def test_the_anchor_predicts_exit_time_and_next_zone_from_the_cars_gps() -> None:
    # 10.1-10.2: the anchor has an open task of the car, so it gets its GPS (9.6). The car drives east at
    # 10 m/s
    # along y = 300 from x = 300; z1's edge is at x = 650, so T_exit = (650 - x) / 10 s; next: z2.
    from pcnme.car.car import Task

    c = city_of({"c000": Line((300.0, 300.0), (900.0, 300.0), 10.0)})
    app = c.apps[0]
    c.clock.run_until(4.9)
    t = c.cfg.task_types["T2"]  # 720 ms here: the task (and the anchor role) is open past 5.0 s
    task = Task(
        TaskId("c000", 0, 99),
        "c000",
        "T2",
        t.cls,
        t.deadline["loose"],
        t.l_max["loose"],
        t.input_bits,
        t.result_bits,
        c.clock.now,
    )  # type: ignore[index]
    app.car._tasks[task.task_id] = task
    c._task(app, task)
    c.clock.run_until(5.3)  # anchored at z1 once the clip is in (about 5.04 s); the GPS keeps coming
    g = c.zones["z1"].registry["c000"]
    t_exit, nxt = c.zones["z1"].predict("c000")  # type: ignore[misc]
    assert t_exit == pytest.approx((650.0 - g.x) / 10.0) and nxt == "z2"  # z1's edge at x = 650
    assert 350.0 < g.x < 355.0


def test_the_prediction_is_from_the_cars_current_zone_not_the_anchors() -> None:
    # The car drives east past the z1-z2 handover (x = 611 at 31.1 s); z1 anchors a T2 made at 30.5 s (its
    # clip is in by about 30.63 s, before the handover; it runs 720 ms) and gets the car's GPS from z2:
    # T_exit from z2's circle (edge at x = 1250), next zone east of
    # z2 = the nearest other centre to (1250, 300): z4 (694.6 m; z1 950 m).
    from pcnme.car.car import Task

    c = city_of({"c000": Line((300.0, 300.0), (900.0, 300.0), 10.0)})
    app = c.apps[0]
    c.clock.run_until(30.5)
    t = c.cfg.task_types["T2"]
    task = Task(
        TaskId("c000", 0, 98),
        "c000",
        "T2",
        t.cls,
        t.deadline["loose"],
        t.l_max["loose"],
        t.input_bits,
        t.result_bits,
        c.clock.now,
    )  # type: ignore[index]
    app.car._tasks[task.task_id] = task
    c._task(app, task)
    c.clock.run_until(31.3)
    g = c.zones["z1"].registry["c000"]
    assert g.zone_id == "z2"
    t_exit, nxt = c.zones["z1"].predict("c000")  # type: ignore[misc]
    assert t_exit == pytest.approx((1250.0 - g.x) / 10.0) and nxt == "z4"
