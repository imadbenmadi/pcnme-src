"""The car container (PLAN M9.1 step 6; final-design 15.3, 15.4, 9.7, 10): the fast-mode cars and mobility
over HTTP: attach, GPS every period, uploads with B_car and R_car, cuts on a handover, held frames."""

from __future__ import annotations

from typing import Any

from pcnme.car.car import Task
from pcnme.ids import TaskId
from pcnme.sim.clock import EventClock
from pcnme.testbed.car import CarFleet
from pcnme.testbed.messages import Attach, GpsBatch, Upload, UploadReply
from tests.zone.harness import fixed_delay_config


class Mover:
    def __init__(self, path: list[tuple[float, float]]) -> None:
        self.path = path
        self.i = 0

    def position(self, t: float | None = None) -> tuple[float, float]:
        return self.path[min(self.i, len(self.path) - 1)]

    def velocity(self, t: float | None = None) -> tuple[float, float]:
        return (10.0, 0.0)


class Net:
    """The fake network: records every request; uploads stay open until the test answers them; attaches are
    answered 200 unless `lose_attaches` (a zone not up yet: the request fails)."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str, Any]] = []
        self.open: dict[Any, Any] = {}
        self.lose_attaches = 0
        self.unknown: list[str] = []

    def __call__(self, zone: str, method: str, path: str, body: Any, on_reply: Any) -> None:
        self.sent.append((zone, method, path, body))
        if isinstance(body, Attach):
            if self.lose_attaches:
                self.lose_attaches -= 1  # lost: no reply at all (connection refused)
            else:
                on_reply(200, None)
        elif isinstance(body, GpsBatch):
            on_reply(200, self.unknown)  # the cars this zone does not serve (QUESTIONS Q61)
            self.unknown = []
        elif isinstance(body, Upload):
            self.open[body.task.task_id] = on_reply
        elif method == "DELETE":
            on_reply(200 if self.open.pop(body, None) else 409, None)  # still open: the cut works (9.7)


def fleet(path: list[tuple[float, float]]) -> tuple[CarFleet, EventClock, Net, Mover]:
    cfg = fixed_delay_config()
    clock = EventClock()
    net = Net()
    mover = Mover(path)
    f = CarFleet(cfg, "S2", 2000, clock, ["c000"], net, "http://car-1:8000", mover=lambda c, h: mover)
    return f, clock, net, mover


def make_task(f: CarFleet, clock: EventClock, kind: str, seq: int) -> Task:
    t = f.cfg.task_types[kind]
    tid = TaskId("c000", 0, seq)
    task = Task(
        tid,
        "c000",
        kind,
        t.cls,
        t.deadline["loose"],
        t.l_max["loose"],
        t.input_bits,
        t.result_bits,
        clock.now,
    )
    f.apps[0].car._tasks[tid] = task  # the car knows its own task (as Car._create does)
    return task


def test_a_car_attaches_sends_gps_and_uploads_with_its_budget_and_rate() -> None:
    z1 = fixed_delay_config().layout.zone_positions_m["z1"]
    f, clock, net, _ = fleet([z1])
    clock.run_until(0.25)
    kinds = [type(b).__name__ for _, _, _, b in net.sent]
    assert kinds[0] == "Attach" and "GpsBatch" in kinds
    att = net.sent[0][3]
    assert isinstance(att, Attach) and att.url == "http://car-1:8000" and net.sent[0][0] == "z1"
    app = f.apps[0]
    task = make_task(f, clock, "T1", 99)
    f.task(app, task)
    up = next(b for _, _, _, b in net.sent if isinstance(b, Upload) and b.task == task)
    assert up.b_car == app.car.budget(task, clock.now) and up.rate > 0
    net.open.pop(task.task_id)(200, UploadReply(task.task_id, "z1"))
    assert task.task_id not in app.uploading


def test_a_handover_cuts_the_open_upload_and_re_uploads_it_to_the_new_zone() -> None:
    pos = fixed_delay_config().layout.zone_positions_m
    f, clock, net, mover = fleet([pos["z1"], pos["z2"]])
    clock.run_until(0.15)
    app = f.apps[0]
    task = make_task(f, clock, "T4", 7)
    f.task(app, task)
    mover.i = 1  # the car is now in z2's circle
    clock.run_until(0.35)
    sent = [(z, m, type(b).__name__) for z, m, _, b in net.sent]
    assert ("z1", "DELETE", "TaskId") in sent
    ups = [z for z, _, _, b in net.sent if isinstance(b, Upload) and b.task == task]
    assert ups == ["z1", "z2"]  # the same task id to the next zone (9.7)
    assert any(isinstance(b, Attach) and z == "z2" for z, _, _, b in net.sent)
    assert all(not isinstance(b, GpsBatch) or z == "z2" for z, _, _, b in net.sent[-2:])


def test_one_gps_request_per_zone_and_period_carries_every_car_there() -> None:
    """Transport only: the cars of one container share the request; each fix is still its own (5.2, 10)."""
    z1 = fixed_delay_config().layout.zone_positions_m["z1"]
    cfg = fixed_delay_config()
    clock = EventClock()
    net = Net()
    f = CarFleet(cfg, "S2", 3000, clock, ["c000", "c001", "c002"], net, "u", mover=lambda c, h: Mover([z1]))
    clock.run_until(0.05)
    batches = [b for z, _, _, b in net.sent if isinstance(b, GpsBatch)]
    assert len(batches) == 1 and sorted(g.car_id for g in batches[0].fixes) == ["c000", "c001", "c002"]
    assert f.apps[0].serving == "z1"


def test_started_cars_make_tasks_and_upload_them() -> None:
    z1 = fixed_delay_config().layout.zone_positions_m["z1"]
    f, clock, net, _ = fleet([z1])
    f.start(20.0)
    clock.run_until(20.0)
    assert any(isinstance(b, Upload) for _, _, _, b in net.sent)  # lambda_car > 0 (18)


def test_a_smaller_fleet_keeps_its_ids_inside_the_fleet() -> None:
    from pcnme.testbed.car import car_ids

    assert car_ids("0-199/10:3")[:2] == ["c003", "c013"] and len(car_ids("0-199/10:3")) == 20
    assert car_ids("0-199/10:0", 20) == [f"c{i:03d}" for i in range(20)]  # one container: c000..c019
    assert car_ids("0-199/10:1", 40)[:2] == ["c001", "c003"] and car_ids("0-199/10:5", 40) == []


def test_a_lost_attach_is_sent_again_until_the_zone_confirms_it() -> None:
    """9.6: the zone must know the car to push its results (PLAN Q1: zones still starting refused the first
    attach, and every result of those cars was lost)."""
    _, clock, net, _ = fleet([(300.0, 300.0)])
    net.lose_attaches = 2
    clock.run_until(1.0)  # 10 GPS periods
    attaches = [s for s in net.sent if s[2] == "/cars"]
    assert len(attaches) == 3  # two lost, the third confirmed
    assert all(a[0] == "z1" and isinstance(a[3], Attach) for a in attaches)


def test_a_confirmed_attach_is_not_sent_again() -> None:
    _, clock, net, _ = fleet([(300.0, 300.0)])
    clock.run_until(1.0)
    assert len([s for s in net.sent if s[2] == "/cars"]) == 1


def test_a_car_its_zone_no_longer_serves_attaches_again() -> None:
    """QUESTIONS Q61: a zone that dropped the car (T_leave fired on a late GPS, 9.6) names it in its reply to
    the GPS; the car attaches again, so its results reach it again."""
    _, clock, net, _ = fleet([(300.0, 300.0)])
    clock.run_until(0.25)
    assert len([s for s in net.sent if s[2] == "/cars"]) == 1
    net.unknown = ["c000"]
    clock.run_until(0.5)
    attaches = [s for s in net.sent if s[2] == "/cars"]
    assert len(attaches) == 2 and attaches[1][0] == "z1"
