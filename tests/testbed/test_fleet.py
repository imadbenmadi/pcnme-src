"""The car container (PLAN M9.1 step 6; final-design 15.3, 15.4, 9.7, 10): the fast-mode cars and mobility
over HTTP: attach, GPS every period, uploads with B_car and R_car, cuts on a handover, held frames."""

from __future__ import annotations

from typing import Any

from pcnme.car.car import Task
from pcnme.ids import TaskId
from pcnme.sim.clock import EventClock
from pcnme.testbed.car import CarFleet
from pcnme.testbed.messages import Attach, Gps, Upload, UploadReply
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
    """The fake network: records every request; uploads stay open until the test answers them."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str, Any]] = []
        self.open: dict[Any, Any] = {}

    def __call__(self, zone: str, method: str, path: str, body: Any, on_reply: Any) -> None:
        self.sent.append((zone, method, path, body))
        if isinstance(body, Upload):
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
    assert kinds[0] == "Attach" and "Gps" in kinds
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
    assert all(not isinstance(b, Gps) or z == "z2" for z, _, _, b in net.sent[-2:])


def test_started_cars_make_tasks_and_upload_them() -> None:
    z1 = fixed_delay_config().layout.zone_positions_m["z1"]
    f, clock, net, _ = fleet([z1])
    f.start(20.0)
    clock.run_until(20.0)
    assert any(isinstance(b, Upload) for _, _, _, b in net.sent)  # lambda_car > 0 (18)
