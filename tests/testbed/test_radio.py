"""The radio uplink on the testbed (PLAN M9.1 step 5; final-design 15.3, 5.3 POST /tasks): the car sends R_car
and B_car with its upload, the zone's software pipe (the fast-mode code) shares the airtime, and a remote car
confirms later through its HTTP reply."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from pcnme.car.car import Confirm, Result
from pcnme.testbed.radio import RemoteCar, Upload, UploadReply, software_radio
from tests.zone.harness import Fixed, fixed_delay_config, one_zone, task_now


def test_the_upload_brings_its_own_b_car_and_rate() -> None:
    s = one_zone(fixed_delay_config(), Fixed("here"))
    task = task_now(s, "T1")
    rate = 400e6
    s.zone.upload(task, rate=rate, b_car=0.080)
    s.clock.run_until(1.0)
    (rec,) = s.closed
    push = task.input_bits / rate
    assert rec.t_rx == pytest.approx(push + 0.003, rel=1e-6)
    assert rec.d_a == pytest.approx(rec.t_rx + 0.080 - (push + 0.003), rel=1e-9)  # 6A.3 with the sent B_car


def test_a_car_that_confirms_later_still_closes_its_task_on_time() -> None:
    s = one_zone(fixed_delay_config(), Fixed("here"))
    later: list[tuple[Result, float]] = []

    class Late:
        car_id = "c01"

        def budget(self, task: Any, now: float) -> float:
            return s.car.budget(task, now)

        def receive(self, result: Result, now: float) -> Confirm | None:
            later.append((result, now))
            return None  # the confirm comes back with the HTTP reply

    s.zone._cars["c01"] = (Late(), s.zone._cars["c01"][1])
    task = task_now(s, "T1")
    s.zone.upload(task)
    s.clock.run_until(0.05)
    assert s.closed == [] and len(later) == 1
    result, at = later[0]
    confirm = s.car.receive(result, at)
    assert confirm is not None
    s.zone.car_confirmed("c01", result, confirm)
    s.clock.run_until(1.0)
    (rec,) = s.closed
    assert rec.outcome == "on_time"


def test_software_radio_leaves_the_base_delays_to_netem() -> None:
    cfg, netem = software_radio(fixed_delay_config())
    assert cfg.radio.uplink_delay == (0.0, 0.0) and cfg.radio.downlink_delay == (0.0, 0.0)
    assert netem == {"uplink": (0.003, 0.003), "downlink": (0.003, 0.003)}


def test_a_remote_car_posts_the_result_and_hands_the_confirm_to_the_zone_loop() -> None:
    from pcnme.ids import TaskId
    from pcnme.testbed.clock import RealClock

    clock = RealClock()
    got: list[tuple[str, Result, Confirm, str]] = []

    class Zone:
        def car_confirmed(self, car_id: str, result: Result, confirm: Confirm) -> None:
            got.append((car_id, result, confirm, threading.current_thread().name))

    confirm = Confirm(TaskId("c01", 0, 1), "z1", 1, 0.02)

    import httpx

    from pcnme.testbed.codec import decode, encode
    from pcnme.testbed.netloop import NetLoop

    def car_app(request: httpx.Request) -> httpx.Response:
        assert (request.method, str(request.url)) == ("POST", "http://car-1/results")
        assert isinstance(decode(request.content), Result)
        return httpx.Response(200, content=encode(confirm))

    net = NetLoop(clock, transport=httpx.MockTransport(car_app))
    car = RemoteCar("c01", "http://car-1", Zone(), net)
    res = Result(TaskId("c01", 0, 1), "z1", 1)
    assert car.receive(res, 0.0) is None
    clock.run_for(0.3)
    assert got == [("c01", res, confirm, threading.current_thread().name)]


def test_an_upload_and_its_reply_travel_as_registered_payloads() -> None:
    from pcnme.ids import TaskId
    from pcnme.testbed.codec import decode, encode

    s = one_zone(fixed_delay_config(), Fixed("here"))
    up = Upload(task_now(s, "T1"), b_car=0.09, rate=5e8)
    assert decode(encode(up)) == up
    rep = UploadReply(TaskId("c01", 0, 1), "z1")
    assert decode(encode(rep)) == rep


def test_an_upload_that_brings_b_car_and_rate_needs_no_registry_entry() -> None:
    """Testbed: the car's attach and its upload travel on separate requests; either may arrive first."""
    s = one_zone(fixed_delay_config(), Fixed("here"))
    s.zone.detach("c01")
    task = task_now(s, "T1")
    s.zone.upload(task, rate=4e8, b_car=0.08)
    s.clock.run_until(1.0)
    assert [r.task.task_id for r in s.closed] == [task.task_id]
