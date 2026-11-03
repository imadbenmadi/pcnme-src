"""The zone on a real network (QUESTIONS Q53; final-design 4.3, 9.3, CHANGES C7; PLAN M9.1 step 6).

netem gives the real delays, so a zone with real-network calls sends every request now, with no modelled
pipe delay; the upload timeout starts when the request body has been written (on_sent). The own backhaul
pipe still books the bits, as the estimate behind the agent's inputs and masks (4.3)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from tests.zone.harness import Fixed, city, city_task, fixed_delay_config, one_zone, task_now


class RealCalls:
    real_network = True

    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []

    def post(
        self,
        target: str,
        arrive_at: float | None,
        req: Any,
        back: float,
        on_reply: Callable[[Any], None],
        at_target: Callable[[], None] = lambda: None,
        on_sent: Callable[[], None] = lambda: None,
    ) -> None:
        self.posts.append(
            {"target": target, "arrive_at": arrive_at, "req": req, "reply": on_reply, "sent": on_sent}
        )

    def probe(self, target: str, arrive_at: float, key: Any, back: float, on_answer: Any) -> None:
        pass

    def cancel(self, target: str, arrive_at: float, key: Any) -> None:
        pass


def test_a_cloud_send_leaves_now_and_books_the_own_backhaul() -> None:
    s = one_zone(fixed_delay_config(), Fixed("cloud"))
    s.zone.calls = calls = RealCalls()  # type: ignore[assignment]
    task = task_now(s, "T4")
    s.zone.upload(task)
    s.clock.run_until(0.006)  # t_rx = 3 ms + push
    (p,) = calls.posts
    assert p["target"] == "cloud" and p["arrive_at"] is None
    t_send = s.zone.records[task.task_id].t_send
    push = task.input_bits / s.cfg.links.backhaul_bw
    assert t_send is not None and s.zone.backhaul.wait_at(t_send) == pytest.approx(push)  # booked (4.3)


def test_the_upload_timer_starts_when_the_body_is_written() -> None:
    c = city(fixed_delay_config(), lambda z: Fixed("n1"))
    z1 = c.zones["z1"]
    c.clock.run_until(0.3)  # the neighbours' first snapshots arrive (9.2)
    z1.calls = calls = RealCalls()  # type: ignore[assignment]
    task = city_task(c, "T4")
    z1.upload(task)
    c.clock.run_until(0.306)
    (p,) = calls.posts
    assert p["arrive_at"] is None and p["target"] != "cloud"
    rec = z1.records[task.task_id]
    info = rec.attempts[1]
    cap_timer = info.timer
    assert cap_timer is not None
    c.clock.run_until(0.350)
    assert info.timer is cap_timer  # nothing modelled the send's end: only the deadline cap applies (C7)
    p["sent"]()
    assert info.timer is not cap_timer
    assert info.timer.time == pytest.approx(c.clock.now + info.t_up)
