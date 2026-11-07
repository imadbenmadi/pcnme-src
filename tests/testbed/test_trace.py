"""Hop times on the testbed (src/pcnme/testbed/trace.py; final-design 5.4: analysis only)."""

from __future__ import annotations

import pytest

from pcnme.testbed.trace import Tracer, hop_durations


def test_only_every_nth_task_is_traced() -> None:
    tr = Tracer("z1", 10)
    tr.mark(("c000", 0, 10), 1, "x")
    tr.mark(("c000", 0, 11), 1, "x")
    assert [r[1] for r in tr.rows] == ["c000.0.10"]


def test_an_own_clock_time_becomes_wall_time() -> None:
    tr = Tracer("z1", 1, clock_now=lambda: 5.0)
    tr.mark(("c000", 0, 1), 1, "exec start", at=4.75)  # 0.25 s ago on the zone's clock
    tr.mark(("c000", 0, 1), 1, "now", None)
    assert tr.rows[1][0] - tr.rows[0][0] == pytest.approx(0.25, abs=0.01)


def test_hops_follow_the_attempt_that_reached_the_car() -> None:
    def e(t: float, ev: str, attempt: int = 1, at: str = "z1") -> dict[str, object]:
        return {"t": t, "task": "c000.0.10", "attempt": attempt, "event": ev, "at": at}

    rows = [
        e(0.000, "created", 0, "car"),
        e(0.001, "upload sent", 0, "car"),
        e(0.005, "intake (upload complete)", 0),
        e(0.007, "sent (n1)"),
        e(0.012, "exec start", at="z2"),
        e(0.035, "exec finish", at="z2"),
        e(0.036, "publish called", at="z2"),
        e(0.040, "result at z2", at="z2"),
        e(0.044, "result at z1"),
        e(0.045, "push to the car"),
        e(0.050, "result at the car", 0, "car"),
    ]
    h = hop_durations(rows)
    assert h["run"] == [pytest.approx(0.023)]
    assert h["MQTT: publish -> result at the serving zone"] == [pytest.approx(0.008)]  # z1 pushed, not z2
    assert h["HTTP: push -> result at the car"] == [pytest.approx(0.005)]
    assert h["total: created -> result at the car"] == [pytest.approx(0.050)]
