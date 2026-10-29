"""Placement to neighbours and the cloud, refusals and the one re-decision (PLAN M3.5; final-design 6C, 9.1,
9.2, 12.5; QUESTIONS Q22). Every delay at its mean and every run at its mean, so the numbers are by hand."""

from __future__ import annotations

import pytest

from pcnme.energy import EnergyModel
from pcnme.node.executor import Refused
from tests.zone.harness import City, Fixed, Seq, city, city_task, fixed_delay_config

PUSH = 1.2e6 / 539.084e6  # a 150 KB frame from 300 m: 2.226 ms


def two(placement: object, **overrides: object) -> City:
    cfg = fixed_delay_config(**overrides)
    seq = {"z1": placement, "z2": Fixed(None)}
    c = city(cfg, lambda z: seq[z], zone_ids=("z1", "z2"))
    c.clock.run_until(0.1)  # every state is in (the cloud's at 66.5 ms)
    return c


def refuse(c: City, zone: str, reason: str = "full") -> None:
    c.zones[zone].handle = lambda req: (Refused(reason), c.zones[zone].snapshot())  # type: ignore[method-assign]


def test_a_t1_run_at_a_neighbour_by_hand() -> None:
    """6C.9, 4.3, 10.3. Created at 0.1 s; t_rx = +5.226 ms, d_A = 0.2 s.
    T̂_in = 0 + 1.2e6 / 1e9 + 3 ms = 4.2 ms; T̂_ret = 3 + 1 + 3 + 3 = 10 ms; B_k = 100 - 5.226 - 4.2 - 10 =
    80.574 ms. The input reaches z2 4.2 ms after the send; it runs 8 ms at once; z2 publishes the result:
    3 + 1 + 3 ms to z1 (the serving zone), + 3 ms radio: L = 5.226 + 4.2 + 8 + 10 = 27.426 ms."""
    c = two(Fixed("n1"))
    c.zones["z1"].upload(city_task(c, "T1"))
    c.clock.run_until(1.0)
    (rec,) = c.closed
    assert (rec.option, rec.executor, rec.reply, rec.outcome) == ("n1", "z2", "accepted", "on_time")
    upload = rec.t_rx - rec.task.created  # 5.226 ms
    assert upload == pytest.approx(0.003 + PUSH, rel=1e-4)
    assert rec.budget == pytest.approx(0.100 - upload - 0.0042 - 0.010, rel=1e-12)  # 80.574 ms
    assert rec.parts["input"] == pytest.approx(0.0042, rel=1e-9)
    assert rec.report is not None and rec.report.start == pytest.approx(rec.t_at_executor)
    assert rec.confirm is not None and rec.confirm.latency == pytest.approx(0.027426, rel=1e-3)
    assert sum(rec.parts.values()) == pytest.approx(rec.confirm.latency, abs=1e-12)
    assert c.zones["z1"].store.wan.last_sample is None  # only cloud uploads sample ŵ_WAN (4.3)


def test_the_neighbours_reply_carries_its_snapshot_and_clears_the_pending_send() -> None:
    # R1 (9.2): pending from the send until the reply, 4.2 + 3 ms later; the reply's version is held.
    c = two(Fixed("n1"))
    z1 = c.zones["z1"]
    z1.upload(city_task(c, "T1"))
    c.clock.run_until(0.1 + 0.003 + PUSH + 0.001)
    assert len(z1.store.pending_on("z2", "gpu")) == 1
    before = z1.store.snapshot("z2").version  # type: ignore[union-attr]
    c.clock.run_until(0.1 + 0.003 + PUSH + 0.0042 + 0.003 + 1e-6)
    assert z1.store.pending_on("z2", "gpu") == ()
    assert z1.store.snapshot("z2").version > before  # type: ignore[union-attr]


def test_a_refusal_gets_one_re_decision_as_a_new_attempt() -> None:
    # 9.1: refused at once; the anchor re-decides once (with fresh state from the reply). Each placement is
    # its own attempt (5.5): attempt 1 to z2, attempt 2 here. The refused one cost its forwarding (12.5).
    placement = Seq("n1", "here")
    c = two(placement)
    refuse(c, "z2")
    c.zones["z1"].upload(city_task(c, "T1"))
    c.clock.run_until(1.0)
    (rec,) = c.closed
    assert placement.calls == 2 and rec.outcome == "on_time"
    assert [(e.attempt, e.option, e.executor, e.reply) for e in rec.earlier] == [(1, "n1", "z2", "full")]
    assert rec.report is not None and rec.report.key.attempt == 2
    assert EnergyModel(c.cfg).earlier(rec) == pytest.approx(EnergyModel(c.cfg).forwarding("neighbour", 1.2e6))
    # the decision waited for the refusal: send 1 at t_rx, refusal back 4.2 + 3 ms later, then here
    assert rec.parts["before_send"] == pytest.approx(0.0072, rel=1e-9)
    assert rec.confirm is not None and sum(rec.parts.values()) == pytest.approx(
        rec.confirm.latency, abs=1e-12
    )


def test_a_hard_task_refused_twice_is_dropped() -> None:
    placement = Seq("n1", "n1")
    c = two(placement)
    refuse(c, "z2", "late")
    c.zones["z1"].upload(city_task(c, "T1"))
    c.clock.run_until(1.0)
    (rec,) = c.closed
    assert placement.calls == 2 and rec.outcome == "drop_refused_late" and len(rec.earlier) == 1


def test_a_hard_task_with_no_slack_at_the_re_decision_is_dropped() -> None:
    placement = Seq("n1", None)  # every option masked at the re-decision
    c = two(placement)
    refuse(c, "z2")
    c.zones["z1"].upload(city_task(c, "T1"))
    c.clock.run_until(1.0)
    assert c.closed[0].outcome == "drop_refused_full" and placement.calls == 2


def test_a_soft_task_refused_twice_goes_to_the_cloud() -> None:
    # 9.1: soft: cloud, unless the cloud cannot finish by deadline + L_max either. A rule, not a decision.
    placement = Seq("n1", "n1")
    c = two(placement)
    refuse(c, "z2")
    c.zones["z1"].upload(city_task(c, "T4"))
    c.clock.run_until(3.0)
    (rec,) = c.closed
    assert placement.calls == 2
    assert (rec.option, rec.executor, rec.outcome) == ("cloud", "cloud", "on_time")
    assert (
        [e.attempt for e in rec.earlier] == [1, 2] and rec.report is not None and rec.report.key.attempt == 3
    )


def test_a_soft_task_with_no_slack_goes_to_the_cloud_at_once() -> None:
    placement = Seq("n1", None)
    c = two(placement)
    refuse(c, "z2")
    c.zones["z1"].upload(city_task(c, "T4"))
    c.clock.run_until(3.0)
    assert (c.closed[0].option, c.closed[0].outcome) == ("cloud", "on_time")


def test_a_soft_task_the_cloud_cannot_finish_is_dropped_instead() -> None:
    # T4 with D = L_max = 60 ms: the cloud path takes 67.8 + 12 + 69.5 ms > time left + L_max (about 115 ms).
    placement = Seq("n1", "n1")
    c = two(placement, task_types__T4__deadline__loose=0.060)
    refuse(c, "z2")
    c.zones["z1"].upload(city_task(c, "T4"))
    c.clock.run_until(3.0)
    assert c.closed[0].outcome == "drop_refused_full" and c.closed[0].executor == "z2"


def test_when_the_cloud_itself_refused_a_soft_task_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    # Engineering choice (Q22): the fallback is the cloud; if the cloud just refused, it is not asked again.
    placement = Seq("cloud", "cloud")
    c = two(placement)
    monkeypatch.setattr(c.cloud, "handle", lambda req: (Refused("late"), c.cloud.snapshot()))
    c.zones["z1"].upload(city_task(c, "T4"))
    c.clock.run_until(3.0)
    assert c.closed[0].outcome == "drop_refused_late" and placement.calls == 2
    assert len(c.closed[0].earlier) == 1  # attempt 1 and its re-decision, attempt 2: no third send


def test_a_refused_boulder_is_dropped_without_a_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    # A boulder never had an agent decision (7.1), so there is none to redo (Q22).
    placement = Seq("here")
    c = two(placement)
    monkeypatch.setattr(c.cloud, "handle", lambda req: (Refused("late"), c.cloud.snapshot()))
    c.zones["z1"]._mode = "tight"
    c.zones["z1"].upload(city_task(c, "T2", mode="tight"))
    c.clock.run_until(3.0)
    assert (c.closed[0].route, c.closed[0].outcome, placement.calls) == ("cloud", "drop_refused_late", 0)


def test_two_sends_in_a_row_see_each_other_as_pending() -> None:
    # R1: the second decision, before the first reply, counts the first T4 (25 ms) on z2's GPU queue.
    c = two(Fixed("n1"))
    z1 = c.zones["z1"]
    z1.upload(city_task(c, "T4", seq=1))
    c.clock.run_until(0.1 + 0.003 + PUSH + 1e-6)  # the first is sent
    ctx = z1._context(city_task(c, "T4", seq=2), c.clock.now, c.clock.now + 1.0)
    assert [p.mean for p in ctx.neighbours[0].pending["gpu"]] == [0.025]
