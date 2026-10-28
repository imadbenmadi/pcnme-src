"""Budgets, execution, delivery, ack and close in one zone (PLAN M2.6; final-design 6A-6E, 10.3, 8.6, 9.5;
QUESTIONS Q15, Q16). Delays fixed at their means and run times at theirs, so every number is by hand."""

from __future__ import annotations

import pytest

from pcnme.ids import TaskId
from pcnme.zone.zone import Zone
from tests.zone.harness import Fixed, fixed_delay_config, one_zone, task_now

PUSH = 1.2e6 / 539.084e6  # a 150 KB frame from 300 m: 2.226 ms (M2.3)


def test_a_t1_run_here_by_hand() -> None:
    """One T1 (hard, D = 100 ms) from a car 300 m away, empty system, run here.

    upload: push 1.2e6 / R(300 m) = 2.226 ms + radio delay 3 ms -> t_rx = 5.226 ms
    d_A = t_rx + B_car - (W_radio + push + mean radio delay) = 5.226 + 100 - (0 + 2.226 + 3) = 100 ms
    budget here = d_A - T̂_ret,fog - now = 100 - 10 - 5.226 = 84.774 ms (6C.9: d_k = d_A - T̂_ret = 90 ms)
    run on the F-GPU GPU slot: 8 ms, 5.226 -> 13.226 ms
    return: executor -> broker 3 + broker 1 + broker -> zone 3 + radio down 3 = 10 ms -> at the car 23.226 ms
    L = 23.226 ms <= D: on time
    """
    s = one_zone(fixed_delay_config(), Fixed("here"))
    task = task_now(s, "T1")
    s.zone.upload(task)
    s.clock.run_until(1.0)
    (rec,) = s.closed
    assert rec.t_rx == pytest.approx(0.003 + PUSH, rel=1e-4)
    assert rec.d_a == pytest.approx(0.100, rel=1e-9)
    assert rec.budget == pytest.approx(0.100 - 0.010 - rec.t_rx, rel=1e-9)  # 84.774 ms
    assert rec.report is not None and rec.report.slot == "gpu"
    assert (rec.report.start, rec.report.finish) == (pytest.approx(rec.t_rx), pytest.approx(rec.t_rx + 0.008))
    assert rec.result_at_car == pytest.approx(rec.t_rx + 0.008 + 0.010, rel=1e-9)
    assert rec.confirm is not None and rec.confirm.latency == pytest.approx(0.023226, rel=1e-3)
    assert rec.outcome == "on_time"


def test_a_t4_run_in_the_cloud_by_hand() -> None:
    """One T4 (soft, D = 1 s) from 300 m, empty system, sent to the cloud.

    t_rx = 5.226 ms, d_A = 1 s (as above).
    T̂_in,cloud = 0 + 0 + 1.2e6 / 514.85e6 (2.331 ms) + 3 + 62.5 = 67.831 ms
    T̂_ret,cloud = 62.5 + 1 + 3 + 3 = 69.5 ms
    budget = d_A - now - T̂_in - T̂_ret = 1000 - 5.226 - 67.831 - 69.5 = 857.443 ms
    transfer: backhaul wait 0, d_bh 3 -> enters the WAN at 8.226 ms, push 2.331 ms, d_WAN 62.5 -> 73.057 ms
    (= now + T̂_in exactly, since the delays are at their means), so d_k = 73.057 + 857.443 = 930.5 ms
    run on a cloud GPU slot: 12 ms -> 85.057 ms; return 62.5 + 1 + 3 + 3 = 69.5 ms -> at the car 154.557 ms
    L = 154.557 ms <= D: on time
    """
    s = one_zone(fixed_delay_config(), Fixed("cloud"))
    task = task_now(s, "T4")
    s.zone.upload(task)
    s.clock.run_until(5.0)
    (rec,) = s.closed
    assert rec.transfer is not None
    t_in = 1.2e6 / s.cfg.derived.b_wan + 0.003 + 0.0625
    assert rec.budget == pytest.approx(1.0 - rec.t_rx - t_in - 0.0695, rel=1e-9)  # 857.443 ms
    assert rec.transfer.delivery == pytest.approx(
        rec.t_rx + 0.003 + 1.2e6 / s.cfg.derived.b_wan + 0.0625, rel=1e-9
    )
    assert rec.report is not None and rec.report.busy == pytest.approx(0.012)
    assert rec.result_at_car == pytest.approx(rec.transfer.delivery + 0.012 + 0.0695, rel=1e-9)
    assert rec.confirm is not None and rec.confirm.latency == pytest.approx(0.154557, rel=1e-3)
    assert rec.outcome == "on_time"


def test_a_hard_t1_sent_to_the_cloud_is_refused_late_and_dropped() -> None:
    # Its cloud budget is 100 - 5.226 - 67.831 - 69.5 = -42.6 ms: at the cloud d_k = 73.057 - 42.557 = 30.5 ms
    # is already past, so the executor refuses it "late". M3 PLACEHOLDER (M3.5): no re-decision, dropped.
    s = one_zone(fixed_delay_config(), Fixed("cloud"))
    s.zone.upload(task_now(s, "T1"))
    s.clock.run_until(1.0)
    (rec,) = s.closed
    assert (rec.reply, rec.outcome) == ("late", "drop_refused_late")


def test_a_boulder_goes_to_the_cloud_with_no_decision() -> None:
    # T2 with tight deadlines is a boulder (7.1): straight to the cloud, the placement is never asked.
    placement = Fixed("here")
    s = one_zone(fixed_delay_config(), placement, mode="tight")
    s.zone.upload(task_now(s, "T2", mode="tight"))
    s.clock.run_until(5.0)
    (rec,) = s.closed
    assert (rec.route, rec.option, placement.calls) == ("cloud", "cloud", 0)


def test_a_boulder_the_cloud_cannot_make_is_dropped_at_intake() -> None:
    placement = Fixed("here")
    s = one_zone(fixed_delay_config(task_types__T1__deadline__tight=0.005), placement, mode="tight")
    s.zone.upload(task_now(s, "T1", mode="tight"))
    s.clock.run_until(1.0)
    (rec,) = s.closed
    assert (rec.route, rec.outcome, placement.calls, rec.option) == ("drop", "drop_intake", 0, None)


def test_every_option_masked_drops_the_task_at_once() -> None:
    s = one_zone(fixed_delay_config(), Fixed(None))
    s.zone.upload(task_now(s, "T1"))
    s.clock.run_until(1.0)
    (rec,) = s.closed
    assert (rec.outcome, rec.closed_at) == ("drop_masked", pytest.approx(rec.t_rx))


def test_skips_and_aborts_reach_the_anchor_as_status_messages() -> None:
    """8.6, 10.3: the executor reports a drop at once; the status message crosses executor -> broker -> anchor
    (3 + 1 + 3 = 7 ms) and the anchor closes the task as a drop.

    An F-CPU zone (2 CPU slots) where every run takes 15 x its mean, so a T4 runs 3 s instead of 200 ms. Two
    T4 take both slots; each passes d_k + L_max = 0.99 + 1.0 = 1.99 s while running, so it is aborted then and
    its slot stays busy for its longest step (100 ms), free at about 2.09 s. A third T4 waits in the queue; at
    2.09 s, 2.09 + Ê^low (150 ms) > 1.99 s, so it is skipped.
    """
    from pcnme.node.timetable import Deterministic

    s = one_zone(
        fixed_delay_config(), Fixed("here"), hw_name="F-CPU", shape=lambda r: Deterministic(15 * r.mean)
    )
    for seq in (1, 2, 3):
        s.zone.upload(task_now(s, "T4", seq=seq))
    s.clock.run_until(10.0)
    by_seq = {r.task.task_id.seq: r for r in s.closed}
    for seq in (1, 2):
        rec = by_seq[seq]
        assert rec.report is not None and rec.report.outcome == "aborted"
        assert rec.outcome == "drop_aborted"
        # aborted at d_k + L_max = (d_A - T̂_ret,fog) + L_max; the status leaves then, not after the tail
        assert rec.closed_at == pytest.approx(rec.d_a - 0.010 + 1.0 + 0.007, abs=1e-9)
    skipped = by_seq[3]
    assert skipped.report is not None and skipped.report.outcome == "skipped"
    assert skipped.outcome == "drop_skipped"
    assert skipped.closed_at == pytest.approx(skipped.report.finish + 0.007)


@pytest.mark.parametrize(
    ("cls", "latency", "expected"),
    [
        ("hard", 0.100, "on_time"),
        ("hard", 0.1001, "drop_late"),
        ("soft", 1.0, "on_time"),
        ("soft", 1.5, "late"),
        ("soft", 2.0, "late"),
        ("soft", 2.0001, "drop_late"),
    ],
)
def test_the_outcome_comes_from_the_cars_latency(cls: str, latency: float, expected: str) -> None:
    # QUESTIONS Q15: hard: L <= D on time, else a drop; soft: L <= D on time, L <= D + L_max late, else a
    # drop.
    from pcnme.car.car import Task

    d = 0.100 if cls == "hard" else 1.0
    task = Task(TaskId("c01", 0, 1), "c01", "T1" if cls == "hard" else "T4", cls, d, d, 1.0, 1.0, 0.0)  # type: ignore[arg-type]
    assert Zone._outcome(task, latency) == expected


def test_each_task_closes_exactly_once_under_load() -> None:
    # The car's own Poisson arrivals at S2's rate for 20 s, everything here: every created task closes once.
    s = one_zone(fixed_delay_config(), Fixed("here"))
    s.car.start(until=20.0)
    s.clock.run_until(25.0)
    ids = [r.task.task_id for r in s.closed]
    assert len(ids) == len(set(ids)) == len(s.zone.records) > 20
    assert s.zone.anchor.open_count == 0
