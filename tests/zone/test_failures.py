"""Loss detection, retry and cancel (PLAN M3.6; final-design 9.1, 9.3, 9.4, 9.5.5; QUESTIONS Q23). Two zones,
every delay at its mean and every run at its mean (deterministic), so the times are by hand. The task is
created at 0.1 s; the T1 / T4 upload ends at t_rx = 0.1 + 5.226 ms."""

from __future__ import annotations

import pytest

from tests.zone.harness import City, Fixed, Seq, city, city_task, fixed_delay_config

PUSH = 1.2e6 / 539.1e6  # 2.226 ms
T_RX = 0.1 + 0.003 + PUSH


def two(placement: object, **overrides: object) -> City:
    seq = {"z1": placement, "z2": Fixed(None)}
    c = city(fixed_delay_config(**overrides), lambda z: seq[z], zone_ids=("z1", "z2"))
    c.clock.run_until(0.1)
    return c


def test_a_lost_reply_times_out_and_the_first_result_still_wins() -> None:
    """9.1, 9.3, C7. The accept reply from z2 is lost. The own send ends at t_rx + 4.2 ms; T_up = 2 x
    (4.2 + 3) = 14.4 ms from then: at t_rx + 18.6 the send counts as not placed: the pending send ends, z2
    gets a cancel (it already finished: ignored), and the anchor re-decides (here). z2's result (done at
    t_rx + 4.2 + 8 = 12.2 ms) reaches z1 at 19.2 ms: the first result wins and the here attempt is cancelled.
    L = 5.226 + 22.2 = 27.426 ms."""
    placement = Seq("n1", "here")
    c = two(placement)
    z1 = c.zones["z1"]
    z1.lose_reply = lambda key: key.attempt == 1
    z1.upload(city_task(c, "T1"))
    c.clock.run_until(T_RX + 0.0186 - 1e-6)
    assert placement.calls == 1 and len(z1.store.pending_on("z2", "gpu")) == 1
    c.clock.run_until(T_RX + 0.0186 + 1e-6)
    assert placement.calls == 2 and z1.store.pending_on("z2", "gpu") == ()
    c.clock.run_until(3.0)
    (rec,) = c.closed
    assert rec.outcome == "on_time" and rec.result_attempt == 1 and rec.confirm is not None
    assert rec.confirm.attempt == 1  # the car kept z2's copy
    here = z1.executor.status(next(k for k in [z1_key(rec, 2)]))
    assert here == "cancelled"  # 9.4.3: the attempt still alive was cancelled
    assert rec.confirm.latency == pytest.approx(0.027426, rel=1e-3)
    assert sum(rec.parts.values()) == pytest.approx(rec.confirm.latency, abs=1e-12)


def z1_key(rec, attempt):  # type: ignore[no-untyped-def]
    from pcnme.ids import AttemptKey

    return AttemptKey(rec.task.task_id, "z1", attempt)


def test_an_executor_that_dies_triggers_a_retry_at_once() -> None:
    """9.1: its last will -> every task placed there is retried if the retry mask allows. A T4 (25 ms) goes
    to z2 and is accepted; z2 dies 10 ms after t_rx; its will reaches z1 3 ms later: the retry (here)
    starts then."""
    placement = Seq("n1", "here")
    c = two(placement)
    c.zones["z1"].upload(city_task(c, "T4"))
    c.clock.run_until(T_RX + 0.010)
    c.bus.kill("z2")
    c.clock.run_until(T_RX + 0.013 + 1e-6)
    assert placement.calls == 2
    c.clock.run_until(3.0)
    (rec,) = c.closed
    assert rec.outcome == "on_time" and rec.retries == 1 and rec.executor == "z1"
    assert [(e.attempt, e.executor, e.reply) for e in rec.earlier] == [(1, "z2", "lost")]


def test_a_silently_lost_result_is_found_by_the_watchdog_and_the_probe() -> None:
    """9.3. z2 runs a T4 and its result message is lost. Watchdog from the accept (t_rx + 7.2 ms): t_wd =
    t_acc - 3 + Ŵ 0 + Ê^high 32.5 + δ̂_pub 7 + m_wd 20 = t_rx + 63.7 ms. The probe (3 ms there and back)
    says done; wait δ̂_pub + m_wd = 27 ms; still no result: lost (Q23) -> retry here at t_rx + 96.7 ms."""
    placement = Seq("n1", "here")
    c = two(placement)
    c.bus.faults.drop = lambda topic, key, sub: topic == "car/c01/result" and key[4] == 1
    c.zones["z1"].upload(city_task(c, "T4"))
    c.clock.run_until(T_RX + 0.0637 - 1e-6)
    assert c.zones["z1"].records[next(iter(c.zones["z1"].records))].attempts[1].probes == 0
    c.clock.run_until(T_RX + 0.0637 + 1e-6)
    rec = next(iter(c.zones["z1"].records.values()))
    assert rec.attempts[1].probes == 1
    c.clock.run_until(T_RX + 0.0967 - 1e-6)
    assert placement.calls == 1
    c.clock.run_until(T_RX + 0.0967 + 1e-6)
    assert placement.calls == 2 and rec.retries == 1
    c.clock.run_until(3.0)
    assert rec.outcome == "on_time" and rec.confirm is not None and rec.confirm.attempt == 2


def test_a_probe_with_no_reply_by_t_probe_means_lost() -> None:
    # z2 dies but its will is lost: the probe at t_rx + 63.7 ms gets no reply; after T_probe = 50 ms, lost.
    placement = Seq("n1", "here")
    c = two(placement)
    c.bus.faults.drop = lambda topic, key, sub: topic == "zone/z2/status"
    c.zones["z1"].upload(city_task(c, "T4"))
    c.clock.run_until(T_RX + 0.010)
    c.bus._clients["z2"].alive = False  # dead, will lost: z1 does not know
    c.zones["z1"].neighbours.online["z2"] = True
    c.clock.run_until(T_RX + 0.0637 + 0.050 - 1e-6)
    assert placement.calls == 1
    c.clock.run_until(T_RX + 0.0637 + 0.050 + 1e-6)
    assert placement.calls == 2


@pytest.mark.parametrize(("kind", "limit"), [("T1", 1), ("T4", 2)])
def test_a_hard_task_is_retried_once_a_soft_one_twice(kind: str, limit: int) -> None:
    """9.1, 18: retries hard 1, soft 2. Every result is lost, so each attempt (here) ends in the watchdog,
    a probe that says done, and the Q23 wait; D = 1 s so the cap never cuts the cycle short. T4: watchdog at
    +32.5 + 7 + 20 = 59.5 ms, lost at +86.5 ms, per attempt."""
    c = two(Fixed("here"), task_types__T1__deadline__loose=1.0)
    c.bus.faults.drop = lambda topic, key, sub: topic == "car/c01/result"
    c.zones["z1"].upload(city_task(c, kind))
    c.clock.run_until(5.0)
    (rec,) = c.closed
    assert (rec.outcome, rec.retries, len(rec.attempts)) == ("drop_lost", limit, limit + 1)


def test_the_retry_mask_drops_a_task_with_no_option_left() -> None:
    placement = Seq("n1", None)  # the retry finds every option masked
    c = two(placement)
    c.zones["z1"].upload(city_task(c, "T4"))
    c.clock.run_until(T_RX + 0.010)
    c.bus.kill("z2")
    c.clock.run_until(3.0)
    (rec,) = c.closed
    assert rec.outcome == "drop_lost" and placement.calls == 2 and rec.retries == 1


def test_no_ack_by_the_end_is_a_drop() -> None:
    # 9.5.5: every ack is lost; the hard T1 closes as a drop at d_A + T_ack (Q23).
    c = two(Fixed("here"))
    c.bus.faults.drop = lambda topic, key, sub: topic == "car/c01/ack"
    c.zones["z1"].upload(city_task(c, "T1"))
    c.clock.run_until(1.0)
    (rec,) = c.closed
    assert rec.outcome == "drop_no_ack" and rec.closed_at == pytest.approx(rec.d_a + 0.2, abs=1e-9)
    assert rec.result_at_car is not None  # the car got it; only the acks were lost (9.5.5: very rare)


def test_closing_cancels_every_live_timer() -> None:
    # A task closed on its ack leaves no watchdog behind: nothing fires later.
    c = two(Fixed("n1"))
    c.zones["z1"].upload(city_task(c, "T4"))
    c.clock.run_until(3.0)
    (rec,) = c.closed
    assert rec.outcome == "on_time"
    assert all(not i.live and i.timer is None and i.probe_timer is None for i in rec.attempts.values())
    assert rec.attempts[1].probes == 0


def test_the_upload_timeout_cancels_the_first_target() -> None:
    # 9.1: "cancel the first target, in case only its reply was lost". A T4 (25 ms) at z2 is still running
    # when the cancel lands (sent at t_rx + 14.4, there at + 17.4 ms): aborted, so the here attempt wins.
    c = two(Seq("n1", "here"))
    z1 = c.zones["z1"]
    z1.lose_reply = lambda key: key.attempt == 1
    z1.upload(city_task(c, "T4"))
    c.clock.run_until(3.0)
    (rec,) = c.closed
    assert c.zones["z2"].executor.status(z2_key(rec, 1)) == "cancelled"
    assert (rec.outcome, rec.result_attempt) == ("on_time", 2)


def z2_key(rec, attempt):  # type: ignore[no-untyped-def]
    return z1_key(rec, attempt)  # the anchor (z1) names the attempt; the executor is z2


def test_no_watchdog_past_the_cap() -> None:
    # A hard T1 with D = 40 ms here, its result lost: t_wd = t_rx + 10.4 + 7 + 20 = t_rx + 37.4 ms is after
    # d_A (t_rx + 34.8 ms), so no watchdog is armed (9.3): no probe, no retry; dropped at d_A + T_ack.
    placement = Seq("here", "here")
    c = two(placement, task_types__T1__deadline__loose=0.040)
    c.bus.faults.drop = lambda topic, key, sub: topic == "car/c01/result"
    c.zones["z1"].upload(city_task(c, "T1"))
    c.clock.run_until(1.0)
    (rec,) = c.closed
    assert (rec.outcome, placement.calls, rec.attempts[1].probes) == ("drop_no_ack", 1, 0)


def test_a_later_result_of_another_attempt_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    # 9.4.3. Attempt 1 (z2, its reply lost) finishes first; attempt 2 (z2 again) runs too because the cancel
    # is lost (best effort, 9.4.4): its result comes later and is ignored.
    c = two(Seq("n1", "n1"))
    z1, z2 = c.zones["z1"], c.zones["z2"]
    z1.lose_reply = lambda key: key.attempt == 1
    monkeypatch.setattr(z2.executor, "cancel", lambda key: False)
    # The acks are lost too, so the task is still open when attempt 2's result arrives (t_rx + 33.6 ms).
    c.bus.faults.drop = lambda topic, key, sub: topic == "car/c01/ack"
    z1.upload(city_task(c, "T1"))
    c.clock.run_until(1.0)
    (rec,) = c.closed
    assert z2.executor.status(z2_key(rec, 2)) == "done"  # it did run to the end
    assert rec.result_attempt == 1 and rec.report is not None and rec.report.key.attempt == 1


def test_closing_a_task_cancels_its_live_attempt() -> None:
    # 6E.15: close cancels any other live attempt (DELETE, 3 ms to z2): a running T4 there is aborted.
    c = two(Fixed("n1"))
    z1 = c.zones["z1"]
    z1.upload(city_task(c, "T4"))
    c.clock.run_until(T_RX + 0.010)  # accepted and running at z2
    rec = next(iter(z1.records.values()))
    z1._close(rec, "drop_test")
    c.clock.run_until(T_RX + 0.0131)
    assert c.zones["z2"].executor.status(z2_key(rec, 1)) == "cancelled"


def test_a_task_whose_whole_budget_went_on_the_upload_is_dropped_without_a_crash() -> None:
    # Under a heavy radio (S3) a hard frame can arrive after d_A + T_ack: no time left, nothing to schedule
    # in the past. It is dropped at its arrival.
    from pcnme.car.car import Task
    from pcnme.ids import TaskId

    c = two(Fixed("here"))
    t = c.cfg.task_types["T1"]
    old = Task(
        TaskId("c01", 0, 7),
        "c01",
        "T1",
        t.cls,
        t.deadline["loose"],
        t.l_max["loose"],
        t.input_bits,
        t.result_bits,
        c.clock.now - 1.0,
    )  # type: ignore[index]  # made 1 s ago: B_car = -0.9 s
    c.car._tasks[old.task_id] = old
    c.zones["z1"].upload(old)
    c.clock.run_until(1.0)
    (rec,) = c.closed
    # dropped at its arrival (a fixed placement ignores the masks, so here it is refused late instead)
    assert rec.outcome.startswith("drop_") and rec.closed_at == pytest.approx(rec.t_rx)


def test_an_upload_still_being_sent_is_not_a_failure() -> None:
    # CHANGES C7. The WAN already holds 216 Mbit (3 T2 clips, 419.6 ms of push) when a T4 goes to the cloud,
    # so its input takes about 420 + 67.8 ms. The old T_up from the send start, 2 x (67.8 + 65.5) = 266.7 ms,
    # would have fired mid-send and re-sent it into the same queue; now T_up starts when the send is done,
    # the accept comes 65.5 ms later, and there is no re-send.
    c = two(Fixed("cloud"))
    wan = c.zones["z1"]._cloud_path._wan
    wan.send(now=c.clock.now, size_bits=216e6, delay=0.0)
    c.zones["z1"].upload(city_task(c, "T4"))
    c.clock.run_until(5.0)
    (rec,) = c.closed
    assert rec.earlier == [] and rec.transfer is not None and rec.transfer.w_wan > 0.4
    assert rec.outcome in ("on_time", "late") and rec.reply == "accepted"


def test_a_lost_cloud_reply_times_out_after_the_send_not_at_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    # C7: the T4's send to the cloud ends at t_rx + 67.8 ms; T_up = 2 x (67.8 + 65.5) = 266.7 ms after that,
    # well before the cap (d_A + L_max = 2 s): the re-decision (here) runs then. The cloud runs nothing
    # (it refuses) and the refusal is lost, so neither a reply nor a result comes.
    from pcnme.node.executor import Refused

    placement = Seq("cloud", "here")
    c = two(placement)
    monkeypatch.setattr(c.cloud, "handle", lambda req: (Refused("late"), c.cloud.snapshot()))
    c.zones["z1"].lose_reply = lambda key: key.attempt == 1
    c.zones["z1"].upload(city_task(c, "T4"))
    c.clock.run_until(T_RX + 0.0678 + 0.2667 - 0.002)
    assert placement.calls == 1
    c.clock.run_until(T_RX + 0.0678 + 0.2667 + 0.002)
    assert placement.calls == 2
