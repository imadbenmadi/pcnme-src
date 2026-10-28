"""Drops, aborts, cancels and T_keep (PLAN M1.7; final-design 8.6, 9.4, 5.5, 12.2, 12.5).

One test or more per row of the 8.6 table, on the executor's side:

| Where         | Hard task                                | Soft task                               | Tests |
| On arrival    | Refused "full" or "late"                 | Refused "full", or "late" past D+L_max  | M1.5  |
| Head of queue | Skipped when t + Ê^low > d_k             | Skipped when t + Ê^low > d_k + L_max    | M1.6  |
| While running | Aborted when t > d_k                     | Aborted when t > d_k + L_max            | here  |
| Anchor cancel | Removed if queued, aborted if running    | Same                                    | here  |
| No ack by end | Drop (anchor side, M2/M3)                | Drop (anchor side, M2/M3)               | -     |
"""

from __future__ import annotations

import pytest

from pcnme.node.executor import AttemptKey, Ignored, Refused, TaskRequest
from tests.node.helpers import report, row, tiny

T_KEEP = 10.0  # config scheduler.T_keep


# --- While running: aborted at d_k (hard) or d_k + L_max (soft) ---------------------------------------------


def test_a_hard_task_running_past_d_k_is_aborted_with_its_tail() -> None:
    # H: mean 100 ms (accepted, F = 100 <= 200 ms) but really runs 300 ms; longest step 50 ms. Aborted at
    # d_k = 200 ms; the slot stays busy for min(remaining 100, d_abort 50) = 50 ms, so it frees at 250 ms.
    # Energy = PUE 1 x 20 W x (200 + 50 ms) = 5 J, reported at once at 200 ms.
    n = tiny([row("H", "cpu", 0.100, run=0.300), row("next", "cpu", 0.010)])
    h = n.submit("H", "hard", 0.200)
    nxt = n.submit("next", "hard", 10.0)
    n.clock.run_until(0.200)
    rep = report(n, h.key)  # already reported at the abort, before the tail ends (8.6: "at once")
    assert (rep.outcome, rep.start, rep.finish, rep.busy) == (
        "aborted",
        0.0,
        pytest.approx(0.250),
        pytest.approx(0.250),
    )
    assert rep.energy == pytest.approx(20 * 0.250)
    n.clock.run_until(5.0)
    assert report(n, nxt.key).start == pytest.approx(0.250)


def test_the_tail_is_the_remaining_run_when_that_is_shorter_than_d_abort() -> None:
    # Runs 210 ms, aborted at 200 ms: remaining 10 ms < d_abort 50 ms, so the slot frees at 210 ms.
    n = tiny([row("H", "cpu", 0.100, run=0.210)])
    h = n.submit("H", "hard", 0.200)
    n.clock.run_until(5.0)
    assert report(n, h.key).finish == pytest.approx(0.210)


def test_finishing_exactly_at_d_k_is_in_time() -> None:
    # "Aborted when t > d_k": a run that ends at d_k is delivered.
    n = tiny([row("H", "cpu", 0.125, run=0.25)])
    h = n.submit("H", "hard", 0.25)
    n.clock.run_until(5.0)
    assert report(n, h.key).outcome == "done"


def test_a_soft_task_runs_late_until_d_k_plus_l_max() -> None:
    # Soft, d_k = 200 ms, L_max = 100 ms. A run ending at 280 ms is late but delivered; one ending at 350 ms
    # is aborted at 300 ms.
    late = tiny([row("S", "cpu", 0.100, run=0.280)])
    s1 = late.submit("S", "soft", 0.200, l_max=0.100)
    too_late = tiny([row("S", "cpu", 0.100, run=0.350)])
    s2 = too_late.submit("S", "soft", 0.200, l_max=0.100)
    late.clock.run_until(5.0)
    too_late.clock.run_until(5.0)
    assert report(late, s1.key).outcome == "done"
    r2 = report(too_late, s2.key)
    assert (r2.outcome, r2.finish) == ("aborted", pytest.approx(0.350))  # tail 50


# --- Anchor cancel: removed if queued, aborted if running (8.6, 9.4) ----------------------------------------


def test_cancel_removes_a_queued_task() -> None:
    n = tiny([row("A", "cpu", 0.100), row("B", "cpu", 0.100)])
    n.submit("A", "hard", 10.0)
    b = n.submit("B", "hard", 10.0)
    n.clock.run_until(0.050)
    assert n.ex.cancel(b.key) is True
    n.clock.run_until(5.0)
    rep = report(n, b.key)
    assert (rep.outcome, rep.start, rep.energy) == ("cancelled", None, 0.0)
    assert b.key not in n.started()


def test_cancel_aborts_a_running_task_with_its_tail() -> None:
    # Cancelled at 30 ms while running (would end at 100 ms): busy until 30 + min(70, 50) = 80 ms.
    n = tiny([row("A", "cpu", 0.100), row("B", "cpu", 0.010)])
    a = n.submit("A", "hard", 10.0)
    b = n.submit("B", "hard", 10.0)
    n.clock.run_until(0.030)
    assert n.ex.cancel(a.key) is True
    n.clock.run_until(5.0)
    rep = report(n, a.key)
    assert (rep.outcome, rep.finish, rep.busy) == (
        "cancelled",
        pytest.approx(0.080),
        pytest.approx(0.080),
    )
    assert rep.energy == pytest.approx(20 * 0.080)  # execution really spent (12.5)
    assert report(n, b.key).start == pytest.approx(0.080)


def test_cancelling_an_unknown_or_finished_attempt_is_ignored_and_twice_is_harmless() -> None:
    n = tiny([row("A", "cpu", 0.010), row("B", "cpu", 0.100)])
    a = n.submit("A", "hard", 10.0)
    n.clock.run_until(1.0)
    assert n.ex.cancel(a.key) is False  # finished
    assert n.ex.cancel(AttemptKey(("c99", 0, 1), "z1", 1)) is False  # unknown
    b = n.submit("B", "hard", 10.0)
    assert n.ex.cancel(b.key) is True
    assert n.ex.cancel(b.key) is False  # second cancel: no effect
    n.clock.run_until(5.0)
    assert [r.outcome for r in n.reports] == ["done", "cancelled"]


def test_a_cancel_during_the_abort_tail_changes_nothing() -> None:
    n = tiny([row("H", "cpu", 0.100, run=0.300)])
    h = n.submit("H", "hard", 0.200)
    n.clock.run_until(0.220)  # aborted at 200 ms, tail until 250 ms
    assert n.ex.cancel(h.key) is False
    n.clock.run_until(5.0)
    assert [r.outcome for r in n.reports] == ["aborted"]


# --- (attempt key, outcome) kept for T_keep (8.6, 5.5) ------------------------------------------------------


def test_status_follows_an_attempt_and_its_outcome_is_kept_for_t_keep() -> None:
    n = tiny([row("A", "cpu", 0.100), row("B", "cpu", 0.100)])
    a = n.submit("A", "hard", 10.0)
    b = n.submit("B", "hard", 10.0)
    assert (n.ex.status(a.key), n.ex.status(b.key)) == ("running", "queued")
    n.clock.run_until(0.150)
    assert (n.ex.status(a.key), n.ex.status(b.key)) == ("done", "running")
    n.clock.run_until(0.100 + T_KEEP - 0.001)
    assert n.ex.status(a.key) == "done"  # still kept just before T_keep
    n.clock.run_until(0.100 + T_KEEP + 0.001)
    assert n.ex.status(a.key) == "not found"
    assert n.ex.status(AttemptKey(("c99", 0, 1), "z1", 1)) == "not found"


@pytest.mark.parametrize("outcome", ["skipped", "aborted", "cancelled"])
def test_every_drop_outcome_is_kept(outcome: str) -> None:
    rows = [row("long", "cpu", 0.200, run=0.300), row("H", "cpu", 0.090), row("X", "cpu", 0.100, run=0.300)]
    n = tiny(rows)
    if outcome == "skipped":
        n.submit("long", "hard", 10.0)
        r = n.submit("H", "hard", 0.300)  # hopeless at 300 ms (8.4 step 1)
    elif outcome == "aborted":
        r = n.submit("X", "hard", 0.200)
    else:
        r = n.submit("X", "hard", 10.0)
        n.ex.cancel(r.key)
    n.clock.run_until(1.0)
    assert n.ex.status(r.key) == outcome


def test_a_repeated_upload_of_the_same_attempt_is_ignored() -> None:
    # 8.6: the kept outcome lets the executor ignore a repeated upload of the same attempt, live or finished.
    n = tiny([row("A", "cpu", 0.100)])
    a = n.submit("A", "hard", 10.0)
    assert n.ex.submit(a) == Ignored()  # while it runs
    n.clock.run_until(1.0)
    assert n.ex.submit(a) == Ignored()  # finished, still kept
    n.clock.run_until(1.0 + T_KEEP)
    assert n.ex.submit(a) != Ignored()  # forgotten after T_keep
    assert len([r for r in n.reports if r.key == a.key]) == 1  # never ran twice in the window


def test_another_attempt_of_the_same_task_is_not_a_repeat() -> None:
    n = tiny([row("A", "cpu", 0.100)])
    first = n.submit("A", "hard", 10.0)
    task_id = first.key.task_id
    retry = TaskRequest(AttemptKey(task_id, "z1", 2), "A", "hard", 0.0, 10.0)  # attempt 2
    other_anchor = TaskRequest(AttemptKey(task_id, "z2", 1), "A", "hard", 0.0, 10.0)
    assert n.ex.submit(retry) != Ignored()
    assert n.ex.submit(other_anchor) != Ignored()  # 5.5: the anchor id is part of the key


def test_a_refused_attempt_is_not_kept() -> None:
    # Engineering choice: a refusal is answered at once and leaves nothing behind.
    n = tiny([row("A", "cpu", 0.100)])
    r = n.req("A", "hard", 0.050)  # F = 100 ms > 50 ms
    assert n.ex.submit(r) == Refused(reason="late")
    assert n.ex.status(r.key) == "not found"


# --- R_q during an abort tail (8.5; Q9) ---------------------------------------------------------------------


def test_a_slot_in_its_abort_tail_counts_its_exact_remaining_time() -> None:
    n = tiny([row("H", "cpu", 0.100, run=0.300), row("Q", "cpu", 0.010)])
    n.submit("H", "hard", 0.200)
    n.clock.run_until(0.210)  # aborted at 200 ms; tail until 250 ms
    # 40 ms of tail left (not max(0, 100 - 210) = 0 from the mean).
    assert n.ex.wait_estimate("cpu", "hard") == pytest.approx(0.040)
