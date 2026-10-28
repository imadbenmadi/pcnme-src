"""Discrete-event clock (PLAN M1.1; final-design 15.2)."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pcnme.sim.clock import EventClock, EventHandle


def fresh() -> tuple[EventClock, list[tuple[float, str]]]:
    return EventClock(), []


def recorder(clock: EventClock, log: list[tuple[float, str]], name: str) -> Callable[[], None]:
    def callback() -> None:
        log.append((clock.now, name))

    return callback


# --- R1, R2: order ------------------------------------------------------------------------------------------


def test_events_run_in_time_order() -> None:
    clock, log = fresh()
    for t, name in [(0.3, "c"), (0.1, "a"), (0.2, "b")]:
        clock.schedule(t, recorder(clock, log, name))
    clock.run_until(1.0)
    assert log == [(0.1, "a"), (0.2, "b"), (0.3, "c")]


def test_equal_times_run_in_scheduling_order() -> None:
    clock, log = fresh()
    for name in "abcde":
        clock.schedule(0.5, recorder(clock, log, name))
    clock.run_until(1.0)
    assert [n for _, n in log] == list("abcde")


@given(st.lists(st.sampled_from([0.0, 0.001, 0.002, 0.5, 1.0, 2.5]), max_size=60))
def test_order_is_time_then_scheduling_order(times: list[float]) -> None:
    # Any mix of times, many of them equal: the run order is the sort by (time, scheduling index).
    clock, ran = EventClock(), list[int]()
    for i, t in enumerate(times):
        clock.schedule(t, lambda i=i: ran.append(i))
    clock.run_until(3.0)
    assert ran == sorted(range(len(times)), key=lambda i: (times[i], i))


# --- R3: cancel ---------------------------------------------------------------------------------------------


def test_cancelled_events_never_run() -> None:
    clock, log = fresh()
    keep = clock.schedule(0.1, recorder(clock, log, "keep"))
    drop = clock.schedule(0.2, recorder(clock, log, "drop"))
    assert clock.cancel(drop) is True
    clock.run_until(1.0)
    assert log == [(0.1, "keep")]
    assert keep.done and drop.cancelled


def test_cancel_twice_or_after_running_is_a_no_op() -> None:
    clock = EventClock()
    h = clock.schedule(0.1, lambda: None)
    assert clock.cancel(h) is True
    assert clock.cancel(h) is False
    ran = clock.schedule(0.2, lambda: None)
    clock.run_until(1.0)
    assert clock.cancel(ran) is False


def test_an_event_can_cancel_a_later_one_at_the_same_time() -> None:
    clock, log = fresh()
    later: list[EventHandle] = []
    clock.schedule(0.5, lambda: clock.cancel(later[0]))
    later.append(clock.schedule(0.5, recorder(clock, log, "never")))
    clock.run_until(1.0)
    assert log == []


def test_pending_counts_live_events_only() -> None:
    clock = EventClock()
    handles = [clock.schedule(0.1 * i, lambda: None) for i in range(5)]
    clock.cancel(handles[1])
    assert clock.pending == 4
    clock.run_until(0.25)  # runs 0.0, 0.1 and 0.2
    assert clock.pending == 2


def test_cancelling_a_handle_of_another_clock_is_refused() -> None:
    a, b = EventClock(), EventClock()
    h = a.schedule(0.1, lambda: None)
    with pytest.raises(ValueError, match="another clock"):
        b.cancel(h)


# --- R4: now ------------------------------------------------------------------------------------------------


def test_now_is_the_event_time_inside_a_callback() -> None:
    clock, seen = EventClock(), list[float]()
    clock.schedule(0.42, lambda: seen.append(clock.now))
    assert clock.now == 0.0
    clock.run_until(1.0)
    assert seen == [0.42]


def test_the_clock_can_start_at_a_given_time() -> None:
    clock = EventClock(start=5.0)
    assert clock.now == 5.0
    with pytest.raises(ValueError, match="past"):
        clock.schedule(4.0, lambda: None)


# --- R5: run_until ------------------------------------------------------------------------------------------


def test_run_until_includes_its_end_and_leaves_now_there() -> None:
    clock, log = fresh()
    clock.schedule(1.0, recorder(clock, log, "at end"))
    clock.schedule(1.5, recorder(clock, log, "after"))
    clock.run_until(1.0)
    assert log == [(1.0, "at end")]
    assert clock.now == 1.0
    assert clock.pending == 1
    clock.run_until(2.0)
    assert log == [(1.0, "at end"), (1.5, "after")]
    assert clock.now == 2.0


def test_run_until_with_no_events_still_advances_now() -> None:
    clock = EventClock()
    clock.run_until(3.0)
    assert clock.now == 3.0


def test_schedule_in_is_relative_to_now() -> None:
    clock, log = fresh()
    clock.run_until(2.0)
    clock.schedule_in(0.25, recorder(clock, log, "x"))
    clock.run_until(3.0)
    assert log == [(2.25, "x")]


# --- R6: refused --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_times_are_refused(bad: float) -> None:
    clock = EventClock()
    with pytest.raises(ValueError, match="finite"):
        clock.schedule(bad, lambda: None)
    with pytest.raises(ValueError, match="finite"):
        clock.run_until(bad)


def test_scheduling_in_the_past_is_refused() -> None:
    clock = EventClock()
    clock.run_until(1.0)
    with pytest.raises(ValueError, match="past"):
        clock.schedule(0.5, lambda: None)
    with pytest.raises(ValueError, match="negative"):
        clock.schedule_in(-0.1, lambda: None)


def test_going_back_in_time_is_refused() -> None:
    clock = EventClock()
    clock.run_until(1.0)
    with pytest.raises(ValueError, match="past"):
        clock.run_until(0.5)


def test_run_until_inside_a_callback_is_refused() -> None:
    clock = EventClock()
    clock.schedule(0.1, lambda: clock.run_until(0.5))
    with pytest.raises(RuntimeError, match="inside"):
        clock.run_until(1.0)


def test_a_failing_callback_stops_the_run_and_leaves_the_clock_usable() -> None:
    clock, log = fresh()

    def boom() -> None:
        raise KeyError("boom")

    clock.schedule(0.1, boom)
    clock.schedule(0.2, recorder(clock, log, "next"))
    with pytest.raises(KeyError, match="boom"):
        clock.run_until(1.0)
    assert clock.now == 0.1  # stopped at the failing event
    clock.run_until(1.0)  # the remaining events still run afterwards
    assert log == [(0.2, "next")]


# --- R7: events scheduled from a callback -------------------------------------------------------------------


def test_an_event_scheduled_now_from_a_callback_runs_after_those_already_waiting() -> None:
    clock, log = fresh()

    def first() -> None:
        log.append((clock.now, "first"))
        clock.schedule(clock.now, recorder(clock, log, "spawned"))

    clock.schedule(0.5, first)
    clock.schedule(0.5, recorder(clock, log, "second"))
    clock.run_until(1.0)
    assert [n for _, n in log] == ["first", "second", "spawned"]


def test_a_chain_of_events_runs_within_one_run_until() -> None:
    clock, ticks = EventClock(), list[float]()

    def tick() -> None:
        ticks.append(clock.now)
        if len(ticks) < 10:
            clock.schedule_in(0.1, tick)

    clock.schedule(0.0, tick)
    clock.run_until(0.55)
    assert ticks == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    clock.run_until(10.0)
    assert len(ticks) == 10


def test_same_schedule_gives_the_same_run() -> None:
    # Determinism: no hidden state (wall clock, hash order) changes the order of a run.
    def run() -> list[tuple[float, str]]:
        clock, log = fresh()
        for i in range(200):
            clock.schedule((i * 7919) % 97 / 100, recorder(clock, log, f"e{i}"))
        clock.run_until(1.0)
        return log

    assert run() == run()
