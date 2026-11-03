"""The testbed's real clock (final-design 15.2; PLAN M9.1 step 1): EventClock's interface on real time, one
loop thread running every callback, work handed in from other threads."""

from __future__ import annotations

import threading
import time

import pytest

from pcnme.testbed.clock import RealClock


def test_events_run_in_time_order_then_scheduling_order() -> None:
    clock = RealClock()
    out: list[str] = []
    clock.schedule_in(0.03, lambda: out.append("c"))
    clock.schedule_in(0.01, lambda: out.append("a"))
    clock.schedule_in(0.01, lambda: out.append("b"))  # same time: after a
    clock.run_for(0.1)
    assert out == ["a", "b", "c"]


def test_an_event_runs_no_earlier_than_its_time() -> None:
    clock = RealClock()
    seen: list[float] = []
    clock.schedule_in(0.05, lambda: seen.append(clock.now))
    clock.run_for(0.1)
    assert len(seen) == 1 and 0.05 <= seen[0] < 0.09


def test_cancel_and_pending() -> None:
    clock = RealClock()
    h = clock.schedule_in(0.01, lambda: pytest.fail("cancelled"))
    clock.schedule_in(0.02, lambda: None)
    assert clock.pending == 2 and clock.cancel(h) and not clock.cancel(h) and clock.pending == 1
    clock.run_for(0.05)
    assert clock.pending == 0 and h.cancelled


def test_the_past_and_bad_times_are_refused() -> None:
    clock = RealClock()
    with pytest.raises(ValueError):
        clock.schedule_in(-1.0, lambda: None)
    with pytest.raises(ValueError):
        clock.schedule(float("nan"), lambda: None)


def test_other_threads_hand_work_to_the_loop() -> None:
    clock = RealClock()
    ran_on: list[str] = []
    loop = threading.Thread(target=clock.run_for, args=(0.3,), name="loop")
    loop.start()
    time.sleep(0.02)
    threading.Thread(
        target=lambda: clock.call_soon(lambda: ran_on.append(threading.current_thread().name))
    ).start()
    loop.join()
    assert ran_on == ["loop"]  # the callback ran on the loop thread, not the caller's


def test_stop_ends_the_loop_early() -> None:
    clock = RealClock()
    clock.schedule_in(0.01, clock.stop)
    t0 = time.monotonic()
    clock.run_for(5.0)
    assert time.monotonic() - t0 < 1.0


def test_a_callback_error_stops_the_loop_and_propagates() -> None:
    clock = RealClock()

    def boom() -> None:
        raise RuntimeError("x")

    clock.schedule_in(0.0, boom)
    with pytest.raises(RuntimeError):
        clock.run_for(1.0)
