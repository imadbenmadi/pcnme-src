"""Discrete-event clock for fast mode (final-design 15.2; PLAN M1.1).

One event queue and a simulated time; nothing ever waits on a real clock. Events run in time order, and
events at the same time in the order they were scheduled (a sequence number), so a run is deterministic.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from typing import Literal

State = Literal["pending", "done", "cancelled"]


class EventHandle:
    """A scheduled event. Pass it to `EventClock.cancel`; `done` once it has run."""

    __slots__ = ("_clock", "callback", "seq", "state", "time")

    def __init__(self, clock: EventClock, time: float, seq: int, callback: Callable[[], None]) -> None:
        self._clock = clock
        self.time = time
        self.seq = seq
        self.callback = callback
        self.state: State = "pending"

    @property
    def done(self) -> bool:
        return self.state == "done"

    @property
    def cancelled(self) -> bool:
        return self.state == "cancelled"

    def __repr__(self) -> str:
        return f"EventHandle(time={self.time!r}, seq={self.seq}, {self.state})"


def _finite(time: float, what: str) -> float:
    if not math.isfinite(time):
        raise ValueError(f"{what} must be a finite time in seconds, got {time!r}")
    return time


class EventClock:
    """The simulated clock and its event queue (seconds)."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = _finite(start, "start")
        self._queue: list[tuple[float, int, EventHandle]] = []
        self._next_seq = 0  # ties at equal times run in scheduling order (PLAN M1.1)
        self._pending = 0
        self._running = False

    @property
    def now(self) -> float:
        """The current simulated time: the time of the running event, or where the last run ended."""
        return self._now

    @property
    def pending(self) -> int:
        """Events scheduled and not yet run or cancelled."""
        return self._pending

    def schedule(self, time: float, callback: Callable[[], None]) -> EventHandle:
        """Run `callback` at simulated `time` (>= now)."""
        _finite(time, "an event time")
        if time < self._now:
            raise ValueError(f"cannot schedule in the past: {time!r} < now {self._now!r}")
        handle = EventHandle(self, time, self._next_seq, callback)
        self._next_seq += 1
        heapq.heappush(self._queue, (time, handle.seq, handle))
        self._pending += 1
        return handle

    def schedule_in(self, delay: float, callback: Callable[[], None]) -> EventHandle:
        """Run `callback` after `delay` seconds (>= 0) of simulated time."""
        if delay < 0:
            raise ValueError(f"a delay cannot be negative, got {delay!r}")
        return self.schedule(self._now + _finite(delay, "a delay"), callback)

    def cancel(self, handle: EventHandle) -> bool:
        """Cancel a pending event. True if it was pending; False if it already ran or was cancelled."""
        if handle._clock is not self:
            raise ValueError("this event belongs to another clock")
        if handle.state != "pending":
            return False
        handle.state = "cancelled"  # left in the queue and skipped when it comes up
        self._pending -= 1
        return True

    def run_until(self, end: float) -> None:
        """Run every event with time <= `end`, in order, then leave `now` at `end`.

        Events scheduled by a callback join the same queue. If a callback raises, the run stops at that event
        (`now` = its time) and the error propagates; the remaining events stay queued.
        """
        _finite(end, "the end time")
        if self._running:
            raise RuntimeError("run_until cannot be called inside an event callback")
        if end < self._now:
            raise ValueError(f"cannot run into the past: {end!r} < now {self._now!r}")
        self._running = True
        try:
            while self._queue and self._queue[0][0] <= end:
                time, _, handle = heapq.heappop(self._queue)
                if handle.state == "cancelled":
                    continue
                self._now = time
                handle.state = "done"  # before the call, so the event cannot cancel itself
                self._pending -= 1
                handle.callback()
            self._now = end
        finally:
            self._running = False
