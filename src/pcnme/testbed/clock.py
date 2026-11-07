"""The testbed's real clock (final-design 15.2; PLAN M9.1 step 1).

EventClock's interface (now, pending, schedule, schedule_in, cancel) on real time, so the zone, car and cloud
code runs unchanged in a container. `now` is seconds since the clock started (time.monotonic), never another
machine's time (5.4). One loop thread runs every callback in time order, ties in scheduling order, exactly as
in fast mode; MQTT and HTTP threads hand work in with call_soon, which is the only thread-safe entry point.
"""

from __future__ import annotations

import heapq
import threading
import time
from collections import Counter
from collections.abc import Callable

from pcnme.sim.clock import EventHandle, _finite


class RealClock:
    def __init__(self) -> None:
        self._t0 = time.monotonic()
        self._queue: list[tuple[float, int, EventHandle]] = []
        # how late events ran and which callbacks held the loop longest (diagnosis, PLAN Q1)
        self.lag: Counter[str] = Counter()
        self.slow: Counter[str] = Counter()
        self._next_seq = 0
        self._pending = 0
        self._cv = threading.Condition()
        self._stopped = False

    @property
    def now(self) -> float:
        return time.monotonic() - self._t0

    @property
    def pending(self) -> int:
        return self._pending

    def schedule(self, at: float, callback: Callable[[], None]) -> EventHandle:
        """Run `callback` at clock time `at`; a time already past runs as soon as the loop is free."""
        _finite(at, "an event time")
        with self._cv:
            handle = EventHandle(self, at, self._next_seq, callback)  # type: ignore[arg-type]
            self._next_seq += 1
            heapq.heappush(self._queue, (at, handle.seq, handle))
            self._pending += 1
            self._cv.notify()
        return handle

    def schedule_in(self, delay: float, callback: Callable[[], None]) -> EventHandle:
        if delay < 0:
            raise ValueError(f"a delay cannot be negative, got {delay!r}")
        return self.schedule(self.now + _finite(delay, "a delay"), callback)

    def call_soon(self, callback: Callable[[], None]) -> EventHandle:
        """From any thread: run `callback` on the loop thread, after the events already due."""
        return self.schedule(self.now, callback)

    def cancel(self, handle: EventHandle) -> bool:
        if handle._clock is not self:
            raise ValueError("this event belongs to another clock")
        with self._cv:
            if handle.state != "pending":
                return False
            handle.state = "cancelled"
            self._pending -= 1
            return True

    def stop(self) -> None:
        """End run_for at the next turn of the loop (from any thread)."""
        with self._cv:
            self._stopped = True
            self._cv.notify()

    def run_for(self, seconds: float) -> None:
        """The loop: run every event as its time comes, for `seconds` of real time or until stop(). A
        callback that raises ends the loop and the error propagates."""
        end = self.now + seconds
        self._stopped = False
        while True:
            with self._cv:
                while True:
                    if self._stopped or self.now >= end:
                        return
                    if self._queue and self._queue[0][2].state == "cancelled":
                        heapq.heappop(self._queue)
                        continue
                    wait = min(end, self._queue[0][0] if self._queue else end) - self.now
                    if self._queue and wait <= 0:
                        due, _, handle = heapq.heappop(self._queue)
                        late = self.now - due
                        handle.state = "done"
                        self._pending -= 1
                        break
                    self._cv.wait(timeout=max(wait, 0.0))
            for limit in (0.01, 0.05, 0.2, 1.0):
                if late > limit:
                    self.lag[f"events late > {int(limit * 1000)} ms"] += 1
            t0 = time.monotonic()
            handle.callback()  # outside the lock: a callback may schedule more work
            took = time.monotonic() - t0
            if took > 0.02:
                name = getattr(handle.callback, "__qualname__", "?")
                self.slow[f"{name} > 20 ms"] += 1
                self.slow[f"{name} total ms"] += round(took * 1000)
