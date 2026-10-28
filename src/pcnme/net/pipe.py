"""One shared FIFO pipe, the model of every link that carries task data (final-design 4.1; PLAN M1.2).

    W_pipe = max(0, t_free - t_now)          t_free <- max(t_now, t_free) + S / B
    delivery = t_now + W_pipe + S / B + d

S = size in bits, B = bandwidth in bit/s, d = the transfer's random delay. The delay does not hold the pipe:
the next transfer waits only for this one's push, so it can even arrive first. The pipe is pure arithmetic;
the caller draws d (keyed by task, attempt and hop: CHANGES C3) and schedules the delivery on its clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Transfer:
    """One transfer through a pipe, in seconds."""

    start: float  # when it was handed to the pipe
    wait: float  # W_pipe: time queued behind earlier pushes
    push: float  # S / B
    delay: float  # d
    push_end: float  # start + wait + push: the pipe is free again (t_free)
    delivery: float  # push_end + delay: the data has arrived

    @property
    def duration(self) -> float:
        """T_transfer = W_pipe + S / B + d (4.1)."""
        return self.delivery - self.start


def _positive(value: float, what: str) -> float:
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{what} must be a finite number > 0, got {value!r}")
    return value


class FifoPipe:
    """A FIFO pipe of fixed bandwidth (bit/s) shared by everything sent through it."""

    def __init__(self, bandwidth: float) -> None:
        self._bandwidth = _positive(bandwidth, "the bandwidth")
        self._t_free = 0.0  # when the pipe is next free (4.1)
        self._last_now = -math.inf

    @property
    def bandwidth(self) -> float:
        return self._bandwidth

    @property
    def t_free(self) -> float:
        return self._t_free

    def wait_at(self, now: float) -> float:
        """W_pipe at `now`, without sending: the zone knows its own pipe's wait exactly (4.3)."""
        self._check_now(now)
        return max(0.0, self._t_free - now)

    def send(self, now: float, size_bits: float, delay: float) -> Transfer:
        """Push `size_bits` into the pipe at `now`; `delay` is added after the push and does not hold it."""
        _positive(size_bits, "the size in bits")
        if not (math.isfinite(delay) and delay >= 0):
            raise ValueError(f"the delay must be a finite number >= 0, got {delay!r}")
        wait = self.wait_at(now)
        push = size_bits / self._bandwidth
        push_end = max(now, self._t_free) + push  # t_free <- max(t_now, t_free) + S / B (4.1)
        self._t_free = push_end
        self._last_now = now
        return Transfer(
            start=now, wait=wait, push=push, delay=delay, push_end=push_end, delivery=push_end + delay
        )

    def _check_now(self, now: float) -> None:
        if not math.isfinite(now):
            raise ValueError(f"the time must be finite, got {now!r}")
        if now < self._last_now:
            raise ValueError(
                f"FIFO: a transfer at {now!r} is earlier than the last one at {self._last_now!r}"
            )
