"""What a node publishes per slot type, and the wait estimate (final-design 8.5; CHANGES C4).

Per slot type with c_q > 0 a node publishes c_q, n^max_q, n_q^hard, n_q^soft, Ω_q^hard and Ω_q^soft (the mean
run times of the queued tasks), the remaining time T^rem_q,j of each of its c_q slots (0 when free) and
n^busy_q (busy slots). The wait of a new task is 0 while a slot is free (n^busy_q + the anchor's own pending
sends to q < c_q); otherwise it is the water level Ŵ^queue_q at which the class backlog Ω, poured onto the
slots in the order they free up, fills them all:

    sum_j max(0, Ŵ^queue_q - T^rem_q,j) = Ω,    Ω = Ω_q^hard (hard),    Ω = Ω_q^hard + Ω_q^soft (soft)

with the anchor's pending sends (R1, 9.2) added to the backlog of their class (CHANGES C4; for c_q = 1 this is
T^rem + Ω, the pooled formula it replaces). The executor's own estimate (8.3) is the same function on its
exact snapshot with no pending sends (QUESTIONS Q8).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pcnme.config import TaskClass


@dataclass(frozen=True, slots=True)
class SlotState:
    """One slot type of a node, as published (8.5, C4). Times in seconds."""

    c: int  # c_q
    capacity: int  # n^max_q
    n_hard: int
    n_soft: int
    b_hard: float  # sum of the mean runs of the queued hard tasks
    b_soft: float
    # T^rem_q,j of each of the c_q slots, ascending; 0 = free (QUESTIONS Q9 for T^rem)
    remaining: tuple[float, ...]
    u: int  # busy slots

    @property
    def r(self) -> float:
        """R_q, the sum of the remaining times: kept for logs; the estimate uses each T^rem_q,j (C4)."""
        return sum(self.remaining)


@dataclass(frozen=True, slots=True)
class Pending:
    """One of the anchor's own sends to this slot type, not yet answered (R1, 9.2)."""

    cls: TaskClass
    mean: float  # its mean run time on that slot type


def water_fill(remaining: Sequence[float], backlog: float) -> float:
    """The level Ŵ^queue_q >= min T^rem_q,j with sum_j max(0, Ŵ^queue_q - T^rem_q,j) = backlog (CHANGES
    C4)."""
    r = sorted(remaining)
    filled = 0.0
    for k in range(1, len(r) + 1):
        filled += r[k - 1]
        level = (backlog + filled) / k  # the k soonest-free slots share the backlog
        if k == len(r) or level <= r[k]:
            return level
    raise ValueError("no slots")  # unreachable for len(r) >= 1


def wait_estimate(state: SlotState, cls: TaskClass, pending: Sequence[Pending] = ()) -> float:
    """Ŵ^queue_q for a new task of class `cls` (8.5, CHANGES C4)."""
    if state.u + len(pending) < state.c:
        return 0.0  # a slot is free
    b_hard = state.b_hard + sum(p.mean for p in pending if p.cls == "hard")
    b_soft = state.b_soft + sum(p.mean for p in pending if p.cls == "soft")
    # The hard backlog is every queued hard task; the soft one adds the queued soft tasks and ignores hard
    # tasks that arrive later (optimistic; the status probe covers that, 9.3).
    return water_fill(state.remaining, b_hard + (b_soft if cls == "soft" else 0.0))
