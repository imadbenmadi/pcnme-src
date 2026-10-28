"""What a node publishes per slot type, and the wait estimate (final-design 8.5; CHANGES C4).

Per slot type with c_q > 0 a node publishes c_q, Q_q, n_q^hard, n_q^soft, B_q^hard and B_q^soft (the mean
run times of the queued tasks), the remaining time r_i of each of its c_q slots (0 when free) and u_q (busy
slots). The wait of a new task is 0 while a slot is free (u_q + the anchor's own pending sends to q < c_q);
otherwise it is the water level W at which the class backlog B, poured onto the slots in the order they free
up, fills them all:

    sum_i max(0, W - r_i) = B,    B_hard = B_q^hard,    B_soft = B_q^hard + B_q^soft

with the anchor's pending sends (R1, 9.2) added to the backlog of their class (CHANGES C4; for c_q = 1 this is
r + B, the pooled formula it replaces). The executor's own estimate (8.3) is the same function on its exact
snapshot with no pending sends (QUESTIONS Q8).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pcnme.config import TaskClass


@dataclass(frozen=True, slots=True)
class SlotState:
    """One slot type of a node, as published (8.5, C4). Times in seconds."""

    c: int  # c_q
    capacity: int  # Q_q
    n_hard: int
    n_soft: int
    b_hard: float  # sum of the mean runs of the queued hard tasks
    b_soft: float
    remaining: tuple[float, ...]  # r_i of each of the c_q slots, ascending; 0 = free (QUESTIONS Q9 for r_i)
    u: int  # busy slots

    @property
    def r(self) -> float:
        """R_q, the sum of the remaining times: kept for logs; the estimate uses each r_i (C4)."""
        return sum(self.remaining)


@dataclass(frozen=True, slots=True)
class Pending:
    """One of the anchor's own sends to this slot type, not yet answered (R1, 9.2)."""

    cls: TaskClass
    mean: float  # its mean run time on that slot type


def water_fill(remaining: Sequence[float], backlog: float) -> float:
    """The level W >= min r_i with sum_i max(0, W - r_i) = backlog (CHANGES C4)."""
    r = sorted(remaining)
    filled = 0.0
    for k in range(1, len(r) + 1):
        filled += r[k - 1]
        level = (backlog + filled) / k  # the k soonest-free slots share the backlog
        if k == len(r) or level <= r[k]:
            return level
    raise ValueError("no slots")  # unreachable for len(r) >= 1


def wait_estimate(state: SlotState, cls: TaskClass, pending: Sequence[Pending] = ()) -> float:
    """Ŵ_q for a new task of class `cls` (8.5, CHANGES C4)."""
    if state.u + len(pending) < state.c:
        return 0.0  # a slot is free
    b_hard = state.b_hard + sum(p.mean for p in pending if p.cls == "hard")
    b_soft = state.b_soft + sum(p.mean for p in pending if p.cls == "soft")
    # The hard backlog is every queued hard task; the soft one adds the queued soft tasks and ignores hard
    # tasks that arrive later (optimistic; the status probe covers that, 9.3).
    return water_fill(state.remaining, b_hard + (b_soft if cls == "soft" else 0.0))
