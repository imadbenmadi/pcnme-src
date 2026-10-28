"""What a node publishes per slot type, and the anchor's wait estimate (final-design 8.5).

Per slot type with c_q > 0 a node publishes c_q, Q_q, n_q^hard, n_q^soft, B_q^hard and B_q^soft (the mean
run times of the queued tasks), R_q (the remaining mean time of the running tasks) and u_q (busy slots).
The anchor estimates the wait of a new task as a shared pool of c_q slots without preemption: 0 while a
slot is free (u_q + its own pending sends to q < c_q), otherwise

    Ŵ_hard = (R_q + B_hard) / c_q          Ŵ_soft = (R_q + B_hard + B_soft) / c_q

with its own pending sends (R1, 9.2) added to the backlog of their class. The executor's own estimate (8.3)
is the same function on its exact snapshot with no pending sends (QUESTIONS Q8).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pcnme.config import TaskClass


@dataclass(frozen=True, slots=True)
class SlotState:
    """One slot type of a node, as published (8.5). Times in seconds."""

    c: int  # c_q
    capacity: int  # Q_q
    n_hard: int
    n_soft: int
    b_hard: float  # sum of the mean runs of the queued hard tasks
    b_soft: float
    r: float  # R_q: remaining mean time of the running tasks (QUESTIONS Q9)
    u: int  # busy slots


@dataclass(frozen=True, slots=True)
class Pending:
    """One of the anchor's own sends to this slot type, not yet answered (R1, 9.2)."""

    cls: TaskClass
    mean: float  # its mean run time on that slot type


def wait_estimate(state: SlotState, cls: TaskClass, pending: Sequence[Pending] = ()) -> float:
    """Ŵ_q for a new task of class `cls` (8.5)."""
    if state.u + len(pending) < state.c:
        return 0.0  # a slot is free
    b_hard = state.b_hard + sum(p.mean for p in pending if p.cls == "hard")
    b_soft = state.b_soft + sum(p.mean for p in pending if p.cls == "soft")
    # The hard estimate waits behind every queued hard task (safe side); the soft one ignores hard tasks that
    # arrive later (optimistic; the status probe covers that, 9.3).
    backlog = state.r + b_hard + (b_soft if cls == "soft" else 0.0)
    return backlog / state.c
