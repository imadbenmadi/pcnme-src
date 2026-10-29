"""Training samples: the 1-step and the n-step parts of every transition (final-design 11.3.6-11.3.7, 13.5;
PLAN M6.2; QUESTIONS Q36). THIS TOUCHES THE AGENT.

Time-based discount (11.3.6): a step of Δt is discounted by γ^(Δt/τ). The 1-step target bootstraps from s'
with γ^(Δt/τ); the n-step one over the same anchor's next n decisions (13.5):

    R_n = Σ_{k<m} γ^((t_k - t_0)/τ) r_k,   bootstrap γ^((t_m - t_0)/τ) Q_target(s_m, a*)

A transition with no s' (the anchor's last of an episode, truncated, Q36) is left out of every bootstrapped
target: its own 1-step target, and every n-step chain stops before it. Engineering choice: near the end of a
chain the n-step part uses the m < n steps that still have an s' (m = 0: no n-step target).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pcnme.agent.transitions import Transition


@dataclass(frozen=True, slots=True)
class Sample:
    """One transition as the learner uses it."""

    s: tuple[float, ...]
    a: int
    r: float
    valid: tuple[bool, ...]
    expert: bool
    # 1-step: s', its valid actions and γ^(Δt/τ); has1 False when there is no s'
    has1: bool
    s1: tuple[float, ...] | None
    valid1: tuple[bool, ...] | None
    g1: float
    # n-step over m <= n steps: the discounted reward sum, s_m, its valid actions and γ^((t_m - t_0)/τ)
    hasn: bool
    m: int
    rn: float
    sn: tuple[float, ...] | None
    validn: tuple[bool, ...] | None
    gn: float


def discount(dt: float, gamma_per_tau: float, tau: float) -> float:
    """11.3.6: γ^(Δt/τ)."""
    return float(gamma_per_tau ** (dt / tau))


def build(
    chain: Sequence[Transition],
    n: int,
    gamma_per_tau: float,
    tau: float,
    expert: Sequence[bool] | None = None,
) -> list[Sample]:
    """The samples of one anchor's transitions, in decision order (each one's s' is the next one's s)."""
    for a, b in itertools.pairwise(chain):
        if a.next_key != (b.task_id, b.attempt):
            raise ValueError(f"not a chain: {a.task_id}/{a.attempt} is followed by {b.task_id}/{b.attempt}")
    flags = expert if expert is not None else [False] * len(chain)
    out = []
    for i, t in enumerate(chain):
        has1 = t.s_next is not None and t.dt is not None
        m, rn = 0, 0.0
        while m < n and i + m < len(chain) and chain[i + m].s_next is not None:
            k = chain[i + m]
            rn += discount(k.time - t.time, gamma_per_tau, tau) * k.r
            m += 1
        last = chain[i + m - 1] if m else None
        t_m = last.time + last.dt if last is not None and last.dt is not None else t.time
        out.append(
            Sample(
                s=t.s,
                a=t.a,
                r=t.r,
                valid=t.valid,
                expert=flags[i],
                has1=has1,
                s1=t.s_next,
                valid1=t.valid_next,
                g1=discount(t.dt, gamma_per_tau, tau) if has1 and t.dt is not None else 0.0,
                hasn=m > 0,
                m=m,
                rn=rn,
                sn=last.s_next if last is not None else None,
                validn=last.valid_next if last is not None else None,
                gn=discount(t_m - t.time, gamma_per_tau, tau) if m else 0.0,
            )
        )
    return out


def anchor_samples(
    transitions: Sequence[Transition],
    order: Sequence[tuple[object, int]],
    n: int,
    gamma_per_tau: float,
    tau: float,
    expert: Callable[[Transition], bool] = lambda t: False,
) -> dict[tuple[object, int], Sample]:
    """The samples of one anchor's transitions, keyed by (task id, attempt). `order` is the anchor's decisions
    in decision order; a decision whose transition is missing (discarded at an episode end) splits the chain,
    and the n-step part of the one before it stops there (it still bootstraps from its own s')."""
    by_key = {(t.task_id, t.attempt): t for t in transitions}
    out: dict[tuple[object, int], Sample] = {}
    segment: list[Transition] = []

    def flush() -> None:
        for t, smp in zip(
            segment, build(segment, n, gamma_per_tau, tau, [expert(t) for t in segment]), strict=True
        ):
            out[(t.task_id, t.attempt)] = smp
        segment.clear()

    for key in order:
        t = by_key.get(key)
        if t is None:
            flush()
            continue
        if segment and segment[-1].next_key != key:
            flush()
        segment.append(t)
    flush()
    return out
