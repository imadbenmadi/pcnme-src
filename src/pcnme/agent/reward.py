"""Rewards, one per attempt (final-design 11.4, 12.5, 13.2; PLAN M4.2). THIS TOUCHES THE AGENT.

Every reward is B minus the per-task cost NSGA-II minimises (13.2): w1 c^time + w2 ê, with w1 = w, w2 = 1 - w,
ê = min(E / E_ref, ē) and c^time = L/D + w3 max(0, L - D)/D.

    Delivered first, on time                    B - w1 L/D - w2 ê
    Delivered first, soft and late              B - w1 (L/D + w3 (L - D)/D) - w2 ê
    Refused, lost, cancelled, or came second    -(w1 T_waste/D + w2 min(E_waste/E_ref, ē))
    Last attempt of a dropped task              B - w1 C_drop - w2 ē        (C_drop = 5 from the config)

A drop is charged the worst energy ē, not what it spent (the spent energy is logged). So any drop is worse
than any delivery for every w > 0 (N18): a delivery's c^time is at most 1 + w3 L_max/D < C_drop, its ê <= ē.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pcnme.config import Config
from pcnme.energy import EnergyModel

if TYPE_CHECKING:
    from pcnme.zone.zone import TaskRecord


@dataclass(frozen=True, slots=True)
class RewardParams:
    b: float
    w3: float
    c_drop: float
    w: float  # w1; w2 = 1 - w (QUESTIONS Q32 stand-in until M5)
    e_bar: float

    @classmethod
    def from_config(cls, cfg: Config) -> RewardParams:
        r = cfg.reward
        return cls(b=r.b, w3=r.w3, c_drop=r.c_drop, w=r.w, e_bar=cfg.derived.e_bar)


def c_time(latency: float, deadline: float, w3: float) -> float:
    """13.2: c^time = L/D + w3 max(0, L - D)/D of a delivered task."""
    return latency / deadline + w3 * max(0.0, latency - deadline) / deadline


def e_hat(energy: float, e_ref: float, e_bar: float) -> float:
    """12.3: ê = min(E / E_ref, ē)."""
    return min(energy / e_ref, e_bar)


def delivered(p: RewardParams, latency: float, deadline: float, energy_ratio: float) -> float:
    """11.4: the attempt the car kept, on time or (soft) late; energy_ratio = E / E_ref of that attempt."""
    return p.b - p.w * c_time(latency, deadline, p.w3) - (1 - p.w) * min(energy_ratio, p.e_bar)


def wasted(p: RewardParams, t_waste: float, deadline: float, energy_ratio: float) -> float:
    """11.4: refused, lost, cancelled or second: the anchor time it wasted and the energy it really spent."""
    return -(p.w * t_waste / deadline + (1 - p.w) * min(energy_ratio, p.e_bar))


def dropped(p: RewardParams) -> float:
    """11.4: the last attempt of a dropped task, and a task dropped with every option masked (7.3)."""
    return p.b - p.w * p.c_drop - (1 - p.w) * p.e_bar


@dataclass(frozen=True, slots=True)
class AttemptReward:
    """One attempt's reward, final at its task's close + G (11.3.3). kind: delivered (the copy the car kept),
    dropped (the last attempt of a dropped task) or wasted (refused, lost, cancelled or second). energy: what
    the attempt really spent by 12.5 (J; logged, and for a drop not in its reward); t_waste: the anchor's time
    from the send to its refusal, loss detection or cancel (wasted attempts only). decided: a placement
    decision made this attempt (11.3.1); a boulder's attempt and the soft fallback to the cloud (9.1) were
    rules, so they get a reward but no transition."""

    attempt: int
    kind: str
    reward: float
    energy: float
    t_waste: float | None
    decided: bool


def settle(rec: TaskRecord, anchor: str, p: RewardParams, em: EnergyModel) -> dict[int, AttemptReward]:
    """Every attempt of a closed task (11.4, 12.5), from the anchor's own record: the car's confirm (L and
    the attempt it kept), each attempt's send and end times, and the executors' reports that reached it by
    close + G. The last attempt is the kept one if the car confirmed, else the highest-numbered one."""
    task = rec.task
    e_ref = em.e_ref(task.task_type)
    confirm = rec.confirm if rec.confirm is not None and rec.confirm.anchor_id == anchor else None
    final = confirm.attempt if confirm is not None else max(rec.attempts, default=0)
    drop = rec.outcome not in ("on_time", "late")
    out: dict[int, AttemptReward] = {}
    for a, info in sorted(rec.attempts.items()):
        kept = confirm is not None and a == confirm.attempt
        energy = em.forwarding(em.destination(info.option), task.input_bits)
        if info.report is not None:
            energy += info.report.energy  # execution really spent; 0 if it never ran (12.5)
        if kept:  # 12.2: the return, once per publish (the executor's, then each republish)
            executor: Literal["fog", "cloud"] = "cloud" if info.option == "cloud" else "fog"
            energy += em.ret(executor, task.result_bits, 1 + rec.republishes)
        t_waste = None
        if a == final and drop:
            kind, r = "dropped", dropped(p)
        elif kept:
            assert confirm is not None
            kind, r = "delivered", delivered(p, confirm.latency, task.deadline, energy / e_ref)
        else:
            end = info.t_end if info.t_end is not None else rec.closed_at
            assert end is not None
            t_waste = end - info.t_send
            kind, r = "wasted", wasted(p, t_waste, task.deadline, energy / e_ref)
        out[a] = AttemptReward(a, kind, r, energy, t_waste, info.decided)
    return out
