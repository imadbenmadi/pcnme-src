"""The masks and q* (final-design 7.2; PLAN M2.8, extended by M3.3 and M3.4).

For each of the 6 options (here, the 4 neighbour positions, the cloud) the anchor finds the usable slot
types U_k (c_q > 0, a time-table row, a place for the task's class as it knows it, pending sends included)
and assumes the executor picks the earliest finish, q* = argmin over U_k of (Ŵ_q for the task's class + Ê_q).
An option is masked when:
    empty         no zone in that neighbour position
    stale         its newest snapshot is older than 3 x the state period (or there is none)
    cannot run    no slot type there can run the task
    full          no usable slot type has room as the anchor knows it (pending sends included): n_q >= Q_q - h
                  (hard), >= Q_q - r - h (soft); h = 0 for the anchor's own exact queue; the cloud never fills
    cannot meet   hard: T̂_in + Ŵ + Ê + T̂_ret on q* > time left; soft: the same against time left + L_max
The anchor's own pending sends (R1) add their mean run to the backlog of their class on their q* (8.5).
M3.3 adds the cloud's state from the state store; M3.4 adds "offline" and the neighbours.
THIS TOUCHES THE AGENT: the masks decide which of its 6 actions it may take.
"""

from __future__ import annotations

from collections.abc import Mapping

from pcnme.config import SLOT_TYPES, Config, SlotType
from pcnme.energy import EnergyModel
from pcnme.node.state import Pending, SlotState, wait_estimate
from pcnme.zone.decision import OPTIONS, DecisionContext, OptionView, Reason

__all__ = ["OPTIONS", "OptionView", "Reason", "views"]


def _has_place(
    state: SlotState, cls: str, headroom: int, reserve: int, never_fills: bool, pending: int = 0
) -> bool:
    if never_fills:
        return True
    n = state.n_hard + state.n_soft + pending
    return n < state.capacity - headroom if cls == "hard" else n < state.capacity - reserve - headroom


def _view(
    cfg: Config,
    em: EnergyModel,
    ctx: DecisionContext,
    option: str,
    hw: str,
    states: Mapping[SlotType, SlotState],
    pending: Mapping[SlotType, tuple[Pending, ...]],
    headroom: int,
    t_in: float,
    t_ret: float,
) -> OptionView:
    task = ctx.task
    node = cfg.hardware[hw]
    rows = {
        q: r for q in SLOT_TYPES if node.slots(q) > 0 and (r := cfg.time_table.row(task.task_type, hw, q))
    }
    if not rows:
        return OptionView(option, "cannot_run")
    reserve = cfg.scheduler.reserve_hard_r
    usable = {
        q: r
        for q, r in rows.items()
        if _has_place(states[q], task.cls, headroom, reserve, node.is_cloud, len(pending.get(q, ())))
    }
    if not usable:
        return OptionView(option, "full")
    waits = {q: wait_estimate(states[q], task.cls, pending.get(q, ())) for q in usable}
    q_star = min(usable, key=lambda q: (waits[q] + usable[q].mean, SLOT_TYPES.index(q)))  # 7.2: q*
    run = usable[q_star].mean
    finish = t_in + waits[q_star] + run + t_ret
    limit = ctx.time_left if task.cls == "hard" else ctx.time_left + task.l_max
    destination = "cloud" if node.is_cloud else "here"
    energy = em.estimate(task.task_type, destination, hw, q_star, run)  # type: ignore[arg-type]
    masked: Reason | None = "cannot_meet" if finish > limit else None
    return OptionView(option, masked, q_star, waits[q_star], run, t_in, t_ret, energy)


def views(cfg: Config, em: EnergyModel, ctx: DecisionContext) -> list[OptionView]:
    """The 6 options in OPTIONS order (11.2). The neighbour positions are empty until M3.4."""
    est = ctx.estimates
    here = _view(cfg, em, ctx, "here", ctx.here_hw, ctx.here, {}, 0, est.t_in_here(), est.t_ret_fog())
    if ctx.cloud is None or ctx.cloud_stale:
        cloud = OptionView("cloud", "stale")
    else:
        cloud_t_in = est.t_in_cloud(ctx.task.input_bits, ctx.backhaul_wait)
        h = cfg.scheduler.headroom_h
        cloud = _view(
            cfg, em, ctx, "cloud", "cloud", ctx.cloud, ctx.cloud_pending, h, cloud_t_in, est.t_ret_cloud()
        )
    neighbours = [OptionView(o, "empty") for o in OPTIONS[1:5]]
    return [here, *neighbours, cloud]
