"""The masks and q* (final-design 7.2; PLAN M2.8, extended by M3.4).

For each of the 6 options (here, the 4 neighbour positions, the cloud) the anchor finds the usable slot
types U_k (c_q > 0, a time-table row, a place for the task's class as it knows it) and assumes the executor
picks the earliest finish, q* = argmin over U_k of (Ŵ_q for the task's class + Ê_q). An option is masked when:

    empty         no zone in that neighbour position
    cannot run    no slot type there can run the task
    full          no usable slot type has room: n_q >= Q_q - h (hard), >= Q_q - r - h (soft); h = 0 for the
                  anchor's own exact queue; the cloud never fills
    cannot meet   hard: T̂_in + Ŵ + Ê + T̂_ret on q* > time left; soft: the same against time left + L_max

M2 builds the rules that apply in one zone. "Offline" and "stale" need the state plane and come in M3.4.
THIS TOUCHES THE AGENT: the masks decide which of its 6 actions it may take.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pcnme.config import SLOT_TYPES, Config, SlotType
from pcnme.energy import EnergyModel
from pcnme.node.state import SlotState, wait_estimate
from pcnme.zone.zone import DecisionContext

OPTIONS: tuple[str, ...] = ("here", "n1", "n2", "n3", "n4", "cloud")  # the agent's 6 outputs (11.2)
Reason = Literal["empty", "cannot_run", "full", "cannot_meet"]


@dataclass(frozen=True, slots=True)
class OptionView:
    """One option as the anchor sees it: masked with a reason, or q* and its estimates."""

    option: str
    masked: Reason | None
    slot: SlotType | None = None  # q*
    wait: float = 0.0  # Ŵ on q*
    run: float = 0.0  # Ê on q*
    t_in: float = 0.0
    t_ret: float = 0.0
    energy: float = 0.0  # Ê_k (12.4)

    @property
    def finish(self) -> float:
        """The estimated finish at the car: T̂_in + Ŵ + Ê + T̂_ret."""
        return self.t_in + self.wait + self.run + self.t_ret


def _has_place(state: SlotState, cls: str, headroom: int, reserve: int, never_fills: bool) -> bool:
    if never_fills:
        return True
    n = state.n_hard + state.n_soft
    return n < state.capacity - headroom if cls == "hard" else n < state.capacity - reserve - headroom


def _view(
    cfg: Config,
    em: EnergyModel,
    ctx: DecisionContext,
    option: str,
    hw: str,
    states: dict[SlotType, SlotState],
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
    usable = {
        q: r
        for q, r in rows.items()
        if _has_place(states[q], task.cls, headroom, cfg.scheduler.reserve_hard_r, node.is_cloud)
    }
    if not usable:
        return OptionView(option, "full")
    waits = {q: wait_estimate(states[q], task.cls) for q in usable}
    q_star = min(usable, key=lambda q: (waits[q] + usable[q].mean, SLOT_TYPES.index(q)))  # 7.2: q*
    run = usable[q_star].mean
    finish = t_in + waits[q_star] + run + t_ret
    limit = ctx.time_left if task.cls == "hard" else ctx.time_left + task.l_max
    destination = "cloud" if node.is_cloud else "here"
    energy = em.estimate(task.task_type, destination, hw, q_star, run)  # type: ignore[arg-type]
    masked: Reason | None = "cannot_meet" if finish > limit else None
    return OptionView(option, masked, q_star, waits[q_star], run, t_in, t_ret, energy)


def views(cfg: Config, em: EnergyModel, ctx: DecisionContext) -> list[OptionView]:
    """The 6 options in order (11.2). One zone: the 4 neighbour positions are empty."""
    est = ctx.estimates
    here = _view(cfg, em, ctx, "here", ctx.here_hw, ctx.here, 0, est.t_in_here(), est.t_ret_fog())
    cloud_t_in = est.t_in_cloud(ctx.task.input_bits, ctx.backhaul_wait)
    cloud = _view(
        cfg, em, ctx, "cloud", "cloud", ctx.cloud, cfg.scheduler.headroom_h, cloud_t_in, est.t_ret_cloud()
    )
    neighbours = [OptionView(f"n{i}", "empty") for i in range(1, 5)]
    return [here, *neighbours, cloud]
