"""The agent's 48 inputs (final-design 11.1; PLAN M4.1). THIS TOUCHES THE AGENT.

All built from the anchor's own memory (the DecisionContext: its exact state, its state store, its own pending
sends, its car registry and predictor), with no network call. Times are divided by the task's D and capped at
2 (= 1 + L_max/D: an option needing longer is masked anyway); energies are min(Ê_k / E_ref, ē) / ē.

    Task (5)         input_size, ref_exec, slack, hard, D_norm
    Per option (36)  for here, n1-n4, cloud in that order: present, wait, exec, network, energy, age
    Mobility (6)     exit (T_exit / D, capped at 1; a stopped car = 1); next_zone one-hot over 5:
                     stays (the anchor's zone), neighbour position 1-4
    Trade-off (1)    w (13.3; until M5 the config's stand-in, QUESTIONS Q32)

A masked option still gets its numbers (zeros where unknown, 11.1); which numbers is QUESTIONS Q33, built on
its stand-in: every number the anchor's memory can give, computed as for an open option (the network part
for any zone in the position; wait, exec and energy on q* from the snapshot it holds, with q* over every slot
type that can run the task when none has room); 0 where it holds nothing (no zone, no snapshot, no slot type
that can run the task). Unknown mobility (no GPS sample, or a predicted zone outside the 5 positions) gives
zeros (QUESTIONS Q34).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from pcnme.config import Config, SlotType
from pcnme.energy import EnergyModel
from pcnme.node.state import Pending, SlotState
from pcnme.zone.decision import OPTIONS, DecisionContext, Mobility, NeighbourView, OptionView
from pcnme.zone.masks import option_view

TIME_CAP = 2.0  # 11.1: time inputs are capped at 2 (= 1 + L_max/D)
EXIT_CAP = 1.0  # 11.1: exit = T_exit / D, capped at 1
TASK_INPUTS = ("input_size", "ref_exec", "slack", "hard", "D_norm")
OPTION_INPUTS = ("present", "wait", "exec", "network", "energy", "age")
MOBILITY_INPUTS = ("exit", "next_stays", "next_n1", "next_n2", "next_n3", "next_n4")
INPUT_NAMES: tuple[str, ...] = (
    *TASK_INPUTS,
    *(f"{o}_{f}" for o in OPTIONS for f in OPTION_INPUTS),
    *MOBILITY_INPUTS,
    "w",
)
N_INPUTS = 48
assert len(INPUT_NAMES) == N_INPUTS
_ABSENT = ("empty", "offline", "stale")  # 11.1: present = 0
__all__ = [
    "INPUT_NAMES",
    "N_INPUTS",
    "Decision",
    "Mobility",
    "Scales",
    "build_inputs",
    "next_zone_index",
    "numbers",
]


@dataclass(frozen=True, slots=True)
class Scales:
    """The fixed divisors of the task inputs, from the config: the largest input size and the largest deadline
    of any enabled task type, over both deadline modes (11.1)."""

    max_input_bits: float
    max_deadline: float
    reference_class: str
    stale_period: float  # 3 x the state period (11.1: age)

    @classmethod
    def from_config(cls, cfg: Config) -> Scales:
        types = cfg.task_types.values()
        return cls(
            max_input_bits=max(t.input_bits for t in types),
            max_deadline=max(d for t in types for d in t.deadline.values()),
            reference_class=cfg.energy.reference_class,
            stale_period=3 * cfg.stateplane.state_period,
        )


@dataclass(frozen=True, slots=True)
class Decision:
    """One placement decision at an anchor (11.3.1): the 48 inputs, each option's mask (None = open) and the
    action taken (its index in OPTIONS, 11.2). kind: the task's first decision, the one re-decision after a
    refusal or a missing reply (9.1), or a retry after a lost attempt (9.1)."""

    time: float
    anchor: str
    task_id: object
    attempt: int
    kind: str
    inputs: tuple[float, ...]
    masked: tuple[str | None, ...]
    action: int

    @property
    def valid(self) -> tuple[bool, ...]:
        """The actions the agent may take (11.3.7: the same masks as acting)."""
        return tuple(m is None for m in self.masked)


def _time(x: float) -> float:
    return min(x, TIME_CAP)


def numbers(cfg: Config, em: EnergyModel, ctx: DecisionContext, view: OptionView) -> OptionView:
    """The numbers of one option for the inputs (11.1): an open option or one masked "cannot meet" keeps the
    mask's own q* and estimates; any other masked option gets the stand-in numbers of QUESTIONS Q33."""
    if view.masked in (None, "cannot_meet", "empty"):
        return view
    est = ctx.estimates
    i = OPTIONS.index(view.option)
    states: Mapping[SlotType, SlotState] | None
    pending: Mapping[SlotType, tuple[Pending, ...]]
    if view.option == "here":
        hw, states, pending, h = ctx.here_hw, ctx.here, {}, 0
        t_in, t_ret = est.t_in_here(), est.t_ret_fog()
    elif view.option == "cloud":
        hw, states, pending, h = "cloud", ctx.cloud, ctx.cloud_pending, cfg.scheduler.headroom_h
        t_in, t_ret = est.t_in_cloud(ctx.task.input_bits, ctx.backhaul_wait), est.t_ret_cloud()
    else:
        n = ctx.neighbours[i - 1] if i - 1 < len(ctx.neighbours) else NeighbourView(None)
        hw, states, pending, h = n.hw, n.slots, n.pending, cfg.scheduler.headroom_h
        t_in, t_ret = est.t_in_neighbour(ctx.task.input_bits, ctx.backhaul_wait), est.t_ret_fog()
    path_only = OptionView(view.option, view.masked, t_in=t_in, t_ret=t_ret)
    if states is None or hw not in cfg.hardware:
        return path_only  # no snapshot held: only the path is known
    v = option_view(cfg, em, ctx, view.option, hw, states, pending, h, t_in, t_ret)
    if v.masked == "full":
        v = option_view(cfg, em, ctx, view.option, hw, states, pending, h, t_in, t_ret, ignore_room=True)
    if v.masked == "cannot_run":
        return path_only
    return replace(v, masked=view.masked)


def _option(
    view: OptionView, age: float | None, d: float, e_ref: float, e_bar: float, s: Scales
) -> list[float]:
    present = 0.0 if view.masked in _ABSENT else 1.0
    if view.masked == "empty":
        return [0.0] * len(OPTION_INPUTS)
    energy = min(view.energy / e_ref, e_bar) / e_bar
    age_in = 0.0 if age is None else _time(age / s.stale_period)
    return [
        present,
        _time(view.wait / d),
        _time(view.run / d),
        _time((view.t_in + view.t_ret) / d),
        energy,
        age_in,
    ]


def next_zone_index(target: str | None, anchor: str, positions: Sequence[str | None]) -> int | None:
    """0 = stays (the anchor's zone), 1-4 = that neighbour position; None if not one of the 5 (Q34)."""
    if target is None:
        return None
    if target == anchor:
        return 0
    return positions.index(target) + 1 if target in positions else None


def _mobility(m: Mobility | None, d: float, anchor: str, positions: Sequence[str | None]) -> list[float]:
    if m is None:
        return [0.0] * len(MOBILITY_INPUTS)  # no GPS sample: unknown (Q34)
    exit_in = EXIT_CAP if math.isinf(m.t_exit) else min(m.t_exit / d, EXIT_CAP)
    hot = [0.0] * 5
    k = next_zone_index(m.target, anchor, positions)
    if k is not None:
        hot[k] = 1.0
    return [exit_in, *hot]


def build_inputs(
    cfg: Config,
    em: EnergyModel,
    ctx: DecisionContext,
    scales: Scales,
    anchor: str,
    positions: Sequence[str | None],
) -> tuple[float, ...]:
    """The 48 inputs in the order of 11.1 (INPUT_NAMES). ctx.views must hold the 6 masks' views."""
    task = ctx.task
    d = task.deadline
    _, ref = cfg.time_table.fastest(task.task_type, scales.reference_class)
    out = [
        task.input_bits / scales.max_input_bits,
        _time(ref.mean / d),
        _time(ctx.time_left / d),
        1.0 if task.cls == "hard" else 0.0,
        d / scales.max_deadline,
    ]
    e_ref, e_bar = em.e_ref(task.task_type), cfg.derived.e_bar
    ages = (None, *(n.age for n in ctx.neighbours), *([None] * (4 - len(ctx.neighbours))), ctx.cloud_age)
    for view, age in zip(ctx.views, ages, strict=True):
        out += _option(numbers(cfg, em, ctx, view), age, d, e_ref, e_bar, scales)
    out += _mobility(ctx.mobility, d, anchor, positions)
    out.append(cfg.reward.w)
    return tuple(out)
