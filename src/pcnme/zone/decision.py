"""What a placement decision sees and returns (final-design 6B, 7.2, 7.3, 11.2).

The anchor builds the context, the masks turn it into one view per option (zone/masks.py), and a placement
method (Greedy, Random, later the agent) picks one unmasked option, or None when every option is masked.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol

from pcnme.car.car import Task
from pcnme.config import SlotType
from pcnme.node.state import Pending, SlotState
from pcnme.zone.estimates import PathEstimates

Option = Literal["here", "n1", "n2", "n3", "n4", "cloud"]
OPTIONS: tuple[Option, ...] = ("here", "n1", "n2", "n3", "n4", "cloud")  # the agent's 6 outputs (11.2)
Reason = Literal["empty", "offline", "stale", "cannot_run", "full", "cannot_meet"]

_NO_PENDING: Mapping[SlotType, tuple[Pending, ...]] = MappingProxyType({})


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


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """What the anchor knows when it places a pebble (6B): its own exact state, and the cloud's newest
    snapshot from its state store (None if it has none) with its own pending sends there (R1, 9.2)."""

    task: Task
    now: float
    d_a: float
    here: Mapping[SlotType, SlotState]
    cloud: Mapping[SlotType, SlotState] | None
    backhaul_wait: float  # the zone's own backhaul pipe, known exactly (4.3)
    estimates: PathEstimates
    here_hw: str = "F-GPU"  # this zone's hardware class
    cloud_pending: Mapping[SlotType, tuple[Pending, ...]] = field(default_factory=lambda: _NO_PENDING)
    cloud_stale: bool = False
    views: tuple[OptionView, ...] = ()  # filled by the anchor from the masks, in OPTIONS order

    @property
    def time_left(self) -> float:
        return self.d_a - self.now


class Placement(Protocol):
    """A placement method: an option, or None when every option is masked (7.3)."""

    def choose(self, ctx: DecisionContext) -> Option | None: ...
