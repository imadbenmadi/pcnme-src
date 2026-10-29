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
class Mobility:
    """The predictor's output for one car (10.2), as the anchor holds it.

    t_exit: seconds until the car leaves its current zone going straight (infinite when stopped).
    target: the zone the car is predicted to be in next: the next zone when it moves, its current zone when
    stopped (QUESTIONS Q34 reading).
    """

    t_exit: float
    target: str | None


@dataclass(frozen=True, slots=True)
class NeighbourView:
    """What the anchor knows of the zone in one neighbour position (7.2, 7.5): None for an empty position."""

    zone_id: str | None
    hw: str = ""  # its hardware class, from the deployment configuration (QUESTIONS Q21)
    slots: Mapping[SlotType, SlotState] | None = None  # its newest snapshot (None: none yet)
    pending: Mapping[SlotType, tuple[Pending, ...]] = field(default_factory=lambda: _NO_PENDING)
    offline: bool = False  # its last will said offline
    stale: bool = False
    age: float | None = None  # its snapshot's age on the anchor's clock (5.4), None if none held


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
    cloud_offline: bool = False
    neighbours: tuple[NeighbourView, ...] = ()  # the 4 positions in order; fewer = the rest empty
    cloud_age: float | None = None  # the cloud snapshot's age (5.4), None if none held
    mobility: Mobility | None = None  # the predictor's output for the task's car, None without a GPS sample
    views: tuple[OptionView, ...] = ()  # filled by the anchor from the masks, in OPTIONS order
    inputs: tuple[float, ...] = ()  # the agent's 48 inputs (11.1), filled by the anchor after the views

    @property
    def time_left(self) -> float:
        return self.d_a - self.now


class Placement(Protocol):
    """A placement method: an option, or None when every option is masked (7.3)."""

    def choose(self, ctx: DecisionContext) -> Option | None: ...
