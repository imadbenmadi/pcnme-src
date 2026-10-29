"""Who is a neighbour (final-design 7.5, 18 Layout; PLAN M3.2).

1. Candidates: every zone whose coverage circle overlaps this one (centre distance < 2 R_z) or whose centre
   is within d_nb; the cloud is never a neighbour position.
2. Up to 4 positions, nearest first, fixed at start-up (equal distances: by zone id, an engineering choice).
   A zone that joins takes the lowest free position of each zone that counts it as a candidate; a zone with
   4 neighbours keeps them. Positions never change during a run.
3. An offline neighbour keeps its position, masked, until it is online again.
4. Not symmetric: A may count B while B does not count A.

A zone learns the others' circles from their retained zone/{id}/status messages (5.2). The zones active at
the start come from the deployment configuration (18: z5 joins later), so start-up positions are fixed from
them at time 0 (an engineering choice: the design does not say how long start-up lasts); every zone that
announces itself online later joins by rule 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

POSITIONS = 4  # the agent's 4 neighbour outputs (11.2)
CLOUD = "cloud"  # 5.5: the cloud's id; never a neighbour position (7.5.1)


@dataclass(frozen=True, slots=True)
class Site:
    """A zone's coverage circle, as its status message announces it (5.2: x, y, radius)."""

    zone_id: str
    x: float
    y: float
    radius: float

    def distance(self, other: Site) -> float:
        return math.dist((self.x, self.y), (other.x, other.y))


def is_candidate(me: Site, other: Site, d_nb: float) -> bool:
    """7.5.1: the circles overlap, or the centre is within d_nb."""
    if other.zone_id in (me.zone_id, CLOUD):
        return False
    d = me.distance(other)
    return d < me.radius + other.radius or d <= d_nb


@dataclass(slots=True)
class NeighbourTable:
    """One zone's 4 neighbour positions (7.5) and what it knows of each neighbour's status."""

    me: Site
    d_nb: float
    positions: list[str | None] = field(default_factory=lambda: [None] * POSITIONS)
    sites: dict[str, Site] = field(default_factory=dict)
    online: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def at_start(cls, me: Site, others: list[Site], d_nb: float) -> NeighbourTable:
        """7.5.2: the candidates among the zones active at start-up, nearest first, up to 4."""
        table = cls(me, d_nb)
        found = sorted(
            (s for s in others if is_candidate(me, s, d_nb)), key=lambda s: (me.distance(s), s.zone_id)
        )
        for i, s in enumerate(found[:POSITIONS]):
            table.positions[i] = s.zone_id
        for s in others:
            table.sites[s.zone_id] = s
            table.online[s.zone_id] = True
        return table

    def permute(self, order: list[int]) -> None:
        """13.6, training only: position k takes the zone that was in position order[k], for one episode
        (empty positions move too; next_zone follows, since it is read from the positions)."""
        if sorted(order) != list(range(POSITIONS)):
            raise ValueError(f"not a permutation of the {POSITIONS} positions: {order}")
        self.positions = [self.positions[i] for i in order]

    def on_status(self, site: Site, online: bool) -> None:
        """A zone/{id}/status message: a zone new to this table joins (7.5.2); a known one goes offline or
        comes back, keeping its position either way (7.5.3)."""
        if site.zone_id == self.me.zone_id:
            return
        self.sites[site.zone_id] = site
        self.online[site.zone_id] = online
        if not online or site.zone_id in self.positions or not is_candidate(self.me, site, self.d_nb):
            return
        free = [i for i, z in enumerate(self.positions) if z is None]
        if free:  # a zone with 4 neighbours keeps them
            self.positions[free[0]] = site.zone_id

    def position(self, zone_id: str) -> int | None:
        """The position (0-3) of a neighbour, or None."""
        return self.positions.index(zone_id) if zone_id in self.positions else None

    def is_online(self, zone_id: str) -> bool:
        return self.online.get(zone_id, False)
