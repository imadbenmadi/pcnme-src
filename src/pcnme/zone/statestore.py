"""A zone's state store: its neighbours' and the cloud's newest snapshots, its own pending sends, and the WAN-
wait estimate (final-design 3, 4.3, 5.2, 5.4, 9.2; PLAN M3.3).

- Snapshots come by two paths: zone/{id}/state every state period, and every POST /tasks reply (R1). The
  store keeps, per zone, the one with the highest version (epoch first, 5.5) and its local receive time.
- Age = now - local receive time + δ̂_pub, on this zone's clock only (5.4); stale when older than
  stale_after (7.2). δ̂_pub = the mean delay of a message from that zone to this one: the publisher's mean
  link delay + broker processing + this zone's mean link delay (QUESTIONS Q20).
- R1 pending sends: a task sent but not answered counts in the anchor's copy of that executor, its mean run
  added to the backlog of its class on its q*, until the reply arrives or the upload times out.
- ŵ_WAN (4.3): after each cloud upload, sample = (accept reply - send start) - (own backhaul wait + δ̄_bh) -
  S / B_WAN - δ̄_WAN - δ̂_reply, then ŵ <- (1 - λ) ŵ + λ sample. Each zone publishes its ŵ_WAN and the age
  of its last sample; a zone uses the freshest estimate among its own and its neighbours'. Before any
  sample, 0.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pcnme.config import SlotType, TaskClass
from pcnme.ids import SnapshotVersion
from pcnme.node.state import Pending, SlotState
from pcnme.sim.clock import EventClock

PendingKey = tuple[object, ...]  # (task_id, attempt) at this anchor (5.5)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """A zone's (or the cloud's) published state (5.2 zone/{id}/state, and the snapshot in a POST reply)."""

    zone_id: str
    version: SnapshotVersion
    slots: Mapping[SlotType, SlotState]
    wan_wait: float  # its ŵ_WAN (0 for the cloud: it cannot see the router's queue, 4.3)
    wan_age: float | None  # age of its last WAN sample when it sent this, None before any
    load: float  # busy share of its slots (logs; QUESTIONS Q20)


@dataclass(slots=True)
class _Held:
    snapshot: Snapshot
    received: float


@dataclass(slots=True)
class _Wan:
    estimate: float = 0.0
    last_sample: float | None = None  # when this zone took its last sample


@dataclass(slots=True)
class StateStore:
    """One zone's state store (3)."""

    zone_id: str
    clock: EventClock
    stale_after: float
    pub_delay: Mapping[str, float]  # δ̂_pub per zone id (Q20)
    wan_lambda: float
    held: dict[str, _Held] = field(default_factory=dict)
    pending: dict[str, dict[PendingKey, tuple[SlotType, Pending]]] = field(default_factory=dict)
    wan: _Wan = field(default_factory=_Wan)

    # --- snapshots (R1, 5.5) --------------------------------------------------------------------------------

    def on_snapshot(self, snap: Snapshot) -> bool:
        """Keep it if its version is newer than the one held (epoch first). True if kept."""
        held = self.held.get(snap.zone_id)
        if held is not None and snap.version <= held.snapshot.version:
            return False
        self.held[snap.zone_id] = _Held(snap, self.clock.now)
        return True

    def snapshot(self, zone_id: str) -> Snapshot | None:
        held = self.held.get(zone_id)
        return None if held is None else held.snapshot

    def age(self, zone_id: str) -> float | None:
        """now - local receive time + δ̂_pub (5.4); None if nothing was ever received."""
        held = self.held.get(zone_id)
        if held is None:
            return None
        return self.clock.now - held.received + self.pub_delay[zone_id]

    def is_stale(self, zone_id: str) -> bool:
        """7.2: the newest snapshot is older than stale_after (or there is none)."""
        age = self.age(zone_id)
        return age is None or age > self.stale_after

    # --- pending sends (R1) ---------------------------------------------------------------------------------

    def add_pending(self, zone_id: str, key: PendingKey, slot: SlotType, cls: TaskClass, mean: float) -> None:
        self.pending.setdefault(zone_id, {})[key] = (slot, Pending(cls, mean))

    def clear_pending(self, zone_id: str, key: PendingKey) -> None:
        self.pending.get(zone_id, {}).pop(key, None)

    def pending_on(self, zone_id: str, slot: SlotType) -> tuple[Pending, ...]:
        return tuple(p for q, p in self.pending.get(zone_id, {}).values() if q == slot)

    # --- the WAN wait (4.3) ---------------------------------------------------------------------------------

    def wan_sample(self, sample: float) -> None:
        """One measurement after a cloud upload: the moving average, λ from the config."""
        w = self.wan
        w.estimate = (1 - self.wan_lambda) * w.estimate + self.wan_lambda * sample
        w.last_sample = self.clock.now

    def own_wan_age(self) -> float | None:
        return None if self.wan.last_sample is None else self.clock.now - self.wan.last_sample

    def estimate(self) -> float:
        """ŵ_WAN: the freshest among this zone's own and its neighbours' published ones; 0 before any (4.3).
        A neighbour's estimate is as old as its sample was when sent, plus the snapshot's age."""
        best: tuple[float, float] | None = None  # (age, estimate)
        own = self.own_wan_age()
        if own is not None:
            best = (own, self.wan.estimate)
        for zone_id, held in self.held.items():
            s = held.snapshot
            if s.wan_age is None:
                continue
            age = s.wan_age + (self.age(zone_id) or 0.0)
            if best is None or age < best[0]:
                best = (age, s.wan_wait)
        return 0.0 if best is None else best[1]
