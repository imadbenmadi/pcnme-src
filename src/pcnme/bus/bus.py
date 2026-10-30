"""The fast-mode bus: an in-process topic table with the modelled broker delay (final-design 5.1, 5.2, 9.6,
15.2).

One broker (the EMQX cluster at the metro edge). A message published at t reaches the broker after the
publisher's link delay plus the broker processing (18: 1 ms on top of the backhaul delay), and each subscriber
after its own link delay from there (4.2: MQTT messages pay the link's delay only, no bandwidth). Zones sit on
the backhaul, the cloud on the WAN. Every delay is a keyed draw on the link_delays stream (CHANGES C3): the
publisher names the message with a key (for a task message its task id and attempt), and the bus adds the hop
and the client, so a delay never depends on the order of events. Delays are whole nanoseconds (15.2).

- Subscriptions are counted per (client, topic): one MQTT client serves all of a zone's roles, so it
  subscribes at 0 -> 1 and unsubscribes at 1 -> 0 (9.6). A (un)subscribe takes effect at the broker after
  the client's link delay; a message is routed to the subscribers the broker has when it gets the message.
- Retained: the broker keeps the last retained message per topic, until its expiry (MQTT 5 message expiry,
  9.6), and sends it to every new subscriber.
- Last will: a client that dies stops sending and receiving, and the broker publishes its will at once
  (7.2, 9.1: "immediate").
- QoS 1 means at least once: a message may arrive twice (9.5). The design gives no rate for duplicates or
  losses, so a normal run has none (QUESTIONS Q19); `Faults` injects them in tests.
No wildcards (9.6): topics are exact strings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pcnme.sim.clock import EventClock
from pcnme.sim.quantize import to_ns
from pcnme.sim.rng import KeyPart, RandomStreams

QoS = Literal[0, 1]
Key = tuple[KeyPart, ...]

STREAM = "link_delays"  # MQTT messages cross backhaul or WAN links (Q5)


@dataclass(frozen=True, slots=True)
class Delivery:
    """One message as a subscriber receives it."""

    topic: str
    payload: object
    publisher: str
    qos: QoS
    key: Key
    retained: bool = False  # a retained copy sent on subscribe
    duplicate: bool = False  # a QoS 1 repeat


@dataclass(slots=True)
class Faults:
    """Test hooks only (QUESTIONS Q19): `drop(topic, key, subscriber)` loses one delivery; `duplicate(topic,
    key, subscriber)` sends a QoS 1 delivery a second time."""

    drop: Callable[[str, Key, str], bool] | None = None
    duplicate: Callable[[str, Key, str], bool] | None = None


@dataclass(slots=True)
class _Retained:
    payload: object
    publisher: str
    qos: QoS
    key: Key
    seq: int  # the broker's count for this topic: an expiry only removes the copy it was set for


@dataclass(slots=True)
class _Client:
    client_id: str
    delay: tuple[float, float]  # its link to the broker: (low, high) seconds
    on_message: Callable[[Delivery], None]
    will: tuple[str, object] | None
    alive: bool = True
    counts: dict[str, int] = field(default_factory=dict)  # the client's own reference count per topic
    ops: int = 0  # (un)subscribe operations, to key their delays


class Bus:
    """The broker and its clients on one simulated clock."""

    def __init__(self, clock: EventClock, streams: RandomStreams, broker_delay: float) -> None:
        self._clock = clock
        self._streams = streams
        self._broker_delay = broker_delay
        self._clients: dict[str, _Client] = {}
        self._subscribers: dict[str, list[str]] = {}  # topic -> client ids, in subscription order
        self._retained: dict[str, _Retained] = {}
        self._retained_seq = 0
        self.faults = Faults()
        self.delivered = 0  # deliveries made (broker load, 9.2)
        self.published = 0  # messages that reached the broker (9.2: broker messages per second)

    # --- clients --------------------------------------------------------------------------------------------

    def connect(
        self,
        client_id: str,
        delay: tuple[float, float],
        on_message: Callable[[Delivery], None],
        will: tuple[str, object] | None = None,
    ) -> Client:
        """Connect with a clean start (9.6): no subscriptions survive from before."""
        old = self._clients.get(client_id)
        if old is not None and old.alive:
            raise ValueError(f"client {client_id} is already connected")
        for subs in self._subscribers.values():
            if client_id in subs:
                subs.remove(client_id)
        self._clients[client_id] = _Client(client_id, delay, on_message, will)
        return Client(self, client_id)

    def kill(self, client_id: str) -> None:
        """The client dies: it stops sending and receiving, and the broker publishes its last will at once."""
        c = self._clients[client_id]
        if not c.alive:
            return
        c.alive = False
        if c.will is not None:
            topic, payload = c.will
            self._at_broker(
                self._clock.now, topic, payload, client_id, 1, ("will",), retained=True, expiry=None
            )

    def is_alive(self, client_id: str) -> bool:
        c = self._clients.get(client_id)
        return c is not None and c.alive

    def subscribers(self, topic: str) -> tuple[str, ...]:
        """The clients the broker routes `topic` to right now."""
        return tuple(self._subscribers.get(topic, ()))

    # --- the delay model ------------------------------------------------------------------------------------

    def _delay(self, client: _Client, key: Key, hop: str) -> float:
        low, high = client.delay
        u = self._streams.uniform(STREAM, *key, hop, client.client_id)
        return to_ns(low + u * (high - low))

    # --- publish and route ----------------------------------------------------------------------------------

    def _publish(
        self,
        client_id: str,
        topic: str,
        payload: object,
        qos: QoS,
        key: Key,
        retained: bool,
        expiry: float | None,
    ) -> None:
        c = self._clients[client_id]
        if not c.alive:
            return  # a dead client sends nothing
        at = self._clock.now + self._delay(c, key, "pub") + self._broker_delay
        self._clock.schedule(
            at, lambda: self._at_broker(at, topic, payload, client_id, qos, key, retained, expiry)
        )

    def _at_broker(
        self,
        at: float,
        topic: str,
        payload: object,
        publisher: str,
        qos: QoS,
        key: Key,
        retained: bool,
        expiry: float | None,
    ) -> None:
        self.published += 1
        if retained:
            self._retained_seq += 1
            seq = self._retained_seq
            self._retained[topic] = _Retained(payload, publisher, qos, key, seq)
            if expiry is not None:

                def expire() -> None:
                    kept = self._retained.get(topic)
                    if kept is not None and kept.seq == seq:
                        del self._retained[topic]

                self._clock.schedule(at + expiry, expire)
        for sub in list(self._subscribers.get(topic, ())):
            self._send(at, sub, Delivery(topic, payload, publisher, qos, key))

    def _send(self, at: float, subscriber: str, d: Delivery) -> None:
        c = self._clients[subscriber]
        f = self.faults
        if f.drop is not None and f.drop(d.topic, d.key, subscriber):
            return
        hop = "retained" if d.retained else "to"
        self._clock.schedule(at + self._delay(c, d.key, hop), lambda: self._receive(subscriber, d))
        if d.qos == 1 and f.duplicate is not None and f.duplicate(d.topic, d.key, subscriber):
            dup = Delivery(d.topic, d.payload, d.publisher, d.qos, d.key, d.retained, duplicate=True)
            self._clock.schedule(at + self._delay(c, d.key, "dup"), lambda: self._receive(subscriber, dup))

    def _receive(self, subscriber: str, d: Delivery) -> None:
        c = self._clients[subscriber]
        if c.alive:
            self.delivered += 1
            c.on_message(d)

    # --- subscriptions --------------------------------------------------------------------------------------

    def _change(self, client_id: str, topic: str, step: int) -> int:
        c = self._clients[client_id]
        if not c.alive:
            return c.counts.get(topic, 0)
        before = c.counts.get(topic, 0)
        after = before + step
        if after < 0:
            raise ValueError(f"{client_id} unsubscribed {topic} more often than it subscribed")
        c.counts[topic] = after
        if (before, after) in ((0, 1), (1, 0)):  # 9.6: only 0 -> 1 and 1 -> 0 reach the broker
            c.ops += 1
            key: Key = (client_id, topic, c.ops)
            at = self._clock.now + self._delay(c, key, "sub")
            self._clock.schedule(at, lambda: self._at_broker_sub(at, client_id, topic, after == 1, key))
        return after

    def _at_broker_sub(self, at: float, client_id: str, topic: str, on: bool, key: Key) -> None:
        subs = self._subscribers.setdefault(topic, [])
        c = self._clients[client_id]
        if on and c.alive and c.counts.get(topic, 0) > 0 and client_id not in subs:
            subs.append(client_id)
            kept = self._retained.get(topic)
            if kept is not None:
                d = Delivery(topic, kept.payload, kept.publisher, kept.qos, (*kept.key, *key), retained=True)
                self._send(at, client_id, d)
        elif not on and c.counts.get(topic, 0) == 0 and client_id in subs:
            subs.remove(client_id)


class Client:
    """One MQTT client (one per zone and one for the cloud, 3)."""

    def __init__(self, bus: Bus, client_id: str) -> None:
        self._bus = bus
        self.client_id = client_id

    @property
    def alive(self) -> bool:
        return self._bus.is_alive(self.client_id)

    def subscribe(self, topic: str) -> int:
        """Raise this client's count on `topic`; the broker learns of it at 0 -> 1. Returns the new count."""
        return self._bus._change(self.client_id, topic, +1)

    def unsubscribe(self, topic: str) -> int:
        """Lower the count; the broker learns of it at 1 -> 0. Returns the new count."""
        return self._bus._change(self.client_id, topic, -1)

    def count(self, topic: str) -> int:
        return self._bus._clients[self.client_id].counts.get(topic, 0)

    def publish(
        self,
        topic: str,
        payload: object,
        *,
        qos: QoS,
        key: Key,
        retained: bool = False,
        expiry: float | None = None,
    ) -> None:
        """Publish now. `key` names the message for its delay draws (a task message: task id, attempt)."""
        self._bus._publish(self.client_id, topic, payload, qos, key, retained, expiry)
