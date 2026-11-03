"""The MQTT adapter (PLAN M9.1 step 3; final-design 5.1, 5.2, 9.6, 15.2): the fast-mode bus's interface
(connect -> a client with subscribe, unsubscribe, count, publish, alive) on a real broker (EMQX) via paho.

- Subscription counts per topic as in fast mode: only 0 -> 1 and 1 -> 0 reach the broker (9.6).
- Every message travels in an envelope {"p": publisher, "k": key, "d": payload} (testbed/codec.py), so the
  receiving zone sees the same Delivery as in fast mode; the key names the message, never a clock (5.4).
- QoS 0 or 1, retained, and the MQTT 5 message expiry in whole seconds rounded up (9.6); the last will is
  retained QoS 1 (7.2, 9.1).
- paho's network thread never runs zone code: each delivery is handed to the loop thread (RealClock).
- Link delays are real (netem shapes them, 15.4): `delay` is accepted for the interface and unused.
- is_alive: this process's own client; a peer counts as alive here, and a dead peer shows as an HTTP call
  that never answers (the caller's timers, 9.3) and as its retained offline status (the last will).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from pcnme.bus.bus import Delivery, Key, QoS
from pcnme.testbed.clock import RealClock
from pcnme.testbed.codec import decode, encode


def envelope(publisher: str, key: Key, payload: object) -> bytes:
    return encode({"p": publisher, "k": tuple(key), "d": payload})


def _paho(client_id: str) -> Any:
    import paho.mqtt.client as mqtt

    return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, protocol=mqtt.MQTTv5)


class MqttClient:
    def __init__(self, bus: MqttBus, client_id: str, paho: Any) -> None:
        self._bus = bus
        self.client_id = client_id
        self._paho = paho
        self._counts: dict[str, int] = {}

    @property
    def alive(self) -> bool:
        return self._bus.is_alive(self.client_id)

    def subscribe(self, topic: str) -> int:
        n = self._counts.get(topic, 0) + 1
        self._counts[topic] = n
        if n == 1 and self.alive:
            self._paho.subscribe(topic, qos=1)
        return n

    def unsubscribe(self, topic: str) -> int:
        n = self._counts.get(topic, 0) - 1
        if n < 0:
            raise ValueError(f"{self.client_id} unsubscribed {topic} more often than it subscribed")
        self._counts[topic] = n
        if n == 0 and self.alive:
            self._paho.unsubscribe(topic)
        return n

    def count(self, topic: str) -> int:
        return self._counts.get(topic, 0)

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
        if not self.alive:
            return
        props = None
        if expiry is not None:
            from paho.mqtt.packettypes import PacketTypes
            from paho.mqtt.properties import Properties

            props = Properties(PacketTypes.PUBLISH)
            props.MessageExpiryInterval = max(1, math.ceil(expiry))
        self._paho.publish(
            topic, envelope(self.client_id, key, payload), qos=qos, retain=retained, properties=props
        )


class MqttBus:
    """One process's connection to the broker (one client per zone, car app or cloud, 3)."""

    def __init__(
        self,
        clock: RealClock,
        host: str,
        port: int = 1883,
        paho_factory: Callable[[str], Any] = _paho,
        *,
        connect_wait: float = 1.0,
        connect_timeout: float = 60.0,
    ) -> None:
        self._wait, self._timeout = connect_wait, connect_timeout
        self._clock = clock
        self._host, self._port = host, port
        self._factory = paho_factory
        self._clients: dict[str, MqttClient] = {}
        self._dead: set[str] = set()
        self.published = 0
        self.delivered = 0

    def connect(
        self,
        client_id: str,
        delay: tuple[float, float],
        on_message: Callable[[Delivery], None],
        will: tuple[str, object] | None = None,
    ) -> MqttClient:
        paho = self._factory(client_id)
        if will is not None:
            paho.will_set(will[0], envelope(client_id, ("will",), will[1]), qos=1, retain=True)

        def received(_c: Any, _u: Any, msg: Any) -> None:  # paho's thread: hand over, run nothing here
            env = decode(msg.payload)
            assert isinstance(env, dict)
            d = Delivery(
                msg.topic, env["d"], env["p"], msg.qos, tuple(env["k"]), bool(msg.retain), bool(msg.dup)
            )
            self._clock.call_soon(lambda: self._deliver(client_id, on_message, d))

        paho.on_message = received
        self._first_connect(paho)
        paho.loop_start()
        client = MqttClient(self, client_id, paho)
        self._clients[client_id] = client
        self._dead.discard(client_id)
        return client

    def _first_connect(self, paho: Any) -> None:
        """The broker may still be starting (EMQX takes some seconds): retry the first connect for up to
        `connect_timeout`; after it, paho's loop reconnects by itself (a broker restart, 15.7)."""
        import time

        end = time.monotonic() + self._timeout
        while True:
            try:
                paho.connect(self._host, self._port)
                return
            except OSError:
                if time.monotonic() >= end:
                    raise
                time.sleep(self._wait)

    def _deliver(self, client_id: str, on_message: Callable[[Delivery], None], d: Delivery) -> None:
        if self.is_alive(client_id):
            self.delivered += 1
            on_message(d)

    def is_alive(self, client_id: str) -> bool:
        return client_id not in self._dead

    def kill(self, client_id: str) -> None:
        """Drop the connection without a clean disconnect: the broker publishes the last will."""
        c = self._clients.get(client_id)
        if c is None or client_id in self._dead:
            return
        self._dead.add(client_id)
        c._paho.loop_stop()
