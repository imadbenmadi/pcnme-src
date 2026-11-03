"""The MQTT adapter behind the bus's client interface (PLAN M9.1 step 3; final-design 5.1, 5.2, 9.6), with a
fake paho client: subscription counts (0 -> 1 and 1 -> 0 reach the broker), QoS, retained, expiry, the last
will, and deliveries handed to the loop thread."""

from __future__ import annotations

from typing import Any

from pcnme.bus.bus import Delivery
from pcnme.testbed.clock import RealClock
from pcnme.testbed.codec import decode
from pcnme.testbed.mqtt import MqttBus, envelope
from pcnme.zone.zone import ZoneStatus


class FakePaho:
    """Records what the adapter asks of paho; `deliver` plays a message from the broker."""

    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        self.calls: list[tuple[Any, ...]] = []
        self.will: tuple[Any, ...] | None = None
        self.on_message: Any = None

    def will_set(self, topic: str, payload: bytes, qos: int, retain: bool) -> None:
        self.will = (topic, payload, qos, retain)

    def connect(self, host: str, port: int) -> None:
        self.calls.append(("connect", host, port))

    def loop_start(self) -> None:
        self.calls.append(("loop_start",))

    def loop_stop(self) -> None:
        self.calls.append(("loop_stop",))

    def disconnect(self) -> None:
        self.calls.append(("disconnect",))

    def subscribe(self, topic: str, qos: int) -> None:
        self.calls.append(("subscribe", topic, qos))

    def unsubscribe(self, topic: str) -> None:
        self.calls.append(("unsubscribe", topic))

    def publish(self, topic: str, payload: bytes, qos: int, retain: bool, properties: Any = None) -> None:
        self.calls.append(("publish", topic, payload, qos, retain, properties))

    def deliver(self, topic: str, payload: bytes, retain: bool = False, dup: bool = False) -> None:
        msg = type("M", (), {"topic": topic, "payload": payload, "retain": retain, "dup": dup, "qos": 1})()
        self.on_message(self, None, msg)


def make(
    clock: RealClock, got: list[Delivery], will: tuple[str, object] | None = None
) -> tuple[Any, FakePaho]:
    fakes: list[FakePaho] = []

    def factory(client_id: str) -> FakePaho:
        fakes.append(FakePaho(client_id))
        return fakes[-1]

    bus = MqttBus(clock, "broker", 1883, paho_factory=factory)
    client = bus.connect("z1", (0.0, 0.0), got.append, will=will)
    return client, fakes[0]


def test_connect_sets_the_last_will_and_starts_the_network_thread() -> None:
    will = ZoneStatus("z1", "offline", 1.0, 2.0, 350.0)
    _, paho = make(RealClock(), [], ("zone/z1/status", will))
    topic, payload, qos, retain = paho.will  # type: ignore[misc]
    assert (topic, qos, retain) == ("zone/z1/status", 1, True)
    assert decode(payload) == {"p": "z1", "k": ("will",), "d": will}
    assert paho.calls[:2] == [("connect", "broker", 1883), ("loop_start",)]


def test_subscriptions_are_counted_and_only_edges_reach_the_broker() -> None:
    client, paho = make(RealClock(), [])
    assert client.subscribe("car/c1/result") == 1 and client.subscribe("car/c1/result") == 2
    assert client.unsubscribe("car/c1/result") == 1 and client.unsubscribe("car/c1/result") == 0
    edges = [c for c in paho.calls if c[0] in ("subscribe", "unsubscribe")]
    assert edges == [("subscribe", "car/c1/result", 1), ("unsubscribe", "car/c1/result")]
    assert client.count("car/c1/result") == 0


def test_publish_carries_qos_retain_expiry_and_the_envelope() -> None:
    client, paho = make(RealClock(), [])
    client.publish("zone/z1/state", {"x": 1}, qos=0, key=("z1", "state", 3), retained=True, expiry=2.5)
    _, topic, payload, qos, retain, props = paho.calls[-1]
    assert (topic, qos, retain) == (
        "zone/z1/state",
        0,
        True,
    ) and props.MessageExpiryInterval == 3  # whole s, up
    assert decode(payload) == {"p": "z1", "k": ("z1", "state", 3), "d": {"x": 1}}
    client.publish("car/c1/ack", "a", qos=1, key=("k",))
    assert paho.calls[-1][5] is None  # no expiry: no properties


def test_a_message_reaches_the_zone_on_the_loop_thread() -> None:
    clock = RealClock()
    got: list[Delivery] = []
    _, paho = make(clock, got)
    paho.deliver("car/c1/result", envelope("z2", ("t", 1), [1, 2]), retain=True, dup=True)
    assert got == []  # handed to the loop, not run on paho's thread
    clock.run_for(0.05)
    (d,) = got
    assert (d.topic, d.payload, d.publisher, d.key, d.qos, d.retained, d.duplicate) == (
        "car/c1/result",
        [1, 2],
        "z2",
        ("t", 1),
        1,
        True,
        True,
    )


def test_killing_a_client_stops_it() -> None:
    clock = RealClock()
    bus = MqttBus(clock, "broker", 1883, paho_factory=FakePaho)
    client = bus.connect("z1", (0.0, 0.0), lambda d: None)
    assert client.alive and bus.is_alive("z1") and bus.is_alive("z2")  # peers: HTTP answers decide (step 4)
    bus.kill("z1")
    assert not client.alive and not bus.is_alive("z1")


def test_the_first_connect_waits_for_a_broker_that_is_still_starting() -> None:
    refusals = [2]

    class Slow(FakePaho):
        def connect(self, host: str, port: int) -> None:
            if refusals[0]:
                refusals[0] -= 1
                raise ConnectionRefusedError
            super().connect(host, port)

    bus = MqttBus(RealClock(), "broker", paho_factory=Slow, connect_wait=0.01)
    client = bus.connect("z1", (0.0, 0.0), lambda d: None)
    assert client.alive and refusals == [0]
