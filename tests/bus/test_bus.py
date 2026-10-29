"""The fast-mode bus and calls (PLAN M3.1; final-design 5.1-5.3, 9.5, 9.6, 15.2; QUESTIONS Q19)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcnme.bus.bus import Bus, Delivery
from pcnme.bus.calls import call
from pcnme.config import Config, load_config
from pcnme.sim.clock import EventClock
from pcnme.sim.rng import RandomStreams

FIXED = (0.003, 0.003)  # a backhaul link fixed at its mean, so times are by hand
BROKER = 0.001


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


def make(cfg: Config) -> tuple[EventClock, Bus, dict[str, list[tuple[float, Delivery]]]]:
    clock = EventClock()
    bus = Bus(clock, RandomStreams.from_config(cfg, 2000, "training"), broker_delay=BROKER)
    return clock, bus, {}


def client(clock: EventClock, bus: Bus, got: dict, name: str, delay=FIXED, will=None):  # type: ignore[no-untyped-def]
    got[name] = []
    return bus.connect(name, delay, lambda d: got[name].append((clock.now, d)), will=will)


def test_a_message_pays_publisher_link_broker_and_subscriber_link(cfg: Config) -> None:
    # z1 publishes at 0.010: broker at 0.010 + 3 + 1 ms = 0.014; z2 gets it 3 ms later, at 0.017 (4.2, 18).
    clock, bus, got = make(cfg)
    z1, z2 = client(clock, bus, got, "z1"), client(clock, bus, got, "z2")
    z2.subscribe("zone/z1/state")
    clock.run_until(0.010)
    z1.publish("zone/z1/state", {"n": 1}, qos=0, key=("state", "z1", 1))
    clock.run_until(1.0)
    ((t, d),) = got["z2"]
    assert t == pytest.approx(0.017, abs=1e-12)
    assert (d.topic, d.payload, d.publisher, d.retained, d.duplicate) == (
        "zone/z1/state",
        {"n": 1},
        "z1",
        False,
        False,
    )
    assert got["z1"] == []  # not subscribed


def test_the_cloud_publishes_over_the_wan(cfg: Config) -> None:
    clock, bus, got = make(cfg)
    cloud = client(clock, bus, got, "cloud", delay=(0.0625, 0.0625))
    z1 = client(clock, bus, got, "z1")
    z1.subscribe("car/c01/result")
    clock.run_until(0.010)
    cloud.publish("car/c01/result", "r", qos=1, key=("c01", 0, 1, 1))
    clock.run_until(1.0)
    assert got["z1"][0][0] == pytest.approx(0.010 + 0.0625 + BROKER + 0.003, abs=1e-12)


def test_delays_are_keyed_draws_on_the_link_interval(cfg: Config) -> None:
    # 5 000 messages on U(1, 5) ms links: every delay inside its interval; the same key, the same delay.
    def run() -> list[float]:
        clock, bus, got = make(cfg)
        a, b = (
            client(clock, bus, got, "z1", delay=cfg.links.backhaul_delay),
            client(clock, bus, got, "z2", cfg.links.backhaul_delay),
        )
        b.subscribe("t")
        clock.run_until(0.1)
        for i in range(5000):
            a.publish("t", i, qos=0, key=("m", i))
        clock.run_until(10.0)
        return sorted(t - 0.1 for t, _ in got["z2"])

    first = run()
    assert len(first) == 5000
    assert min(first) >= 0.001 + BROKER + 0.001 and max(first) <= 0.005 + BROKER + 0.005
    assert first == run()


def test_reference_counts_per_client_and_topic(cfg: Config) -> None:
    # 9.6: one MQTT client serves every role; subscribe at 0 -> 1, unsubscribe at 1 -> 0 only.
    clock, bus, got = make(cfg)
    z1, z2 = client(clock, bus, got, "z1"), client(clock, bus, got, "z2")
    assert z2.subscribe("car/c01/gps") == 1
    assert z2.subscribe("car/c01/gps") == 2  # the anchor role and the serving role
    clock.run_until(0.01)
    assert bus.subscribers("car/c01/gps") == ("z2",)
    assert z2.unsubscribe("car/c01/gps") == 1
    clock.run_until(0.02)
    assert bus.subscribers("car/c01/gps") == ("z2",)  # still counted once
    z2.unsubscribe("car/c01/gps")
    clock.run_until(0.03)
    assert bus.subscribers("car/c01/gps") == ()
    z1.publish("car/c01/gps", 1, qos=0, key=("g", 1))
    clock.run_until(1.0)
    assert got["z2"] == []
    with pytest.raises(ValueError, match="more often"):
        z2.unsubscribe("car/c01/gps")


def test_a_subscription_takes_effect_when_it_reaches_the_broker(cfg: Config) -> None:
    # z2 subscribes at 0 (effective at 3 ms); z1's message reaches the broker at 4 ms: delivered. A message
    # at the broker at 2 ms is routed before the subscription arrives: not delivered.
    clock, bus, got = make(cfg)
    z1 = client(clock, bus, got, "z1", delay=(0.001, 0.001))
    z2 = client(clock, bus, got, "z2")
    z2.subscribe("t")
    z1.publish("t", "early", qos=1, key=("a",))  # broker at 1 + 1 = 2 ms
    clock.run_until(0.002)
    z1.publish("t", "late", qos=1, key=("b",))  # broker at 4 ms
    clock.run_until(1.0)
    assert [d.payload for _, d in got["z2"]] == ["late"]


def test_retained_goes_to_new_subscribers_until_it_expires(cfg: Config) -> None:
    # GPS: QoS 0, retained with an MQTT 5 expiry of 1 s (9.6, 18).
    clock, bus, got = make(cfg)
    z1, z2, z3 = (client(clock, bus, got, n) for n in ("z1", "z2", "z3"))
    z1.publish("car/c01/gps", "p1", qos=0, key=("g", 1), retained=True, expiry=1.0)
    z1.publish("car/c01/gps", "p2", qos=0, key=("g", 2), retained=True, expiry=1.0)  # replaces p1
    clock.run_until(0.5)
    z2.subscribe("car/c01/gps")  # at the broker at 0.503: gets p2, 3 ms later
    clock.run_until(1.2)
    assert [(round(t, 9), d.payload, d.retained) for t, d in got["z2"]] == [(0.506, "p2", True)]
    z3.subscribe("car/c01/gps")  # at the broker at 1.203 > 0.004 + 1 s: expired
    clock.run_until(2.0)
    assert got["z3"] == []


def test_an_expiry_does_not_remove_a_newer_retained_message(cfg: Config) -> None:
    clock, bus, got = make(cfg)
    z1, z2 = client(clock, bus, got, "z1"), client(clock, bus, got, "z2")
    z1.publish("s", "old", qos=0, key=("s", 1), retained=True, expiry=0.1)
    clock.run_until(0.05)
    z1.publish("s", "new", qos=0, key=("s", 2), retained=True, expiry=10.0)
    clock.run_until(0.5)  # the old copy's expiry (0.104) has passed
    z2.subscribe("s")
    clock.run_until(1.0)
    assert [d.payload for _, d in got["z2"]] == ["new"]


def test_the_last_will_is_published_at_once_and_the_dead_client_goes_silent(cfg: Config) -> None:
    # 7.2, 9.1: "its last will said offline", immediate. Retained, so a later subscriber sees it too.
    clock, bus, got = make(cfg)
    z1 = client(clock, bus, got, "z1", will=("zone/z1/status", "offline"))
    z2, z3 = client(clock, bus, got, "z2"), client(clock, bus, got, "z3")
    z2.subscribe("zone/z1/status")
    z2.subscribe("zone/z1/state")
    z1.subscribe("zone/z2/state")
    clock.run_until(0.1)
    bus.kill("z1")
    z1.publish("zone/z1/state", "ghost", qos=0, key=("x",))  # a dead client sends nothing
    z2.publish("zone/z2/state", "s", qos=0, key=("y",))  # and receives nothing
    clock.run_until(0.2)
    assert [(round(t, 9), d.payload) for t, d in got["z2"]] == [(0.103, "offline")]
    assert got["z1"] == []
    assert not z1.alive and not bus.is_alive("z1")
    z3.subscribe("zone/z1/status")
    clock.run_until(1.0)
    assert [d.payload for _, d in got["z3"]] == ["offline"]


def test_a_reconnect_starts_clean(cfg: Config) -> None:
    # 9.6: zones connect with a clean start, so a restarted zone has no stale subscriptions.
    clock, bus, got = make(cfg)
    client(clock, bus, got, "z1")
    z2 = client(clock, bus, got, "z2")
    z2.subscribe("t")
    clock.run_until(0.01)
    bus.kill("z2")
    z2b = client(clock, bus, got, "z2")
    assert z2b.count("t") == 0 and bus.subscribers("t") == ()
    with pytest.raises(ValueError, match="already connected"):
        client(clock, bus, got, "z1")


def test_qos1_may_deliver_twice_and_qos0_never_does(cfg: Config) -> None:
    # 9.5: duplicates come from QoS 1 ("at least once"). Injected by a test hook (QUESTIONS Q19).
    clock, bus, got = make(cfg)
    z1, z2 = client(clock, bus, got, "z1"), client(clock, bus, got, "z2")
    z2.subscribe("r")
    z2.subscribe("g")
    clock.run_until(0.01)
    bus.faults.duplicate = lambda topic, key, sub: True
    z1.publish("r", "result", qos=1, key=("a",))
    z1.publish("g", "gps", qos=0, key=("b",))
    clock.run_until(1.0)
    r = [d for _, d in got["z2"] if d.topic == "r"]
    assert [d.duplicate for d in r] == [False, True]
    assert [d.duplicate for _, d in got["z2"] if d.topic == "g"] == [False]


def test_a_dropped_delivery_reaches_only_the_other_subscribers(cfg: Config) -> None:
    clock, bus, got = make(cfg)
    z1, z2, z3 = (client(clock, bus, got, n) for n in ("z1", "z2", "z3"))
    z2.subscribe("t")
    z3.subscribe("t")
    clock.run_until(0.01)
    bus.faults.drop = lambda topic, key, sub: sub == "z2"
    z1.publish("t", 1, qos=0, key=("a",))
    clock.run_until(1.0)
    assert got["z2"] == [] and len(got["z3"]) == 1
    assert bus.delivered == 1


# --- calls (5.3) --------------------------------------------------------------------------------------------


def test_a_call_is_handled_on_arrival_and_replied_after_the_reply_delay() -> None:
    clock = EventClock()
    seen: list[tuple[str, float]] = []
    call(
        clock,
        0.005,
        lambda: True,
        lambda: seen.append(("handled", clock.now)) or "ok",
        lambda reply: 0.003,
        lambda reply: seen.append((reply, clock.now)),
    )
    clock.run_until(1.0)
    assert seen == [("handled", 0.005), ("ok", 0.008)]


def test_a_dead_target_never_replies_and_a_dead_caller_gets_nothing() -> None:
    clock = EventClock()
    seen: list[object] = []
    call(clock, 0.005, lambda: False, lambda: seen.append("handled"), lambda r: 0.0, seen.append)
    call(clock, 0.005, lambda: True, lambda: "ok", lambda r: 0.001, seen.append, caller_alive=lambda: False)
    clock.run_until(1.0)
    assert seen == []
