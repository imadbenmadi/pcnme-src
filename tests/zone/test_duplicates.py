"""Duplicates, subscriptions and cleanup (PLAN M3.7; final-design 6D.14, 9.5, 9.6)."""

from __future__ import annotations

from pcnme.bus.bus import Delivery
from tests.zone.harness import Fixed, city, city_task, fixed_delay_config, one_zone, task_now


def spy(s, topic: str) -> list[Delivery]:  # type: ignore[no-untyped-def]
    got: list[Delivery] = []
    s.bus.connect(f"spy{len(s.bus._clients)}", (0.003, 0.003), got.append).subscribe(topic)
    return got


def test_the_serving_zone_pushes_a_task_once_per_radio_session() -> None:
    # 9.5.3: a QoS 1 repeat that reaches the serving zone after the car confirmed is not pushed. Here the
    # repeat is 40 ms late (a slow redelivery), after the car's confirm (3 ms after the first push).
    s = one_zone(fixed_delay_config(), Fixed("here"))
    s.bus.faults.duplicate = lambda topic, key, sub: topic == "car/c01/result"
    s.bus._delay_orig = s.bus._delay  # type: ignore[attr-defined]
    s.bus._delay = lambda c, key, hop: 0.040 if hop == "dup" else s.bus._delay_orig(c, key, hop)  # type: ignore[method-assign]
    s.zone.upload(task_now(s, "T1"))
    s.clock.run_until(1.0)
    (rec,) = s.closed
    assert rec.outcome == "on_time" and rec.deliveries == 1


def test_a_new_radio_session_gets_the_result_again_until_the_ack() -> None:
    # 6D.14: until the ack, the anchor republishes its copy on every new attach; a new session may push it.
    s = one_zone(fixed_delay_config(), Fixed("here"))
    s.bus.faults.drop = lambda topic, key, sub: topic == "car/c01/ack"  # the ack never comes
    s.zone.upload(task_now(s, "T1"))
    s.clock.run_until(0.030)  # the car has the result (23.2 ms) and confirmed
    rec = next(iter(s.zone.records.values()))
    assert rec.deliveries == 1 and rec.copy is not None
    s.zone.attach(s.car, 300.0)  # the car attaches again: a new radio session
    s.clock.run_until(0.060)
    assert rec.deliveries == 2


def test_the_anchor_republishes_once_when_no_ack_comes_within_t_ack() -> None:
    # 9.5.4: the result reached the anchor at 20.2 ms; no ack by 20.2 + 200 ms: one republish, not more.
    s = one_zone(fixed_delay_config(), Fixed("here"))
    s.bus.faults.drop = lambda topic, key, sub: topic == "car/c01/ack"
    results = spy(s, "car/c01/result")
    s.clock.run_until(0.01)
    s.zone.upload(task_now(s, "T1"))
    s.clock.run_until(1.0)
    publishers = [(d.publisher, d.key[-2] if d.key[-2] == "republish" else "first") for d in results]
    assert publishers == [("z1", "first"), ("z1", "republish")]
    assert s.closed[0].outcome == "drop_no_ack"  # 9.5.5 at d_A + T_ack


def test_another_zones_attach_ends_the_serving_role() -> None:
    # 9.6: the serving zone unsubscribes result and attach when another zone announces an attach for the car,
    # and the car registry follows the serving role.
    c = city(fixed_delay_config(), lambda z: Fixed("here"), zone_ids=("z1", "z2"))
    c.clock.run_until(0.01)
    z1, z2 = c.zones["z1"], c.zones["z2"]
    assert "c01" in z1._cars and z1.mqtt.count("car/c01/result") == 1
    z2.attach(c.car, 200.0)
    c.clock.run_until(0.02)  # the attach crosses the broker (7 ms)
    assert "c01" not in z1._cars and z1.mqtt.count("car/c01/result") == 0
    assert "c01" in z2._cars and c.bus.subscribers("car/c01/result") == ("z2",)


def test_the_anchor_keeps_its_role_while_a_task_is_open_after_the_car_left() -> None:
    # The anchor role (9.6) is independent of serving: z1 still anchors the open task after the car moved to
    # z2, so it keeps result and ack, and drops them at the close.
    c = city(fixed_delay_config(), lambda z: Fixed("here"), zone_ids=("z1", "z2"))
    c.clock.run_until(0.01)
    z1, z2 = c.zones["z1"], c.zones["z2"]
    z1.upload(city_task(c, "T4"))
    c.clock.run_until(0.01 + 0.006)  # open at z1
    z2.attach(c.car, 200.0)
    c.clock.run_until(0.030)
    assert "c01" not in z1._cars and z1.mqtt.count("car/c01/result") == 1  # the anchor role
    c.clock.run_until(3.0)
    (rec,) = c.closed
    assert rec.outcome == "on_time" and rec.confirm is not None  # the result followed the car to z2 (C7)
    assert rec.deliveries == 1  # noted by z2 in the run's log
    assert abs(sum(rec.parts.values()) - rec.confirm.latency) <= 1e-12
    assert z1.mqtt.count("car/c01/result") == 0 and z1.mqtt.count("car/c01/ack") == 0
