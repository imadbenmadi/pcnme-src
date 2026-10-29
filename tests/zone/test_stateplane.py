"""The zone state plane (PLAN M3.3; final-design 4.3, 5.2, 5.4, 5.5, 9.2; QUESTIONS Q20)."""

from __future__ import annotations

import pytest

from pcnme.ids import SnapshotVersion
from pcnme.node.state import SlotState
from pcnme.sim.clock import EventClock
from pcnme.zone.layout import CLOUD
from pcnme.zone.statestore import Snapshot, StateStore
from tests.zone.harness import Fixed, fixed_delay_config, one_zone, task_now

EMPTY = {"gpu": SlotState(2, 32, 0, 0, 0.0, 0.0, (0.0, 0.0), 0)}


def store(clock: EventClock) -> StateStore:
    return StateStore("z1", clock, stale_after=0.3, pub_delay={"z2": 0.007, CLOUD: 0.0665}, wan_lambda=0.2)


def snap(zone: str, epoch: int, counter: int, wan: float = 0.0, age: float | None = None) -> Snapshot:
    return Snapshot(zone, SnapshotVersion(epoch, counter), EMPTY, wan, age, 0.0)  # type: ignore[arg-type]


# --- the store ----------------------------------------------------------------------------------------------


def test_the_newest_version_wins_whatever_the_arrival_order() -> None:
    # 5.5, 9.2 R1: versions from state messages and replies; the epoch is compared first.
    s = store(EventClock())
    assert s.on_snapshot(snap("z2", 0, 5))
    assert not s.on_snapshot(snap("z2", 0, 4))  # older, arrived later
    assert not s.on_snapshot(snap("z2", 0, 5))  # the same version again
    assert s.on_snapshot(snap("z2", 1, 1))  # a restarted zone: new epoch beats any old counter
    assert not s.on_snapshot(snap("z2", 0, 99))
    assert s.snapshot("z2").version == (1, 1)  # type: ignore[union-attr]


def test_age_and_staleness_on_the_anchors_own_clock() -> None:
    # 5.4: age = now - local receive time + δ̂_pub; 7.2: stale beyond 300 ms; nothing received = stale.
    clock = EventClock()
    s = store(clock)
    assert s.age("z2") is None and s.is_stale("z2")
    clock.run_until(1.0)
    s.on_snapshot(snap("z2", 0, 1))
    clock.run_until(1.2)
    assert s.age("z2") == pytest.approx(0.2 + 0.007)
    assert not s.is_stale("z2")
    clock.run_until(1.293)
    assert s.age("z2") == pytest.approx(0.3) and not s.is_stale("z2")  # exactly 300 ms: not older
    clock.run_until(1.294)
    assert s.is_stale("z2")


def test_pending_sends_count_per_executor_and_slot_type_until_cleared() -> None:
    s = store(EventClock())
    s.add_pending("z2", ("t1", 1), "gpu", "hard", 0.008)
    s.add_pending("z2", ("t2", 1), "cpu", "soft", 0.2)
    assert [p.mean for p in s.pending_on("z2", "gpu")] == [0.008]
    s.clear_pending("z2", ("t1", 1))
    s.clear_pending("z2", ("t1", 1))  # twice is harmless
    assert s.pending_on("z2", "gpu") == () and len(s.pending_on("z2", "cpu")) == 1


def test_the_wan_wait_is_a_moving_average_with_lambda_02() -> None:
    # 4.3, 18: ŵ <- 0.8 ŵ + 0.2 sample; 0 before any sample.
    s = store(EventClock())
    assert s.estimate() == 0.0
    s.wan_sample(0.1)
    assert s.estimate() == pytest.approx(0.02)
    s.wan_sample(0.1)
    assert s.estimate() == pytest.approx(0.036)


def test_a_zone_uses_the_freshest_wan_estimate_among_its_own_and_its_neighbours() -> None:
    clock = EventClock()
    s = store(clock)
    s.wan_sample(0.1)  # own: 0.02, taken at 0
    clock.run_until(0.5)
    s.on_snapshot(snap("z2", 0, 1, wan=0.05, age=0.1))  # z2's: 0.1 + 0.007 old < own 0.5
    assert s.estimate() == pytest.approx(0.05)
    s.wan_sample(0.1)  # own again, 0 s old: 0.036
    assert s.estimate() == pytest.approx(0.036)
    s.on_snapshot(snap(CLOUD, 0, 1))  # the cloud publishes no estimate (4.3)
    assert s.estimate() == pytest.approx(0.036)


# --- in a zone ----------------------------------------------------------------------------------------------


def test_the_zone_publishes_its_state_every_state_period() -> None:
    s = one_zone(fixed_delay_config(), Fixed("here"))
    got: list[Snapshot] = []
    s.bus.connect("spy", (0.003, 0.003), lambda d: got.append(d.payload)).subscribe("zone/z1/state")
    s.clock.run_until(1.0)
    # subscribed at 3 ms: the retained copy of the t = 0 state, then t = 0.1 ... 0.9 (at 0.107 ... 0.907)
    assert [g.version.counter for g in got][-3:] == [8, 9, 10]
    assert len(got) == 10 and all(g.zone_id == "z1" for g in got)


def test_before_the_clouds_first_state_arrives_the_cloud_is_masked_stale() -> None:
    # The cloud's t = 0 state reaches z1 at 62.5 + 1 + 3 = 66.5 ms (WAN publisher, Q20).
    s = one_zone(fixed_delay_config(), Fixed("here"))
    t = task_now(s, "T4")
    early = s.zone._context(t, 0.0, 1.0)
    assert early.cloud is None and early.views[-1].masked == "stale"
    s.clock.run_until(0.07)
    later = s.zone._context(t, s.clock.now, 1.0)
    assert later.cloud is not None and later.views[-1].masked is None
    assert s.zone.store.age(CLOUD) == pytest.approx(0.07, abs=1e-9)  # 70 ms since it was sent


def test_a_cloud_send_is_pending_until_its_reply_and_the_reply_carries_a_snapshot() -> None:
    # 9.2 R1. T4 from 300 m: t_rx 5.226 ms; the input reaches the cloud at 73.057 ms; the reply crosses the
    # WAN (62.5) and the backhaul (3): back at 138.557 ms.
    s = one_zone(fixed_delay_config(), Fixed("cloud"))
    s.clock.run_until(0.1)  # the cloud's state is in
    s.zone.upload(task_now(s, "T4"))
    s.clock.run_until(0.15)
    assert len(s.zone.store.pending_on(CLOUD, "gpu")) == 1
    before = s.zone.store.snapshot(CLOUD).version  # type: ignore[union-attr]
    s.clock.run_until(0.1 + 0.138557 + 1e-6)
    assert s.zone.store.pending_on(CLOUD, "gpu") == ()
    assert s.zone.store.snapshot(CLOUD).version > before  # type: ignore[union-attr]


def test_the_wan_sample_is_the_wan_wait_by_hand() -> None:
    """4.3, every delay at its mean, the car 10 m away (931 Mbps). Alone on the WAN the sample is exactly 0,
    so ŵ stays 0. Then a T2 clip, and a T4 that starts uploading 1 ms after the T2's last bit. The T2 holds
    the zone's own 1 Gbps backhaul for 72 ms, so the T4 waits there 72 - 1 - 1.29 = 69.71 ms (known exactly)
    and reaches the WAN 72 ms after the T2; the T2's WAN push takes 72e6 / B_WAN = 139.85 ms, so the T4 waits
    67.85 ms on the WAN. The sample takes the backhaul wait off and is exactly that W_WAN. The T2's own sample
    is 0 (it found the WAN empty), so ŵ = 0.2 x 67.85 ms."""
    s = one_zone(fixed_delay_config(), Fixed("cloud"), distance=10.0)
    s.clock.run_until(0.1)
    s.zone.upload(task_now(s, "T4", seq=1))
    s.clock.run_until(0.6)
    assert s.zone.store.wan.estimate == pytest.approx(0.0, abs=1e-12)
    assert s.zone.store.wan.last_sample is not None
    rate = s.zone._cars["c01"][1]
    s.zone.upload(task_now(s, "T2", seq=2))
    s.clock.run_until(s.clock.now + 72e6 / rate + 0.001)
    s.zone.upload(task_now(s, "T4", seq=3))
    s.clock.run_until(3.0)
    w_wan = 72e6 / s.cfg.derived.b_wan - 72e6 / s.cfg.links.backhaul_bw
    assert w_wan == pytest.approx(0.06785, abs=1e-5)
    t4 = next(r for r in s.zone.records.values() if r.task.task_id.seq == 3)
    assert t4.transfer is not None and t4.transfer.w_wan == pytest.approx(w_wan, rel=1e-9)
    assert t4.transfer.w_bh == pytest.approx(0.072 - 0.001 - 1.2e6 / rate, rel=1e-6)
    assert s.zone.store.wan.estimate == pytest.approx(0.2 * w_wan, rel=1e-6)


def test_a_zone_learns_the_wan_wait_from_its_neighbours_state() -> None:
    from pcnme.bus.bus import Bus
    from pcnme.net.pipe import FifoPipe
    from pcnme.node.timetable import RunTimeTable
    from pcnme.sim.rng import RandomStreams
    from pcnme.zone.layout import Site
    from pcnme.zone.zone import Cloud, Zone

    cfg = fixed_delay_config()
    clock = EventClock()
    streams = RandomStreams.from_config(cfg, 2000, "training")
    bus = Bus(clock, streams, cfg.links.broker_extra_delay)
    table = RunTimeTable.from_config(cfg)
    wan = FifoPipe(cfg.derived.b_wan)
    cloud = Cloud(clock, streams, cfg, table, bus)
    r = cfg.layout.zone_radius_m
    sites = {z: Site(z, *cfg.layout.zone_positions_m[z], r) for z in ("z1", "z2")}
    z1 = Zone(
        "z1",
        clock,
        streams,
        cfg,
        table,
        wan,
        cloud,
        Fixed(None),
        "loose",
        bus=bus,
        start_sites=(sites["z2"],),
    )
    z2 = Zone(
        "z2",
        clock,
        streams,
        cfg,
        table,
        wan,
        cloud,
        Fixed(None),
        "loose",
        bus=bus,
        start_sites=(sites["z1"],),
    )
    z1.store.wan_sample(0.1)  # z1 measured 0.02
    clock.run_until(0.25)  # z1's state at 0.2 reaches z2 at 0.207
    assert z2.store.estimate() == pytest.approx(0.02)


def test_stale_means_strictly_older_than_stale_after() -> None:
    # Dyadic numbers, exact in binary: received at 1.0, δ̂_pub 0.0625; at 1.1875 the age is exactly 0.25.
    clock = EventClock()
    s = StateStore("z1", clock, stale_after=0.25, pub_delay={"z2": 0.0625}, wan_lambda=0.2)
    clock.run_until(1.0)
    s.on_snapshot(snap("z2", 0, 1))
    clock.run_until(1.1875)
    assert s.age("z2") == 0.25 and not s.is_stale("z2")


def test_a_neighbours_estimate_ages_with_its_snapshot() -> None:
    # z2's sample was 0.1 s old when sent, and its snapshot is 0.3 + 0.007 s old: 0.407 > own 0.35.
    clock = EventClock()
    s = store(clock)
    clock.run_until(0.3)
    s.on_snapshot(snap("z2", 0, 1, wan=0.05, age=0.1))
    s.wan_sample(0.1)  # own 0.02 at 0.3
    clock.run_until(0.65)
    assert s.estimate() == pytest.approx(0.02)


def test_the_reply_snapshot_is_held_the_moment_it_arrives() -> None:
    # The reply (made at the cloud at 173.057 ms, its 3rd snapshot after the states of 0 and 0.1 s) lands
    # at 238.557 ms; the cloud's 0.2 s state (4th) only at 266.5 ms.
    s = one_zone(fixed_delay_config(), Fixed("cloud"))
    s.clock.run_until(0.1)
    s.zone.upload(task_now(s, "T4"))
    s.clock.run_until(0.2386)
    assert s.zone.store.snapshot(CLOUD).version.counter == 3  # type: ignore[union-attr]


def test_a_cloud_that_stops_publishing_goes_stale() -> None:
    # Its last state (sent at 0.1 s) is received at 166.5 ms; older than 300 ms from 0.4 s on.
    s = one_zone(fixed_delay_config(), Fixed("here"))
    t = task_now(s, "T4")
    s.clock.run_until(0.15)
    s.bus.kill("cloud")
    s.clock.run_until(0.39)
    assert s.zone._context(t, s.clock.now, 1.0).views[-1].masked is None
    s.clock.run_until(0.41)
    assert s.zone._context(t, s.clock.now, 1.0).views[-1].masked == "stale"
