"""Two pipes in series: a zone's backhaul pipe, then the shared WAN pipe (PLAN M1.3; final-design 4.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcnme.config import load_config
from pcnme.net.pipe import FifoPipe
from pcnme.net.series import CloudPath, SeriesTransfer
from pcnme.sim.clock import EventClock


@pytest.fixture(scope="module")
def bw(default_config_path: Path) -> tuple[float, float]:
    """(backhaul, WAN) bandwidths from the config: 1 Gbps and the derived B_WAN, about 515 Mbps."""
    cfg = load_config(default_config_path)
    return cfg.links.backhaul_bw, cfg.derived.b_wan


T2_BITS = 9000 * 8000  # a 9 MB clip (config task_types.T2.input_kb)
T4_BITS = 150 * 8000  # a 150 KB frame


def test_both_pipes_busy_by_hand(bw: tuple[float, float]) -> None:
    b_bh, b_wan = bw
    clock = EventClock()
    wan = FifoPipe(b_wan)
    zone_a, zone_b = FifoPipe(b_bh), FifoPipe(b_bh)
    path_a, path_b = CloudPath(clock, zone_a, wan), CloudPath(clock, zone_b, wan)
    done: list[SeriesTransfer] = []

    # t = 0: zone A's backhaul is busy with a 2 Mbit transfer to a neighbour (push 2 ms).
    zone_a.send(now=0.0, size_bits=2e6, delay=0.0)
    # t = 0: zone B sends a T2 clip to the cloud: no backhaul wait, d_bh 1 ms -> enters the WAN at 1 ms.
    path_b.send(T2_BITS, d_bh=0.001, d_wan=0.050, on_delivered=done.append)
    # t = 0.5 ms: zone A sends a T4 frame to the cloud.
    clock.run_until(0.0005)
    path_a.send(T4_BITS, d_bh=0.003, d_wan=0.060, on_delivered=done.append)
    clock.run_until(10.0)

    clip, frame = sorted(done, key=lambda t: t.start)
    clip_push = T2_BITS / b_wan
    assert clip.delivery == pytest.approx(0.001 + clip_push + 0.050, rel=1e-12)

    # Zone A: W_bh = 2 ms - 0.5 ms; WAN entry = 0.5 + 1.5 + 3 = 5 ms; the WAN is busy with B's clip until
    # 1 ms + 72 Mbit / B_WAN (139.85 ms), so W_WAN = that - 5 ms.
    w_bh = 0.002 - 0.0005
    entry = 0.0005 + w_bh + 0.003
    w_wan = (0.001 + clip_push) - entry
    push = T4_BITS / b_wan
    assert frame.w_bh == pytest.approx(w_bh, rel=1e-12)
    assert frame.wan_entry == pytest.approx(entry, rel=1e-12)
    assert frame.w_wan == pytest.approx(w_wan, rel=1e-12)
    assert frame.push_wan == pytest.approx(push, rel=1e-12)
    # final-design 4.1: T = W_bh + d_bh + W_WAN + S / B_WAN + d_WAN
    assert frame.duration == pytest.approx(w_bh + 0.003 + w_wan + push + 0.060, rel=1e-12)
    assert frame.delivery == pytest.approx(0.0005 + frame.duration, rel=1e-12)
    assert w_wan == pytest.approx(0.13585, abs=1e-5)  # the clip holds the narrow WAN for about 136 ms


def test_the_backhaul_push_time_is_not_added(bw: tuple[float, float]) -> None:
    # Packets flow through both pipes at once, so the narrower WAN sets the push time: S / B_bh is not in T.
    b_bh, b_wan = bw
    clock = EventClock()
    done: list[SeriesTransfer] = []
    CloudPath(clock, FifoPipe(b_bh), FifoPipe(b_wan)).send(T2_BITS, 0.002, 0.040, done.append)
    clock.run_until(10.0)
    assert done[0].duration == pytest.approx(0.002 + T2_BITS / b_wan + 0.040, rel=1e-12)


def test_the_zone_backhaul_stays_busy_for_its_own_push(bw: tuple[float, float]) -> None:
    # R3: the zone's next transfer (to anyone) waits for this transfer's push on its own pipe: S / B_bh.
    b_bh, b_wan = bw
    clock = EventClock()
    backhaul = FifoPipe(b_bh)
    CloudPath(clock, backhaul, FifoPipe(b_wan)).send(T2_BITS, 0.002, 0.040, lambda t: None)
    assert backhaul.wait_at(0.0) == pytest.approx(T2_BITS / b_bh, rel=1e-12)  # 72 ms at 1 Gbps


def test_transfers_join_the_wan_in_entry_order_not_send_order(bw: tuple[float, float]) -> None:
    # R4: zone A sends first with a long backhaul delay; zone B sends later with a short one and reaches the
    # shared WAN first, so B is pushed first and A waits behind it.
    b_bh, b_wan = bw
    clock = EventClock()
    wan = FifoPipe(b_wan)
    path_a, path_b = CloudPath(clock, FifoPipe(b_bh), wan), CloudPath(clock, FifoPipe(b_bh), wan)
    done: list[SeriesTransfer] = []
    path_a.send(T4_BITS, d_bh=0.005, d_wan=0.025, on_delivered=done.append)  # enters the WAN at 5 ms
    clock.run_until(0.001)
    path_b.send(T4_BITS, d_bh=0.001, d_wan=0.025, on_delivered=done.append)  # enters at 2 ms
    clock.run_until(10.0)
    a = next(t for t in done if t.start == 0.0)
    b = next(t for t in done if t.start == 0.001)
    assert b.w_wan == 0.0
    push = T4_BITS / b_wan
    assert a.w_wan == pytest.approx(max(0.0, 0.002 + push - 0.005), rel=1e-12)  # 0 here: B finished by 4.3 ms
    assert b.wan_entry < a.wan_entry


def test_delivery_is_an_event_at_the_delivery_time(bw: tuple[float, float]) -> None:
    b_bh, b_wan = bw
    clock = EventClock()
    seen: list[tuple[float, float]] = []
    CloudPath(clock, FifoPipe(b_bh), FifoPipe(b_wan)).send(
        T4_BITS, 0.001, 0.030, lambda t: seen.append((clock.now, t.delivery))
    )
    clock.run_until(0.030)
    assert seen == []  # not yet: delivery is at 1 + 2.33 + 30 ms
    clock.run_until(1.0)
    assert seen[0][0] == pytest.approx(seen[0][1], rel=1e-12)


def test_many_zones_share_one_wan(bw: tuple[float, float]) -> None:
    # Four zones send a frame at the same time with the same delays: the WAN pushes them one after another.
    b_bh, b_wan = bw
    clock = EventClock()
    wan = FifoPipe(b_wan)
    done: list[SeriesTransfer] = []
    for _ in range(4):
        CloudPath(clock, FifoPipe(b_bh), wan).send(T4_BITS, 0.001, 0.025, done.append)
    clock.run_until(10.0)
    push = T4_BITS / b_wan
    assert sorted(t.w_wan for t in done) == pytest.approx([0.0, push, 2 * push, 3 * push], rel=1e-12)


@pytest.mark.parametrize(("d_bh", "d_wan"), [(-0.001, 0.03), (0.001, -0.03), (float("nan"), 0.03)])
def test_a_bad_delay_is_refused_before_any_pipe_changes(d_bh: float, d_wan: float) -> None:
    clock, backhaul = EventClock(), FifoPipe(1e9)
    with pytest.raises(ValueError, match="delay"):
        CloudPath(clock, backhaul, FifoPipe(515e6)).send(T4_BITS, d_bh, d_wan, lambda t: None)
    assert backhaul.t_free == 0.0 and clock.pending == 0


def test_a_wan_wider_than_the_backhaul_is_refused() -> None:
    # R5: the formula assumes the WAN is the narrowest pipe (4.1, 4.2).
    with pytest.raises(ValueError, match="narrow"):
        CloudPath(EventClock(), FifoPipe(1e9), FifoPipe(2e9))
