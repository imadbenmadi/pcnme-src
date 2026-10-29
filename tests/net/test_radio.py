"""Radio uplink and the hop delays (PLAN M2.3; final-design 4.1, 4.2, 18; CHANGES C1, C3; QUESTIONS Q17)."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from pcnme.config import Config, load_config
from pcnme.ids import TaskId
from pcnme.net import hops
from pcnme.net.pipe import FifoPipe
from pcnme.net.radio import RadioTransfer, RadioUplink, shannon_rate
from pcnme.sim.clock import EventClock
from pcnme.sim.rng import RandomStreams


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


def test_the_shannon_rate_at_300_m_by_hand(cfg: Config) -> None:
    # lambda = 299 792 458 / 5.9e9 = 0.050812 m; free-space gain (lambda / (4 pi 300))^2 = 1.8155e-10; x G_t
    # G_r (5 + 5 dBi = 10) x 1 W = 1.8155e-9 W received. Noise = 10^-20.4 W/Hz x 40 MHz = 1.5924e-13 W
    # (CHANGES C1). SNR = 11 401; R = 40e6 x log2(1 + 11 401) = 539.1 Mbps.
    lam = 299_792_458 / 5.9e9
    received = 1.0 * 10.0 * (lam / (4 * math.pi * 300)) ** 2
    snr = received / (10**-20.4 * 40e6)
    assert shannon_rate(300.0, cfg.radio) == round(40e6 * math.log2(1 + snr))  # whole bit/s
    assert shannon_rate(300.0, cfg.radio) == pytest.approx(539.1e6, rel=1e-3)


def test_rates_span_540_to_930_mbps_over_10_to_300_m(cfg: Config) -> None:
    # final-design 18 (Radio) and 17: "about 540-930 Mbps at 10-300 m".
    assert shannon_rate(10.0, cfg.radio) == pytest.approx(931e6, rel=2e-3)
    assert shannon_rate(300.0, cfg.radio) == pytest.approx(539e6, rel=2e-3)
    d = np.linspace(10, 300, 50)
    rates = [shannon_rate(float(x), cfg.radio) for x in d]
    assert all(a > b for a, b in itertools.pairwise(rates))  # farther is slower


def test_very_short_distances_count_as_1_m(cfg: Config) -> None:
    # Engineering choice: free-space loss has no limit at d -> 0.
    assert shannon_rate(0.0, cfg.radio) == shannon_rate(1.0, cfg.radio) == shannon_rate(0.4, cfg.radio)


def uplink() -> tuple[EventClock, RadioUplink, dict[str, RadioTransfer]]:
    clock = EventClock()
    return clock, RadioUplink(clock), {}


def send_at(
    clock: EventClock,
    up: RadioUplink,
    done: dict[str, RadioTransfer],
    t: float,
    name: str,
    bits: float,
    rate: float,
    delay: float = 0.0,
) -> None:
    clock.schedule(t, lambda: up.send(bits, rate, delay, lambda tr: done.__setitem__(name, tr)))


def test_a_car_alone_pushes_at_its_own_rate(cfg: Config) -> None:
    # 150 KB (1.2 Mbit) at 300 m (539.1 Mbps): 2.226 ms, no wait; delivered after the 3 ms radio delay.
    clock, up, done = uplink()
    far = shannon_rate(300.0, cfg.radio)
    send_at(clock, up, done, 0.0, "a", 1.2e6, far, delay=0.003)
    clock.run_until(1.0)
    a = done["a"]
    assert (a.push, a.wait) == (pytest.approx(0.002226, rel=1e-3), 0.0)
    assert a.push_end == pytest.approx(1.2e6 / far, rel=1e-12)
    assert a.delivery == pytest.approx(1.2e6 / far + 0.003, rel=1e-12)


def test_a_t1_behind_a_t2_shares_the_air_instead_of_waiting(cfg: Config) -> None:
    """CHANGES C6, by hand. Both cars at 300 m, R = 539.1 Mbps. A T2 (72 Mbit) starts at 0, a T1 (1.2 Mbit) at
    10 ms. While both are on the air each advances at R / 2, so the T1 needs 2 x 2.226 = 4.452 ms and ends at
    14.452 ms (W_radio = 2.226 ms; one car at a time it would wait 123.6 ms). The T2 lost 2.226 ms of airtime
    and ends at 133.56 + 2.226 = 135.79 ms: the air is busy as long as before (72 + 1.2 Mbit) / R."""
    clock, up, done = uplink()
    r = shannon_rate(300.0, cfg.radio)
    send_at(clock, up, done, 0.0, "T2", 72e6, r)
    send_at(clock, up, done, 0.010, "T1", 1.2e6, r)
    clock.run_until(1.0)
    t1, t2 = done["T1"], done["T2"]
    assert t1.push_end == pytest.approx(0.010 + 2 * 1.2e6 / r, rel=1e-12)
    assert t1.wait == pytest.approx(1.2e6 / r, rel=1e-9) == pytest.approx(0.002226, rel=1e-3)
    assert t2.push_end == pytest.approx((72e6 + 1.2e6) / r, rel=1e-12)
    assert t2.wait == pytest.approx(1.2e6 / r, rel=1e-9)


def test_cars_at_different_rates_get_equal_airtime_not_equal_bits(cfg: Config) -> None:
    """Same 1.2 Mbit at 0 from 10 m (931 Mbps, 1.289 ms alone) and 300 m (539 Mbps, 2.226 ms alone). Half the
    airtime each: the near car ends at 2 x 1.289 = 2.578 ms, the far one at 1.289 + 2.226 = 3.515 ms; both
    waited 1.289 ms."""
    clock, up, done = uplink()
    near, far = shannon_rate(10.0, cfg.radio), shannon_rate(300.0, cfg.radio)
    send_at(clock, up, done, 0.0, "near", 1.2e6, near)
    send_at(clock, up, done, 0.0, "far", 1.2e6, far)
    clock.run_until(1.0)
    assert done["near"].push_end == pytest.approx(2 * 1.2e6 / near, rel=1e-12)
    assert done["far"].push_end == pytest.approx(1.2e6 / near + 1.2e6 / far, rel=1e-12)
    assert done["near"].wait == pytest.approx(done["far"].wait, rel=1e-9) == pytest.approx(1.2e6 / near)


def test_equal_uploads_started_together_end_together(cfg: Config) -> None:
    clock, up, done = uplink()
    r = shannon_rate(100.0, cfg.radio)
    for name in ("a", "b", "c"):
        send_at(clock, up, done, 0.0, name, 1.2e6, r)
    clock.run_until(1.0)
    ends = {done[n].push_end for n in "abc"}
    assert len(ends) == 1 and ends.pop() == pytest.approx(3 * 1.2e6 / r, rel=1e-12)  # one event, one time
    assert up.in_progress == 0


def test_the_air_is_busy_exactly_when_one_car_at_a_time_would_be(cfg: Config) -> None:
    # Both are work-conserving in airtime, so their busy periods match: compare the instants the air goes idle
    # over 2 000 random uploads (T1 and T2 sizes, cars at 10-300 m, Poisson starts at radio TI about 0.6).
    rng = np.random.default_rng(7)
    starts = np.cumsum(rng.exponential(1 / 40.0, 2000))
    bits = np.where(rng.random(2000) < 0.03, 72e6, 1.2e6)
    rates = [shannon_rate(float(d), cfg.radio) for d in rng.uniform(10.0, 300.0, 2000)]
    fifo_idle, free = [], 0.0
    for k, (s, b, r) in enumerate(zip(starts, bits, rates, strict=True)):
        free = max(s, free) + b / r
        if k == len(starts) - 1 or starts[k + 1] > free:
            fifo_idle.append(free)
    clock = EventClock()
    up = RadioUplink(clock)
    idle: list[float] = []

    def done(tr: RadioTransfer) -> None:
        if up.in_progress == 0:
            idle.append(tr.push_end)

    for s, b, r in zip(starts, bits, rates, strict=True):
        clock.schedule(float(s), lambda b=b, r=r: up.send(float(b), r, 0.0, done))
    clock.run_until(float(starts[-1]) + 10.0)
    assert len(idle) == len(fifo_idle) > 100
    assert idle == pytest.approx(fifo_idle, rel=1e-9)


def test_the_uplink_rejects_bad_arguments(cfg: Config) -> None:
    _, up, _ = uplink()
    for bits, rate, delay in ((0.0, 1e9, 0.0), (1e6, 0.0, 0.0), (1e6, 1e9, -1.0), (math.inf, 1e9, 0.0)):
        with pytest.raises(ValueError):
            up.send(bits, rate, delay, lambda tr: None)


def test_the_fifo_pipe_takes_a_per_transfer_rate() -> None:
    # The radio pipe has no fixed bandwidth: every push uses the sending car's rate. A fixed pipe still works.
    pipe = FifoPipe(None)
    assert pipe.send(now=0.0, size_bits=1e6, delay=0.0, rate=1e9).push == pytest.approx(0.001)
    with pytest.raises(ValueError, match="rate"):
        pipe.send(now=0.0, size_bits=1e6, delay=0.0)
    assert FifoPipe(1e9).send(now=0.0, size_bits=1e6, delay=0.0).push == pytest.approx(0.001)
    assert FifoPipe(1e9).send(now=0.0, size_bits=1e6, delay=0.0, rate=5e8).push == pytest.approx(0.002)


# --- hop delays (C3, Q17) -----------------------------------------------------------------------------------


def test_hop_names_are_fixed_strings_in_one_place() -> None:
    names = [
        hops.RADIO_UP,
        hops.RADIO_DOWN,
        hops.BH_OUT,
        hops.WAN_IN,
        hops.ACK_UP,
        hops.REPLY_BH,
        hops.REPLY_WAN,
        hops.CALL_BH,
        hops.CALL_WAN,
    ]
    assert len(set(names)) == len(names)
    assert all(isinstance(n, str) for n in names)
    assert (
        hops.STREAM[hops.RADIO_UP] == hops.STREAM[hops.RADIO_DOWN] == hops.STREAM[hops.ACK_UP] == "radio"
    )  # Q5
    assert {
        hops.STREAM[h]
        for h in (hops.BH_OUT, hops.WAN_IN, hops.REPLY_BH, hops.REPLY_WAN, hops.CALL_BH, hops.CALL_WAN)
    } == {"link_delays"}
    assert set(hops.STREAM) == set(names)  # MQTT hops are drawn in the bus (M3.1)


def test_a_hop_delay_is_uniform_on_its_interval_and_keyed_by_task_attempt_hop(cfg: Config) -> None:
    streams = RandomStreams.from_config(cfg, 2000, "training")
    lo, hi = cfg.radio.uplink_delay
    draws = np.array(
        [hops.delay(streams, hops.RADIO_UP, (lo, hi), TaskId("c01", 0, s), 0) for s in range(1, 20_001)]
    )
    assert draws.min() >= lo and draws.max() <= hi
    assert stats.kstest((draws - lo) / (hi - lo), "uniform").pvalue > 1e-3
    t = TaskId("c01", 0, 5)
    u = streams.uniform("radio", *t, 0, hops.RADIO_UP)
    assert hops.delay(streams, hops.RADIO_UP, (lo, hi), t, 0) == round((lo + u * (hi - lo)) * 1e9) / 1e9  # ns
    # Another hop, another attempt or another task: another draw.
    d = hops.delay(streams, hops.RADIO_UP, (lo, hi), t, 0)
    assert d != hops.delay(streams, hops.RADIO_DOWN, (lo, hi), t, 0)
    assert d != hops.delay(streams, hops.RADIO_UP, (lo, hi), t, 1)


def test_the_upload_uses_attempt_0(cfg: Config) -> None:
    # Q17: the upload comes before any attempt (attempts count from 1).
    assert hops.UPLOAD_ATTEMPT == 0


def test_a_cut_upload_leaves_the_air_to_the_others(cfg: Config) -> None:
    # 9.7. A 9 Mbit upload (16.69 ms alone at 539.1 Mbps) shares the air with a T2 from 0; the T2 is cut at
    # 10 ms (a handover). In those 10 ms the small one got half the air: 5 ms of its push, 11.69 ms left,
    # now alone: it ends at 21.69 ms. The cut T2 never arrives.
    clock, up, done = uplink()
    r = shannon_rate(300.0, cfg.radio)
    ids: dict[str, int] = {}
    clock.schedule(
        0.0, lambda: ids.__setitem__("big", up.send(72e6, r, 0.0, lambda tr: done.__setitem__("big", tr)))
    )
    send_at(clock, up, done, 0.0, "small", 9e6, r)
    clock.run_until(0.010)
    assert up.cancel(ids["big"]) and not up.cancel(ids["big"])  # twice: nothing left to cut
    clock.run_until(1.0)
    assert "big" not in done
    assert done["small"].push_end == pytest.approx(0.010 + (9e6 / r - 0.005), rel=1e-9)
