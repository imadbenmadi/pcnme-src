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
from pcnme.net.radio import RadioUplink, shannon_rate
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


def test_one_uplink_per_zone_pushes_one_car_at_a_time_at_its_own_rate(cfg: Config) -> None:
    # A 150 KB frame (1.2 Mbit): the far car (300 m, 539 Mbps) pushes in 2.226 ms; the near one (10 m, 931
    # Mbps) arrives during that push and waits for it, then pushes in 1.289 ms.
    up = RadioUplink()
    far, near = shannon_rate(300.0, cfg.radio), shannon_rate(10.0, cfg.radio)
    a = up.send(now=0.0, size_bits=1.2e6, rate=far, delay=0.003)
    b = up.send(now=0.001, size_bits=1.2e6, rate=near, delay=0.001)
    assert a.push == pytest.approx(1.2e6 / far) == pytest.approx(0.002226, rel=1e-3)
    assert b.wait == pytest.approx(1.2e6 / far - 0.001)
    assert b.push == pytest.approx(1.2e6 / near)
    assert b.delivery == pytest.approx(1.2e6 / far + 1.2e6 / near + 0.001)


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
        hops.PUB_BH,
        hops.PUB_WAN,
        hops.TO_ZONE,
    ]
    assert len(set(names)) == len(names)
    assert all(isinstance(n, str) for n in names)
    assert hops.STREAM[hops.RADIO_UP] == hops.STREAM[hops.RADIO_DOWN] == "radio"  # Q5
    assert {hops.STREAM[h] for h in (hops.BH_OUT, hops.WAN_IN, hops.PUB_BH, hops.PUB_WAN, hops.TO_ZONE)} == {
        "link_delays"
    }


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
