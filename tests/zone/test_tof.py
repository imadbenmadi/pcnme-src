"""TOF-Broker and the path estimates (PLAN M2.5; final-design 7.1, 4.3, 16.2; QUESTIONS Q16)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from pcnme.config import Config, config_from_mapping, load_config, load_yaml
from pcnme.zone.estimates import PathEstimates, ZeroWanWait
from pcnme.zone.tof import TofBroker


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


@pytest.fixture(scope="module")
def est(cfg: Config) -> PathEstimates:
    return PathEstimates(cfg, wan_wait=ZeroWanWait())


# --- the estimates (4.3, Q16) -------------------------------------------------------------------------------


def test_t_in_here_is_zero_and_t_in_cloud_by_hand(cfg: Config, est: PathEstimates) -> None:
    """4.3: T̂_in,cloud = own backhaul wait + ŵ_WAN + S / B_WAN + δ̄_bh + δ̄_WAN.

    A T1 frame (1.2 Mbit), an idle backhaul, ŵ_WAN = 0 (no sample yet): 1.2e6 / 514.85e6 = 2.331 ms, + mean
    backhaul delay 3 ms + mean WAN delay 62.5 ms = 67.83 ms.
    """
    assert est.t_in_here() == 0.0
    expected = 1.2e6 / cfg.derived.b_wan + 0.003 + 0.0625
    assert est.t_in_cloud(1.2e6, backhaul_wait=0.0) == pytest.approx(expected, rel=1e-12)
    assert est.t_in_cloud(1.2e6, backhaul_wait=0.010) == pytest.approx(expected + 0.010, rel=1e-12)


def test_t_ret_is_the_mean_delays_down_to_the_car(est: PathEstimates) -> None:
    """Q16: fog: backhaul to the broker 3 + broker 1 + backhaul to the zone 3 + radio down 3 = 10 ms;
    cloud: WAN 62.5 + broker 1 + backhaul 3 + radio down 3 = 69.5 ms."""
    assert est.t_ret_fog() == pytest.approx(0.010, rel=1e-12)
    assert est.t_ret_cloud() == pytest.approx(0.0695, rel=1e-12)


def test_a_t1_cloud_round_trip_is_about_140_ms(cfg: Config, est: PathEstimates) -> None:
    # final-design 16.2: "T1: about 140 ms mean round trip against D = 100 ms": 67.83 + 4 (cloud GPU) + 69.5.
    rt = est.t_in_cloud(1.2e6, 0.0) + est.e_cloud("T1") + est.t_ret_cloud()
    assert rt == pytest.approx(0.1413, abs=5e-4)


def test_the_wan_wait_estimate_is_part_of_t_in_cloud(cfg: Config) -> None:
    # Once M3.3 has samples, ŵ_WAN adds to T̂_in,cloud; any WanWait source plugs in.
    class Fixed:
        def estimate(self) -> float:
            return 0.050

    with_wait = PathEstimates(cfg, wan_wait=Fixed())
    zero = PathEstimates(cfg, wan_wait=ZeroWanWait())
    assert with_wait.t_in_cloud(1.2e6, 0.0) == pytest.approx(zero.t_in_cloud(1.2e6, 0.0) + 0.050, rel=1e-12)


def test_a_boulder_exactly_on_its_limit_is_not_dropped(cfg: Config) -> None:
    # 7.1.3 drops when the cloud path is ">" the time left: equal is kept (binary-exact estimates).
    class Exact:
        def t_in_cloud(self, size_bits: float, backhaul_wait: float) -> float:
            return 0.25

        def e_cloud(self, task_type: str) -> float:
            return 0.125

        def t_ret_cloud(self) -> float:
            return 0.125

    tof = TofBroker(cfg, Exact())  # type: ignore[arg-type]
    # cloud path 0.5 s; time left -0.25 + L_max 0.75 = 0.5 s exactly
    assert (
        tof.classify("T2", "tight", "soft", 72e6, time_left=-0.25, l_max=0.75, backhaul_wait=0.0) == "cloud"
    )
    late = -0.25 - 2**-20
    assert tof.classify("T2", "tight", "soft", 72e6, time_left=late, l_max=0.75, backhaul_wait=0.0) == "drop"


def test_the_wan_wait_placeholder_is_zero_before_any_sample() -> None:
    # 4.3: "before any sample, ŵ_WAN = 0". M3.3 replaces ZeroWanWait with the state plane's estimate.
    assert ZeroWanWait().estimate() == 0.0


# --- the TOF-Broker (7.1) -----------------------------------------------------------------------------------


def test_t2_is_a_boulder_with_tight_deadlines_and_not_with_loose(cfg: Config, est: PathEstimates) -> None:
    tof = TofBroker(cfg, est)
    assert tof.is_boulder("T2", "tight")  # 0.720 >= 0.8 x 0.792
    assert not tof.is_boulder("T2", "loose")  # 0.720 < 0.8 x 3.0
    assert not any(tof.is_boulder(t, m) for t in ("T1", "T3", "T4") for m in ("loose", "tight"))


def test_a_boulder_goes_to_the_cloud_when_the_cloud_can_make_it(cfg: Config, est: PathEstimates) -> None:
    # T2 tight (soft, D = L_max = 0.792 s), 0.792 s left: cloud 139.85 + 3 + 62.5 + 200 + 69.5 = 474.9 ms.
    tof = TofBroker(cfg, est)
    assert (
        tof.classify("T2", "tight", "soft", 72e6, time_left=0.792, l_max=0.792, backhaul_wait=0.0) == "cloud"
    )


def test_a_soft_boulder_is_dropped_only_past_its_time_left_plus_l_max(
    cfg: Config, est: PathEstimates
) -> None:
    tof = TofBroker(cfg, est)
    cloud = est.t_in_cloud(72e6, 0.0) + est.e_cloud("T2") + est.t_ret_cloud()  # 474.85 ms
    just = cloud - 0.792  # time left at which cloud == left + L_max
    assert (
        tof.classify("T2", "tight", "soft", 72e6, time_left=just + 1e-6, l_max=0.792, backhaul_wait=0.0)
        == "cloud"
    )
    assert (
        tof.classify("T2", "tight", "soft", 72e6, time_left=just - 1e-6, l_max=0.792, backhaul_wait=0.0)
        == "drop"
    )


def test_a_hard_boulder_the_cloud_cannot_make_is_dropped_at_intake(default_config_path: Path) -> None:
    # Make T1 a boulder (tight D = 5 ms: 8 ms >= 0.8 x 5). Its cloud path takes about 141 ms > 5 ms left:
    # dropped at intake, with no agent decision (7.1.3: "no transition is stored").
    data: dict[str, Any] = copy.deepcopy(load_yaml(default_config_path))
    data["task_types"]["T1"]["deadline"]["tight"] = 0.005
    cfg = config_from_mapping(data)
    tof = TofBroker(cfg, PathEstimates(cfg, wan_wait=ZeroWanWait()))
    assert tof.is_boulder("T1", "tight")
    assert (
        tof.classify("T1", "tight", "hard", 1.2e6, time_left=0.005, l_max=0.005, backhaul_wait=0.0) == "drop"
    )
    assert (
        tof.classify("T1", "tight", "hard", 1.2e6, time_left=0.142, l_max=0.005, backhaul_wait=0.0) == "cloud"
    )


def test_a_pebble_goes_to_the_agent(cfg: Config, est: PathEstimates) -> None:
    tof = TofBroker(cfg, est)
    assert tof.classify("T1", "loose", "hard", 1.2e6, time_left=0.1, l_max=0.1, backhaul_wait=0.0) == "agent"
    assert tof.classify("T2", "loose", "soft", 72e6, time_left=3.0, l_max=3.0, backhaul_wait=0.0) == "agent"


def test_a_busy_own_backhaul_counts_in_the_boulder_test(cfg: Config, est: PathEstimates) -> None:
    # The zone's own backhaul wait is known exactly (4.3) and is part of T̂_in,cloud.
    tof = TofBroker(cfg, est)
    assert tof.classify("T2", "tight", "soft", 72e6, time_left=0.0, l_max=0.792, backhaul_wait=0.0) == "cloud"
    assert tof.classify("T2", "tight", "soft", 72e6, time_left=0.0, l_max=0.792, backhaul_wait=0.4) == "drop"
