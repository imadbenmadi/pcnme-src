"""M2 acceptance tests (PLAN M2; final-design 16.5, 7.1, 12.2-12.5, 16.2).

- The hand calculations of 16.5 for "here" and "cloud", within 1 %: tests/zone/test_delivery.py (a T1 here,
  L = 23.226 ms; a T4 in the cloud, L = 154.557 ms; a hard T1 to the cloud refused late), listed in
  REFERENCED below and checked to exist.
- The TOF rule (7.1): tests/zone/test_tof.py and the two boulder tests in test_delivery.py.
- E_ref and ê equal calc_defaults (12.3, 12.4): tests/test_energy.py.
- Cloud-only at TI_WAN 1.5 grows without limit; at 0.5 it stays bounded: below, by hand.
- An end-to-end run of one zone (Greedy, S1 and S2) keeps the invariants (every task closes exactly once,
  L >= the sum of its parts, no hard task on time after D, energy >= 0), hits the target rho_fog and is
  deterministic: below. The 60 s x 3 seeds table is in the M2 report (src/pcnme/experiments/one_zone.py).
"""

from __future__ import annotations

import importlib
import itertools
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from pcnme.config import Config, load_config
from pcnme.experiments.one_zone import check_invariants, run_one_zone, summarise
from pcnme.net.radio import shannon_rate
from tests.zone.harness import Fixed, fixed_delay_config, one_zone, task_now

REFERENCED = {
    "tests.zone.test_delivery": [
        "test_a_t1_run_here_by_hand",
        "test_a_t4_run_in_the_cloud_by_hand",
        "test_a_hard_t1_sent_to_the_cloud_is_refused_late_and_dropped",
        "test_a_boulder_goes_to_the_cloud_with_no_decision",
        "test_a_boulder_the_cloud_cannot_make_is_dropped_at_intake",
        "test_each_task_closes_exactly_once_under_load",
    ],
    "tests.zone.test_tof": [],  # the whole file
    "tests.test_energy": [
        "test_e_ref_and_the_worst_ratios_equal_calc_defaults",
        "test_a_task_delivered_here_at_its_mean_run_costs_exactly_e_ref",
        "test_e_hat_is_capped_at_e_bar",
        "test_a_cloud_delivery_by_hand",
    ],
}


@pytest.mark.parametrize("module", sorted(REFERENCED))
def test_the_referenced_acceptance_tests_exist(module: str) -> None:
    mod = importlib.import_module(module)
    names = REFERENCED[module] or [n for n in dir(mod) if n.startswith("test_")]
    assert names
    for name in names:
        assert callable(getattr(mod, name)), f"{module}.{name} is missing"


# --- the WAN under a cloud-only placement (16.2) ------------------------------------------------------------


def cloud_only_wan_waits(ti: float, seconds: float) -> tuple[list[float], float]:
    """T4s sent to the cloud at a fixed spacing that loads the WAN to `ti`; returns each task's W_WAN.

    The car is 10 m from the access point (931 Mbps) so the radio (1.32 ms per T4) and the 1 Gbps backhaul
    (1.23 ms) keep up with the spacing even at TI 1.5 (1.55 ms), and every delay is fixed: only the WAN
    queues. B_WAN is city-wide, so one zone loads it alone here."""
    cfg = fixed_delay_config()
    s = one_zone(cfg, Fixed("cloud"), distance=10.0)
    size = cfg.task_types["T4"].input_bits
    gap = size / (ti * cfg.derived.b_wan)
    assert size / shannon_rate(10.0, cfg.radio) < gap and size / cfg.links.backhaul_bw < gap
    n = int(seconds / gap)
    for k in range(n):
        s.clock.schedule(k * gap, lambda k=k: s.zone.upload(task_now(s, "T4", seq=k + 1)))
    s.clock.run_until(seconds + 60.0)
    recs = sorted((r for r in s.closed if r.transfer is not None), key=lambda r: r.task.task_id.seq)
    assert len(recs) == n  # every task crossed the WAN
    return [r.transfer.w_wan for r in recs], gap  # type: ignore[union-attr]


def test_cloud_only_at_ti_1_5_the_wan_wait_grows_without_limit() -> None:
    """Hand calculation: with a fixed spacing g = S / (1.5 B_WAN) the k-th task (from 0) waits
    k (S / B_WAN - g) = k S / B_WAN (1 - 1 / 1.5) = k x 2.331 ms / 3 = k x 0.777 ms (S = 1.2 Mbit,
    B_WAN = 514.85 Mbps). Over 2.5 s (1 608 tasks) the last waits 1.25 s; the wait never stops rising.
    (2.5 s keeps every task inside D + L_max = 2 s: each crosses the WAN before the anchor closes it.)"""
    waits, _ = cloud_only_wan_waits(1.5, 2.5)
    step = 1.2e6 / 514_851_485.1485 * (1 - 1 / 1.5)
    assert step == pytest.approx(0.000777, rel=1e-3)
    for k in (1, 100, len(waits) - 1):
        assert waits[k] == pytest.approx(k * step, rel=0.01)
    assert waits[-1] > 1.2
    assert all(b > a for a, b in itertools.pairwise(waits))  # strictly rising: no steady state


def test_cloud_only_at_ti_0_5_the_wan_wait_stays_bounded() -> None:
    """With g = S / (0.5 B_WAN) = 4.66 ms each push (2.33 ms) ends before the next task: no task waits."""
    waits, _ = cloud_only_wan_waits(0.5, 5.0)
    assert max(waits) == 0.0


# --- one zone end to end (16.5) -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


@pytest.mark.parametrize("scenario", ["S1", "S2"])
def test_one_zone_with_greedy_keeps_the_invariants(cfg: Config, scenario: str) -> None:
    run = run_one_zone(cfg, scenario, "greedy", 10.0, 2000)
    assert run.records
    assert check_invariants(run) == []
    zone = summarise(run)["_zone"]
    assert zone["rho_fog"] == pytest.approx(cfg.scenarios[scenario].rho_fog, rel=0.15)


@pytest.mark.slow
@pytest.mark.parametrize("scenario", ["S1", "S2"])
@pytest.mark.parametrize("seed", [2000, 2001, 2002])
def test_the_sanity_run_keeps_the_invariants(cfg: Config, scenario: str, seed: int) -> None:
    """The report's run: 60 s, Greedy, one zone. rho_fog within 5 % of the target on every seed."""
    run = run_one_zone(cfg, scenario, "greedy", 60.0, seed)
    assert check_invariants(run) == []
    zone = summarise(run)["_zone"]
    assert zone["rho_fog"] == pytest.approx(cfg.scenarios[scenario].rho_fog, rel=0.05)
    assert all(r.outcome is not None for r in run.records)


def test_check_invariants_catches_each_violation(cfg: Config) -> None:
    run = run_one_zone(cfg, "S2", "greedy", 2.0, 2000)
    assert check_invariants(run) == []
    assert run.uploaded and len(run.uploaded) == len(run.records) and run.open_at_end == 0
    r = next(r for r in run.records if r.confirm is not None and r.task.cls == "hard")

    run.records.append(r)  # closed twice
    assert "a task closed twice" in check_invariants(run)
    run.records.pop()

    gone = run.records.pop(0)  # uploaded, never closed: invisible to a check of the records alone
    assert f"{gone.task.task_id} never closed" in check_invariants(run)
    run.records.insert(0, gone)

    run.uploaded.remove(r.task.task_id)
    assert f"{r.task.task_id} closed but never uploaded" in check_invariants(run)
    run.uploaded.append(r.task.task_id)

    run.open_at_end = 1
    assert "1 anchor rows still open" in check_invariants(run)
    run.open_at_end = 0

    for shift in (1e-9, -1e-9):  # L must equal its parts, either way off by 1 ns is caught
        r.parts["run"] += shift
        assert any("!= parts" in v for v in check_invariants(run))
        r.parts["run"] -= shift

    saved = (r.outcome, r.confirm)
    assert r.confirm is not None
    r.outcome, r.confirm = "on_time", replace(r.confirm, latency=r.task.deadline + 1e-3)
    r.parts["run"] += r.task.deadline + 1e-3 - saved[1].latency  # type: ignore[union-attr]  # keep L = parts
    assert any("after D" in v for v in check_invariants(run))
    r.parts["run"] -= r.task.deadline + 1e-3 - saved[1].latency  # type: ignore[union-attr]
    r.outcome, r.confirm = saved

    r.outcome = None
    assert any("closed with no outcome" in v for v in check_invariants(run))
    r.outcome = saved[0]
    assert check_invariants(run) == []


@pytest.mark.parametrize("n_cars", [50, 250])
def test_a_zones_load_does_not_depend_on_the_fleet_size(cfg: Config, n_cars: int) -> None:
    # 50 cars = a zone of a 200-car city, 250 = of a 1000-car city: lambda_car = 74.26 / n_cars, so the zone
    # gets 74.26 tasks/s in S2 either way (30 s: about 2 230 tasks, sd 2.1 %).
    run = run_one_zone(cfg, "S2", "greedy", 30.0, 2000, n_cars=n_cars)
    assert len(run.uploaded) / 30.0 == pytest.approx(cfg.derived.arrival_rate_per_zone["S2"], rel=0.07)


def test_the_one_zone_run_is_deterministic(cfg: Config) -> None:
    def digest(seed: int) -> str:
        s = summarise(run_one_zone(cfg, "S2", "greedy", 5.0, seed))
        return json.dumps(s, sort_keys=True, default=str)

    assert digest(2000) == digest(2000)
    assert digest(2000) != digest(2001)


def test_random_runs_a_zone_too(cfg: Config) -> None:
    run = run_one_zone(cfg, "S1", "random", 5.0, 2000)
    assert check_invariants(run) == []
    rows = summarise(run)
    assert sum(rows[t]["cloud"] for t in ("T1", "T3", "T4")) > 0  # Random sends some soft or hard tasks out
    assert not math.isnan(rows["T1"]["mean_latency_ms"])
