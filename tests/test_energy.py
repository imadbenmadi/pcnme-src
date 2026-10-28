"""Energy accounting (PLAN M2.7; final-design 12.1-12.5)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from pcnme.config import Config, load_config
from pcnme.energy import EnergyModel
from tests.zone.harness import Fixed, fixed_delay_config, one_zone, task_now

E_BH, E_WAN, E_DOWN = 0.141e-9 * 1.5, 0.228e-9 * 1.5, 4.2e-9  # J/bit (18; e_down not x PUE, Q3)


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


@pytest.fixture(scope="module")
def em(cfg: Config) -> EnergyModel:
    return EnergyModel(cfg)


# --- the three parts (12.2) ---------------------------------------------------------------------------------


def test_forwarding_energy(em: EnergyModel) -> None:
    s = 1.2e6
    assert em.forwarding("here", s) == 0.0
    assert em.forwarding("neighbour", s) == pytest.approx(E_BH * s)
    assert em.forwarding("cloud", s) == pytest.approx((E_BH + E_WAN) * s)


def test_return_energy_per_delivery(em: EnergyModel) -> None:
    s_r = 3 * 8000
    assert em.ret("fog", s_r) == pytest.approx((2 * E_BH + E_DOWN) * s_r)  # 2 backhaul crossings
    assert em.ret("cloud", s_r) == pytest.approx((E_WAN + E_BH + E_DOWN) * s_r)  # 1 WAN + 1 backhaul
    assert em.ret("fog", s_r, deliveries=2) == pytest.approx(2 * em.ret("fog", s_r))  # each republish pays


def test_e_hat_is_capped_at_e_bar(em: EnergyModel, cfg: Config) -> None:
    assert em.e_hat("T1", 0.3001109) == pytest.approx(1.0, abs=1e-6)  # E = E_ref
    assert em.e_hat("T1", 1e6) == pytest.approx(cfg.derived.e_bar)  # 12.3: capped at ē


# --- E_ref and ê per type equal tools/calc_defaults.py (M2 acceptance) --------------------------------------


def test_e_ref_and_the_worst_ratios_equal_calc_defaults(
    cfg: Config, em: EnergyModel, repo_root: Path
) -> None:
    out = subprocess.run(
        [sys.executable, str(repo_root / "tools" / "calc_defaults.py")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for name in cfg.task_types:
        line = re.search(rf"  {name}: E_ref ([\d.]+) J; worst-case ratios (.*); network", out)
        assert line, out
        assert f"{em.e_ref(name):.3f}" == line.group(1)
        ratios = dict(re.findall(r"(fog GPU|fog CPU|cloud) ([\d.]+)", line.group(2)))
        # The same worst cases from the model's estimate (12.4) at p95: a neighbour's GPU and CPU slot, cloud
        # GPU.
        p95 = {q: r.p95 for q in ("gpu", "cpu") if (r := cfg.time_table.row(name, "F-GPU", q))}
        assert (
            f"{em.estimate(name, 'neighbour', 'F-GPU', 'gpu', p95['gpu']) / em.e_ref(name):.2f}"
            == ratios["fog GPU"]
        )
        if "cpu" in p95:
            got = em.estimate(name, "neighbour", "F-GPU", "cpu", p95["cpu"]) / em.e_ref(name)
            assert f"{got:.2f}" == ratios["fog CPU"]
        cloud_p95 = cfg.time_table.row(name, "cloud", "gpu").p95  # type: ignore[union-attr]
        assert (
            f"{em.estimate(name, 'cloud', 'cloud', 'gpu', cloud_p95) / em.e_ref(name):.2f}" == ratios["cloud"]
        )


# --- the per-attempt table (12.5), on real runs -------------------------------------------------------------


def test_a_task_delivered_here_at_its_mean_run_costs_exactly_e_ref(em: EnergyModel) -> None:
    # 12.3: E_ref is the energy of running it here on the reference class, so ê = 1.
    s = one_zone(fixed_delay_config(), Fixed("here"))
    s.zone.upload(task_now(s, "T1"))
    s.clock.run_until(1.0)
    (rec,) = s.closed
    e = em.of_record(rec)
    assert (e.forwarding, e.execution, e.ret) == (
        0.0,
        pytest.approx(1.5 * 25 * 0.008),
        pytest.approx(em.ret("fog", 24000)),
    )
    assert em.e_hat("T1", e.total) == pytest.approx(1.0, rel=1e-12)


def test_a_cloud_delivery_by_hand(em: EnergyModel) -> None:
    """T4 to the cloud: forwarding (e_bh + e_wan) x 1.2 Mbit = 0.5544 mJ; execution 1.1 x 150 W x 12 ms =
    1.98 J; return (e_wan + e_bh + e_down) x 24 kbit = 0.1135 mJ; E_k = 1.9807 J;
    ê = 1.9807 / 0.9376 = 2.11."""
    s = one_zone(fixed_delay_config(), Fixed("cloud"))
    s.zone.upload(task_now(s, "T4"))
    s.clock.run_until(5.0)
    (rec,) = s.closed
    e = em.of_record(rec)
    assert e.forwarding == pytest.approx((E_BH + E_WAN) * 1.2e6)
    assert e.execution == pytest.approx(1.1 * 150 * 0.012)
    assert e.ret == pytest.approx((E_WAN + E_BH + E_DOWN) * 24000)
    assert em.e_hat("T4", e.total) == pytest.approx(e.total / em.e_ref("T4"))
    assert em.e_hat("T4", e.total) == pytest.approx(2.11, abs=0.01)


def test_a_refused_attempt_costs_its_forwarding_only(em: EnergyModel) -> None:
    s = one_zone(fixed_delay_config(), Fixed("cloud"))
    s.zone.upload(task_now(s, "T1"))  # refused late at the cloud (M2.6)
    s.clock.run_until(1.0)
    e = em.of_record(s.closed[0])
    assert (e.forwarding, e.execution, e.ret) == (pytest.approx((E_BH + E_WAN) * 1.2e6), 0.0, 0.0)


def test_skipped_and_aborted_attempts(em: EnergyModel) -> None:
    from pcnme.node.timetable import Deterministic

    s = one_zone(
        fixed_delay_config(), Fixed("here"), hw_name="F-CPU", shape=lambda r: Deterministic(15 * r.mean)
    )
    for seq in (1, 2, 3):
        s.zone.upload(task_now(s, "T4", seq=seq))
    s.clock.run_until(10.0)
    by = {r.task.task_id.seq: r for r in s.closed}
    aborted = em.of_record(by[1])
    assert aborted.ret == 0.0 and aborted.forwarding == 0.0
    assert aborted.execution == pytest.approx(1.5 * 20 * by[1].report.busy)  # type: ignore[union-attr]  # tail in
    skipped = em.of_record(by[3])
    assert (skipped.forwarding, skipped.execution, skipped.ret) == (
        0.0,
        0.0,
        0.0,
    )  # here: no forwarding either


def test_drops_with_no_attempt_cost_nothing(em: EnergyModel) -> None:
    s = one_zone(fixed_delay_config(), Fixed(None))  # every option masked
    s.zone.upload(task_now(s, "T1"))
    s.clock.run_until(1.0)
    assert em.of_record(s.closed[0]).total == 0.0


# --- reporting only (12.1) ----------------------------------------------------------------------------------


def test_the_car_upload_energy_is_airtime_only(em: EnergyModel) -> None:
    # P_tx x S / R_car = 0.158 W x 1.2 Mbit / 539.1 Mbps = 0.3517 mJ; it is not part of E_k.
    s = one_zone(fixed_delay_config(), Fixed("here"))
    s.zone.upload(task_now(s, "T1"))
    s.clock.run_until(1.0)
    rec = s.closed[0]
    assert em.car_upload(rec) == pytest.approx(0.158 * 1.2e6 / rec.rate)
    assert em.car_upload(rec) == pytest.approx(0.0003517, rel=1e-3)
