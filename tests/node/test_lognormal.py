"""Lognormal run times fitted to each row's mean and p95 (CHANGES C5; QUESTIONS Q7; final-design 14.4, 18)."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import integrate, stats

from pcnme.config import Config, config_from_mapping, load_config, load_yaml
from pcnme.node.timetable import Lognormal, RunTimeTable, lognormal

Z95 = stats.norm.ppf(0.95)  # 1.6449


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


def test_sigma_and_mu_by_hand() -> None:
    # p95 / mean = 1.3: sigma^2 / 2 - 1.6449 sigma + ln 1.3 = 0, the smaller root: sigma = 1.6449 -
    # sqrt(1.6449^2 - 2 ln 1.3) = 1.6449 - sqrt(2.7055 - 0.5247) = 0.1681; mu = ln m - sigma^2 / 2.
    shape = Lognormal.fit(mean=0.008, p95=0.0104)
    expected_sigma = Z95 - math.sqrt(Z95**2 - 2 * math.log(1.3))
    assert shape.sigma == pytest.approx(expected_sigma, rel=1e-12)
    assert shape.sigma == pytest.approx(0.1681, abs=1e-4)
    assert shape.mu == pytest.approx(math.log(0.008) - expected_sigma**2 / 2, rel=1e-12)


@pytest.mark.parametrize("mean", [0.008, 0.090, 0.720])  # T1 GPU, T1 CPU, T2 GPU (config placeholders)
def test_the_mean_and_p95_are_exact(mean: float) -> None:
    shape = Lognormal.fit(mean=mean, p95=1.3 * mean)
    assert shape.inverse_cdf(0.95) == pytest.approx(1.3 * mean, rel=1e-12)
    # The mean, integrated over u in (0, 1): E[X] = integral of the inverse CDF.
    integral, _ = integrate.quad(shape.inverse_cdf, 1e-12, 1 - 1e-12, limit=200)
    assert integral == pytest.approx(mean, rel=1e-6)


def test_the_fitted_p5_is_0_748_of_the_mean() -> None:
    # C5: why the placeholder p5 factor is 0.75.
    shape = Lognormal.fit(mean=1.0, p95=1.3)
    assert shape.inverse_cdf(0.05) == pytest.approx(0.748, abs=5e-4)
    assert shape.p5 == pytest.approx(shape.inverse_cdf(0.05), rel=1e-12)


def test_draws_follow_the_fitted_lognormal(cfg: Config) -> None:
    from pcnme.sim.rng import RandomStreams

    table = RunTimeTable.from_config(cfg)  # lognormal is the default shape now (C5)
    row = table.row("T4", "F-GPU", "gpu")
    assert row is not None and isinstance(row.distribution, Lognormal)
    streams = RandomStreams.from_config(cfg, 2000, "training")
    runs = np.array([table.draw(streams, row, ("c07", 0, s), 1) for s in range(20_000)])
    d = row.distribution
    assert stats.kstest(runs, "lognorm", args=(d.sigma, 0, math.exp(d.mu))).pvalue > 1e-3
    assert runs.mean() == pytest.approx(0.025, rel=0.01)
    assert np.quantile(runs, 0.95) == pytest.approx(0.0325, rel=0.02)


def test_p95_equal_to_the_mean_is_a_fixed_run_and_impossible_ratios_are_refused() -> None:
    assert Lognormal.fit(mean=0.01, p95=0.01).inverse_cdf(0.3) == pytest.approx(0.01)
    with pytest.raises(ValueError, match="p95"):
        Lognormal.fit(mean=0.01, p95=0.009)  # p95 below the mean
    with pytest.raises(ValueError, match="p95"):
        Lognormal.fit(mean=0.01, p95=0.05)  # 5x: no lognormal has p95 / mean > exp(z95^2 / 2) = 3.87


def test_the_placeholder_p5_factor_is_0_75(cfg: Config) -> None:
    row = cfg.time_table.row("T1", "F-GPU", "gpu")
    assert row is not None and row.p5 == pytest.approx(0.75 * 0.008)


def test_e_bar_is_still_11_7(cfg: Config) -> None:
    # ē uses p95 only (12.3), which the fit keeps exact.
    assert round(cfg.derived.e_bar, 2) == 11.70


def test_the_p5_report_puts_the_fitted_p5_next_to_the_rows_p5(default_config_path: Path) -> None:
    # C5: for measured rows, report the fitted p5 next to the measured one. One row is made "measured".
    data: dict[str, Any] = copy.deepcopy(load_yaml(default_config_path))
    data["time_table_placeholder"]["T1"]["F-GPU"]["gpu"] = {"mean": 0.008, "p5": 0.0065, "p95": 0.0110}
    table = RunTimeTable.from_config(config_from_mapping(data), lognormal)
    report = {(r.task, r.hw, r.slot): r for r in table.p5_report()}
    measured = report[("T1", "F-GPU", "gpu")]
    assert measured.measured_p5 == pytest.approx(0.0065)
    assert measured.fitted_p5 == pytest.approx(Lognormal.fit(0.008, 0.0110).p5, rel=1e-12)
    placeholder = report[("T3", "F-GPU", "gpu")]
    assert placeholder.measured_p5 is None  # no measured p5: the 0.75 factor fills it
    assert placeholder.fitted_p5 == pytest.approx(0.748 * 0.010, abs=5e-6)
