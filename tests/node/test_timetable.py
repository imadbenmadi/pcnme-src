"""Time table and run-time draws (PLAN M1.4; final-design 14.4, 12.2, 18; CHANGES C3; QUESTIONS Q7)."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats

from pcnme.config import Config, ConfigError, config_from_mapping, load_config, load_yaml
from pcnme.node.timetable import Deterministic, Exponential, RunTimeTable, deterministic, exponential
from pcnme.sim.rng import RandomStreams

TASK = ("c03-047", 0, 1187)  # task_id = (car_id, epoch, seq), final-design 5.5


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


@pytest.fixture
def streams(cfg: Config) -> RandomStreams:
    return RandomStreams.from_config(cfg, 2000, "training")


# --- R1, R5: rows -------------------------------------------------------------------------------------------


def test_a_row_holds_mean_quantiles_longest_step_and_energy(cfg: Config) -> None:
    row = RunTimeTable.from_config(cfg, exponential).row("T1", "F-GPU", "gpu")
    assert row is not None
    # Placeholders of 18: mean 8 ms; p5 0.75x (C5), p95 1.3x, longest step 0.5x; 25 W per F-GPU GPU slot.
    assert (row.mean, row.low, row.high, row.longest_step) == pytest.approx((0.008, 0.006, 0.0104, 0.004))
    assert row.energy_per_run == pytest.approx(25 * 0.008)  # J of dynamic energy per run
    assert row.power == pytest.approx(25.0)  # p = energy per run / run time (12.2)


def test_rows_follow_the_hardware_class_and_slot_type(cfg: Config) -> None:
    table = RunTimeTable.from_config(cfg, exponential)
    cpu = table.row("T3", "F-CPU", "cpu")
    cloud = table.row("T2", "cloud", "gpu")
    assert cpu is not None and cloud is not None
    assert (cpu.mean, cpu.energy_per_run) == pytest.approx((0.110, 20 * 0.110))
    assert (cloud.mean, cloud.energy_per_run) == pytest.approx((0.200, 150 * 0.200))


def test_each_slot_type_has_its_own_power(cfg: Config) -> None:
    # A node with both slot types: the F-GPU CPU slot draws 20 W, its GPU slot 25 W; the cloud 30 W and 150 W.
    table = RunTimeTable.from_config(cfg, exponential)
    expected = {
        ("F-GPU", "gpu"): 25 * 0.008,
        ("F-GPU", "cpu"): 20 * 0.090,
        ("cloud", "gpu"): 150 * 0.004,
        ("cloud", "cpu"): 30 * 0.045,
    }
    for (hw, q), energy in expected.items():
        row = table.row("T1", hw, q)
        assert row is not None
        assert row.energy_per_run == pytest.approx(energy), (hw, q)


# --- R2: a missing row = cannot run there -------------------------------------------------------------------


def test_a_missing_row_means_cannot_run_there(cfg: Config) -> None:
    table = RunTimeTable.from_config(cfg, exponential)
    assert table.row("T2", "F-GPU", "cpu") is None  # GPU only
    assert table.row("T2", "F-CPU", "cpu") is None
    assert table.slot_types("T2", "F-GPU") == ("gpu",)
    assert table.slot_types("T2", "F-CPU") == ()
    assert table.slot_types("T1", "F-GPU") == ("gpu", "cpu")


# --- R3: draws keyed by task (CHANGES C3) -------------------------------------------------------------------


def test_a_draw_is_the_inverse_cdf_of_the_keyed_run_times_uniform(
    cfg: Config, streams: RandomStreams
) -> None:
    table = RunTimeTable.from_config(cfg, exponential)
    row = table.row("T1", "F-GPU", "gpu")
    assert row is not None
    u = streams.uniform("run_times", *TASK, 1)
    assert table.draw(streams, row, TASK, attempt=1) == pytest.approx(-0.008 * math.log1p(-u), rel=1e-15)


def test_a_task_keeps_its_quantile_on_every_node_and_slot_type(cfg: Config, streams: RandomStreams) -> None:
    # The same (task_id, attempt) gives the same u, so with one shape the run time scales with the row's mean:
    # T1 on a CPU slot (90 ms) is exactly 90 / 8 times its GPU-slot draw, and the cloud GPU (4 ms) half of it.
    table = RunTimeTable.from_config(cfg, exponential)
    rows = {
        (hw, q): table.row("T1", hw, q) for hw, q in [("F-GPU", "gpu"), ("F-GPU", "cpu"), ("cloud", "gpu")]
    }
    runs = {k: table.draw(streams, r, TASK, 1) for k, r in rows.items() if r is not None}
    assert runs[("F-GPU", "cpu")] == pytest.approx(runs[("F-GPU", "gpu")] * 90 / 8, rel=1e-12)
    assert runs[("cloud", "gpu")] == pytest.approx(runs[("F-GPU", "gpu")] * 4 / 8, rel=1e-12)


def test_a_retry_is_a_new_draw(cfg: Config, streams: RandomStreams) -> None:
    table = RunTimeTable.from_config(cfg, exponential)
    row = table.row("T4", "F-GPU", "gpu")
    assert row is not None
    assert table.draw(streams, row, TASK, 1) != table.draw(streams, row, TASK, 2)


def test_draws_do_not_depend_on_order_or_other_draws(cfg: Config) -> None:
    table = RunTimeTable.from_config(cfg, exponential)
    row = table.row("T1", "F-GPU", "gpu")
    assert row is not None
    keys = [(("c01", 0, s), a) for s in range(30) for a in (1, 2)]
    first = RandomStreams.from_config(cfg, 2000, "training")
    forward = [table.draw(first, row, k, a) for k, a in keys]
    second = RandomStreams.from_config(cfg, 2000, "training")
    second.generator("arrivals", "c01").random(100)  # unrelated draws change nothing
    backward = [table.draw(second, row, k, a) for k, a in reversed(keys)]
    assert forward == backward[::-1]


def test_exponential_draws_follow_their_distribution(cfg: Config, streams: RandomStreams) -> None:
    table = RunTimeTable.from_config(cfg, exponential)
    row = table.row("T4", "F-GPU", "gpu")  # 25 ms
    assert row is not None
    runs = np.array([table.draw(streams, row, ("c07", 0, s), 1) for s in range(20_000)])
    assert stats.kstest(runs, "expon", args=(0, 0.025)).pvalue > 1e-3
    assert runs.mean() == pytest.approx(0.025, rel=0.03)


def test_deterministic_draws_are_the_mean(cfg: Config, streams: RandomStreams) -> None:
    table = RunTimeTable.from_config(cfg, deterministic)
    row = table.row("T2", "F-GPU", "gpu")
    assert row is not None
    assert {table.draw(streams, row, ("c02", 0, s), 1) for s in range(50)} == {0.720}


# --- R6: the shape is a separate, swappable piece (Q7) ------------------------------------------------------


def test_the_distribution_shape_is_chosen_by_the_caller(cfg: Config) -> None:
    det = RunTimeTable.from_config(cfg, deterministic).row("T1", "F-GPU", "gpu")
    exp = RunTimeTable.from_config(cfg, exponential).row("T1", "F-GPU", "gpu")
    assert det is not None and exp is not None
    assert isinstance(det.distribution, Deterministic) and isinstance(exp.distribution, Exponential)
    from pcnme.node.timetable import Lognormal

    default = RunTimeTable.from_config(cfg).row("T1", "F-GPU", "gpu")  # C5: lognormal by default
    assert default is not None and isinstance(default.distribution, Lognormal)


@pytest.mark.parametrize("u", [0.0, 1.0, -0.1, 1.5, math.nan])
def test_inverse_cdf_refuses_u_outside_the_open_interval(u: float) -> None:
    for shape in (Deterministic(0.01), Exponential(0.01)):
        with pytest.raises(ValueError, match="u"):
            shape.inverse_cdf(u)


def test_exponential_inverse_cdf_is_increasing_and_has_the_right_quantiles() -> None:
    e = Exponential(0.025)
    assert e.inverse_cdf(0.5) == pytest.approx(0.025 * math.log(2))
    us = np.linspace(0.001, 0.999, 500)
    assert np.all(np.diff([e.inverse_cdf(float(u)) for u in us]) > 0)


# --- R4: quantiles (14.4: the table's p5 and p95 give E_low and E_high) -------------------------------------


def test_config_quantiles_must_be_the_tables_p5_and_p95(default_config_path: Path) -> None:
    data: dict[str, Any] = copy.deepcopy(load_yaml(default_config_path))
    data["timing"]["quantiles"] = {"low": 0.10, "high": 0.90}
    with pytest.raises(ConfigError) as err:
        RunTimeTable.from_config(config_from_mapping(data), exponential)
    assert err.value.key == "timing.quantiles"
