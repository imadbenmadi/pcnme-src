"""16.4 sensitivity factors (QUESTIONS Q48): each scales one quantity after the calibration, the calibrated
values stay those of factor 1."""

from __future__ import annotations

import pytest

from pcnme.config import ConfigError, load_config
from pcnme.experiments.city import City
from tests.zone.harness import fixed_delay_config

BASE = load_config()


def test_the_default_changes_nothing() -> None:
    assert BASE.sweep.is_default
    assert fixed_delay_config(sweep__cars=1.0).task_types == fixed_delay_config().task_types
    with pytest.raises(ConfigError):
        fixed_delay_config(sweep__deadline=0.0)


def test_the_arrival_rate_scales_with_b_wan_kept() -> None:
    c = fixed_delay_config(sweep__arrival_rate=1.5)
    base = fixed_delay_config()
    for s, r in base.derived.arrival_rate_per_zone.items():
        assert c.derived.arrival_rate_per_zone[s] == pytest.approx(1.5 * r)
    assert c.derived.b_wan == base.derived.b_wan
    assert c.arrival_rate_per_car("S2", 200) == pytest.approx(1.5 * base.arrival_rate_per_car("S2", 200))


def test_input_size_and_deadline_scale_alone() -> None:
    base = fixed_delay_config()
    c = fixed_delay_config(sweep__input_size=0.5, sweep__deadline=1.5)
    for name, t in base.task_types.items():
        u = c.task_types[name]
        assert u.input_bits == pytest.approx(0.5 * t.input_bits) and u.result_bits == t.result_bits
        assert u.deadline["loose"] == pytest.approx(1.5 * t.deadline["loose"])
        assert u.l_max["tight"] == pytest.approx(1.5 * t.l_max["tight"])
    assert c.derived.b_wan == base.derived.b_wan and c.derived.e_ref == base.derived.e_ref
    assert c.derived.arrival_rate_per_zone == base.derived.arrival_rate_per_zone


def test_fog_capacity_scales_fog_slots_only() -> None:
    c = fixed_delay_config(sweep__fog_capacity=0.5)
    assert (c.hardware["F-GPU"].c_gpu, c.hardware["F-GPU"].c_cpu) == (1, 1)
    assert (c.hardware["F-CPU"].c_gpu, c.hardware["F-CPU"].c_cpu) == (0, 1)  # no GPU stays no GPU
    assert c.hardware["cloud"] == BASE.hardware["cloud"]
    big = fixed_delay_config(sweep__fog_capacity=1.5)
    assert (big.hardware["F-GPU"].c_gpu, big.hardware["F-GPU"].c_cpu) == (3, 3)
    assert c.derived.arrival_rate_per_zone == BASE.derived.arrival_rate_per_zone  # the load is the same


def test_wan_bandwidth_scales_b_wan() -> None:
    assert fixed_delay_config(sweep__wan_bandwidth=0.5).derived.b_wan == pytest.approx(
        0.5 * BASE.derived.b_wan
    )


def test_more_cars_at_the_same_rate_per_car() -> None:
    base = City(fixed_delay_config(), "S2", "greedy", 2000, n_cars=40)
    more = City(fixed_delay_config(sweep__cars=1.5), "S2", "greedy", 2000, n_cars=40)
    assert (len(base.apps), len(more.apps)) == (40, 60)
    assert more.apps[0].car._rate == base.apps[0].car._rate  # the load grows with the fleet
