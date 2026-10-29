"""Config loader and derived values (PLAN M0.2; final-design 18).

Expected values are computed by hand from the numbers in config/default.yaml, written out in full so a
reader can check them, and cross-checked against the output of tools/calc_defaults.py.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pcnme.config import Config, ConfigError, config_from_mapping, load_config, load_yaml

# Hand calculations from config/default.yaml (placeholders of 27 Oct).
MEAN_GPU_RUN = 0.73 * 0.008 + 0.15 * 0.010 + 0.10 * 0.025 + 0.02 * 0.720  # 0.02424 s, whole mix (16.2, Q1)
ZONE_CAPACITY = 2 / MEAN_GPU_RUN  # c_GPU of F-GPU / mean: 82.5 tasks/s
ACTIVE_ZONES = 4  # z1-z4; z5 joins
SOFT_BITS_PER_TASK = 0.10 * 150 * 8000 + 0.02 * 9000 * 8000  # T4 + T2 input bits by count: 1.56e6
B_WAN = 0.9 * ZONE_CAPACITY * ACTIVE_ZONES * SOFT_BITS_PER_TASK / 0.9  # about 515 Mbps
E_BH = 0.141e-9 * 1.5  # J/bit, x PUE_network (18)
E_WAN = 0.228e-9 * 1.5
E_DOWN = 4.2e-9  # not x PUE_network (Q3)
E_RET_HERE_SMALL = (2 * E_BH + E_DOWN) * 3 * 8000  # 3 KB result through 2 backhaul crossings + downlink
E_REF = {
    "T1": 1.5 * 25 * 0.008 + E_RET_HERE_SMALL,
    "T3": 1.5 * 25 * 0.010 + E_RET_HERE_SMALL,
    "T4": 1.5 * 25 * 0.025 + E_RET_HERE_SMALL,
    "T2": 1.5 * 25 * 0.720 + (2 * E_BH + E_DOWN) * 10 * 8000,
}
# Worst case: T1 sent to a neighbour's CPU slot at p95 = 1.3 x 90 ms (12.3).
E_BAR = (1.5 * 20 * 1.3 * 0.090 + E_BH * 150 * 8000 + E_RET_HERE_SMALL) / E_REF["T1"]  # 11.70


@pytest.fixture(scope="module")
def raw(default_config_path: Path) -> dict[str, Any]:
    return load_yaml(default_config_path)


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


def edited(raw: dict[str, Any], edit: Callable[[dict[str, Any]], None]) -> Config:
    data = copy.deepcopy(raw)
    edit(data)
    return config_from_mapping(data)


def parent_of(data: dict[str, Any], dotted: str) -> tuple[dict[str, Any], str]:
    *head, last = dotted.split(".")
    node = data
    for part in head:
        node = node[part]
    return node, last


def delete(dotted: str) -> Callable[[dict[str, Any]], None]:
    def edit(data: dict[str, Any]) -> None:
        node, last = parent_of(data, dotted)
        del node[last]

    return edit


def assign(dotted: str, value: object) -> Callable[[dict[str, Any]], None]:
    def edit(data: dict[str, Any]) -> None:
        node, last = parent_of(data, dotted)
        node[last] = value

    return edit


def rename(dotted: str, new_name: str) -> Callable[[dict[str, Any]], None]:
    def edit(data: dict[str, Any]) -> None:
        node, last = parent_of(data, dotted)
        node[new_name] = node.pop(last)

    return edit


# --- Reading YAML -------------------------------------------------------------------------------------------


def test_yaml_reads_exponent_floats_as_numbers(tmp_path: Path) -> None:
    # PyYAML (YAML 1.1) reads 40.0e6 and 1e9 as strings; the config writes bandwidths that way.
    path = tmp_path / "c.yaml"
    path.write_text("a: 40.0e6\nb: 1e9\nc: 150\nd: 1000-1999\ne: 1.0e-4\nf: -2.5\n", encoding="utf-8")
    data = load_yaml(path)
    assert data == {"a": 40.0e6, "b": 1e9, "c": 150, "d": "1000-1999", "e": 1.0e-4, "f": -2.5}
    assert isinstance(data["a"], float) and isinstance(data["c"], int)


def test_default_path_loads_the_repo_config(cfg: Config) -> None:
    assert load_config() == cfg


def test_loading_twice_gives_equal_configs(default_config_path: Path) -> None:
    assert load_config(default_config_path) == load_config(default_config_path)


# --- Derived values (PLAN M0.2) -----------------------------------------------------------------------------


def test_mean_gpu_run_of_the_whole_mix(cfg: Config) -> None:
    # final-design 16.2 (whole mix, boulders included: QUESTIONS Q1)
    assert cfg.derived.mean_gpu_run == pytest.approx(MEAN_GPU_RUN, rel=1e-12)
    assert cfg.derived.mean_gpu_run == pytest.approx(0.02424, rel=1e-12)


def test_zone_capacity(cfg: Config) -> None:
    assert cfg.derived.zone_capacity == pytest.approx(ZONE_CAPACITY, rel=1e-12)
    assert round(cfg.derived.zone_capacity, 1) == 82.5


def test_active_zones_leave_out_the_joining_zone(cfg: Config) -> None:
    assert cfg.layout.active_zones == ("z1", "z2", "z3", "z4")
    assert cfg.layout.joining_zones == ("z5",)


def test_b_wan(cfg: Config) -> None:
    # final-design 18 (Links): cloud-eligible bits per second at S2 / 0.9
    assert cfg.derived.b_wan == pytest.approx(B_WAN, rel=1e-12)
    assert cfg.derived.b_wan == pytest.approx(515e6, rel=1e-3)


def test_t2_tight_deadline_is_1_1_times_e_strong(cfg: Config) -> None:
    # final-design 18 (Workload): 1.1 x mean run on the strongest fog class, fastest slot type (720 ms)
    assert cfg.task_types["T2"].deadline["tight"] == pytest.approx(1.1 * 0.720, rel=1e-12)
    assert cfg.task_types["T2"].deadline["tight"] == pytest.approx(0.792, rel=1e-12)


def test_e_strong_is_the_fastest_slot_on_the_strongest_class(cfg: Config) -> None:
    assert dict(cfg.derived.e_strong) == pytest.approx({"T1": 0.008, "T3": 0.010, "T4": 0.025, "T2": 0.720})


def test_only_t2_with_tight_deadlines_is_a_boulder(cfg: Config) -> None:
    # final-design 7.1: boulder iff E_strong >= alpha x D; 0.720 >= 0.8 x 0.792 = 0.634
    boulders = {(t, mode) for t in cfg.task_types for mode in ("loose", "tight") if cfg.is_boulder(t, mode)}
    assert boulders == {("T2", "tight")}


def test_boulder_rule_uses_alpha(raw: dict[str, Any]) -> None:
    # alpha = 0.2: T2 loose 0.720 >= 0.2 x 3.0 = 0.6 -> boulder; T4 loose 0.025 >= 0.2 x 1.0? no -> pebble.
    cfg = edited(raw, assign("tof.alpha", 0.2))
    assert cfg.is_boulder("T2", "loose")
    assert not cfg.is_boulder("T4", "loose")


def test_boulder_rule_is_inclusive(raw: dict[str, Any]) -> None:
    # final-design 7.1 uses >=: E_strong exactly alpha x D is a boulder (T1: 0.008 = 0.8 x 0.010).
    cfg = edited(raw, assign("task_types.T1.deadline.tight", 0.010))
    assert cfg.is_boulder("T1", "tight")


def test_l_max_equals_d(cfg: Config) -> None:
    for task in cfg.task_types.values():
        assert dict(task.l_max) == dict(task.deadline)


@pytest.mark.parametrize(
    ("scenario", "rho"), [("S1", 0.5), ("S2", 0.9), ("S3", 1.5), ("S4", 0.9), ("S5", 0.9), ("S6", 0.9)]
)
def test_arrival_rates(cfg: Config, scenario: str, rho: float) -> None:
    # final-design 18 (Workload): lambda_car = rho_fog x c_GPU / (mean GPU run x cars per zone).
    # S4-S6 keep S2's rate (Q1).
    per_zone = rho * ZONE_CAPACITY
    assert cfg.derived.arrival_rate_per_zone[scenario] == pytest.approx(per_zone, rel=1e-12)
    assert cfg.arrival_rate_per_car(scenario, 200) == pytest.approx(per_zone / 50, rel=1e-12)
    assert cfg.arrival_rate_per_car(scenario, 1000) == pytest.approx(per_zone / 250, rel=1e-12)


def test_arrival_rates_match_the_design_numbers(cfg: Config) -> None:
    # final-design 18: S2 = 74 tasks/s per zone, 1.5 per car at 200 cars, 0.3 at 1 000
    assert round(cfg.derived.arrival_rate_per_zone["S2"]) == 74
    assert round(cfg.arrival_rate_per_car("S2", 200), 1) == 1.5
    assert round(cfg.arrival_rate_per_car("S2", 1000), 1) == 0.3


def test_arrival_rate_per_car_rejects_bad_arguments(cfg: Config) -> None:
    with pytest.raises(KeyError, match="S9"):
        cfg.arrival_rate_per_car("S9", 200)
    with pytest.raises(ValueError, match="n_cars"):
        cfg.arrival_rate_per_car("S2", 0)


@pytest.mark.parametrize(
    ("scenario", "ti"), [("S1", 0.5), ("S2", 0.9), ("S3", 1.5), ("S4", 0.9), ("S5", 0.9), ("S6", 0.9)]
)
def test_ti_wan_equals_rho_fog(cfg: Config, scenario: str, ti: float) -> None:
    # final-design 16.2: one B_WAN gives TI_WAN = rho_fog. In S4-S6 T2 is a boulder but also soft, so the
    # cloud-eligible bits (soft tasks and boulders) are the same.
    assert cfg.derived.ti_wan[scenario] == pytest.approx(ti, rel=1e-12)


def test_cloud_eligible_bits_include_hard_boulders(raw: dict[str, Any]) -> None:
    # A hard boulder is cloud-eligible too (16.2: soft tasks and boulders). Make T1 a boulder in tight
    # deadlines (8 ms >= 0.8 x 5 ms): S4's TI_WAN gains T1's bits.
    cfg = edited(raw, assign("task_types.T1.deadline.tight", 0.005))
    assert cfg.is_boulder("T1", "tight")
    t1_bits = 0.73 * 150 * 8000
    lam_city = 0.9 * ZONE_CAPACITY * ACTIVE_ZONES
    expected = lam_city * (SOFT_BITS_PER_TASK + t1_bits) / B_WAN
    assert cfg.derived.ti_wan["S4"] == pytest.approx(expected, rel=1e-12)
    assert cfg.derived.ti_wan["S2"] == pytest.approx(0.9, rel=1e-12)


@pytest.mark.parametrize("task", ["T1", "T3", "T4", "T2"])
def test_e_ref(cfg: Config, task: str) -> None:
    # final-design 12.3: PUE_fog x p x T on the reference class's fastest slot + E_ret here
    assert cfg.derived.e_ref[task] == pytest.approx(E_REF[task], rel=1e-12)


def test_e_ref_rounded_as_in_the_design(cfg: Config) -> None:
    assert {t: round(e, 3) for t, e in cfg.derived.e_ref.items()} == {
        "T1": 0.300,
        "T3": 0.375,
        "T4": 0.938,
        "T2": 27.000,
    }


def test_e_bar(cfg: Config) -> None:
    # final-design 12.3: largest E_k / E_ref over every type and option, p95 run times (18: about 11.7)
    assert cfg.derived.e_bar == pytest.approx(E_BAR, rel=1e-12)
    assert round(cfg.derived.e_bar, 2) == 11.70


def test_network_energy_per_bit_and_pue(cfg: Config) -> None:
    # final-design 18 (Energy): e_bh and e_wan x PUE 1.5; e_down not (QUESTIONS Q3)
    assert cfg.energy.e_bh == pytest.approx(E_BH, rel=1e-12)
    assert cfg.energy.e_wan == pytest.approx(E_WAN, rel=1e-12)
    assert cfg.energy.e_down == pytest.approx(E_DOWN, rel=1e-12)


def test_derived_values_equal_calc_defaults_output(cfg: Config, repo_root: Path) -> None:
    # PLAN M0.2: the loader and tools/calc_defaults.py (an independent implementation) agree.
    out = subprocess.run(
        [sys.executable, str(repo_root / "tools" / "calc_defaults.py")],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    ).stdout
    d = cfg.derived

    def number(pattern: str) -> str:
        found = re.search(pattern, out)
        assert found, f"calc_defaults.py printed no line matching {pattern!r}:\n{out}"
        return found.group(1)

    assert number(r"mean GPU run of the mix ([\d.]+) ms") == f"{d.mean_gpu_run * 1e3:.1f}"
    assert number(r"zone GPU capacity ([\d.]+) tasks/s") == f"{d.zone_capacity:.1f}"
    assert number(r"B_WAN = (\d+) Mbps") == f"{d.b_wan / 1e6:.0f}"
    assert number(r"e_bar \(placeholder\) = ([\d.]+)") == f"{d.e_bar:.2f}"
    for name, task in cfg.task_types.items():
        assert number(rf"{name}: E_ref ([\d.]+) J") == f"{d.e_ref[name]:.3f}"
        for mode in ("loose", "tight"):
            kind = "boulder" if cfg.is_boulder(name, mode) else "pebble"
            assert number(rf"{name} {mode} D = ([\d.]+ s -> \w+)") == f"{task.deadline[mode]:.3f} s -> {kind}"
    for s in ("S1", "S2", "S3"):
        assert number(rf"{s}: ([\d.]+) tasks/s per zone") == f"{d.arrival_rate_per_zone[s]:.1f}"
        assert number(rf"{s}: .*per car ([\d.]+) /") == f"{cfg.arrival_rate_per_car(s, 200):.3f}"
        assert number(rf"{s}: .*TI_WAN ([\d.]+)") == f"{d.ti_wan[s]:.2f}"


# --- Time table placeholders (QUESTIONS Q2) -----------------------------------------------------------------


def test_placeholder_rows_use_the_factors(cfg: Config) -> None:
    row = cfg.time_table.row("T1", "F-GPU", "cpu")
    assert row is not None
    assert (row.mean, row.p5, row.p95, row.longest_step) == pytest.approx((0.090, 0.0675, 0.117, 0.045))


def test_measured_values_win_over_the_factors(raw: dict[str, Any]) -> None:
    measured = {"mean": 0.090, "p95": 0.200, "longest_step": 0.030}
    cfg = edited(raw, assign("time_table_placeholder.T1.F-GPU.cpu", measured))
    row = cfg.time_table.row("T1", "F-GPU", "cpu")
    assert row is not None
    assert (row.mean, row.p95, row.longest_step) == (0.090, 0.200, 0.030)
    assert row.p5 == pytest.approx(0.75 * 0.090)  # not measured: the factor fills it (C5)
    # e_bar follows the measured p95 (12.3)
    expected = (1.5 * 20 * 0.200 + E_BH * 150 * 8000 + E_RET_HERE_SMALL) / E_REF["T1"]
    assert cfg.derived.e_bar == pytest.approx(expected, rel=1e-12)


def test_a_missing_row_means_cannot_run_there(cfg: Config) -> None:
    assert cfg.time_table.row("T2", "F-GPU", "cpu") is None
    assert cfg.time_table.row("T2", "F-CPU", "cpu") is None
    assert cfg.time_table.slot_types("T2", "F-GPU") == ("gpu",)
    assert cfg.time_table.slot_types("T1", "F-GPU") == ("gpu", "cpu")


def test_inconsistent_measured_row_is_refused(raw: dict[str, Any]) -> None:
    bad = {"mean": 0.090, "p5": 0.100}
    with pytest.raises(ConfigError) as err:
        edited(raw, assign("time_table_placeholder.T1.F-GPU.cpu", bad))
    assert err.value.key == "time_table_placeholder.T1.F-GPU.cpu"


# --- Units at the edge (CLAUDE.md) --------------------------------------------------------------------------


def test_units_are_converted_on_load(cfg: Config) -> None:
    assert cfg.task_types["T1"].input_bits == 150 * 8000
    assert cfg.task_types["T2"].input_bits == 9000 * 8000
    assert cfg.task_types["T2"].result_bits == 10 * 8000
    assert cfg.radio.bandwidth_hz == 40e6
    assert cfg.radio.carrier_hz == 5.9e9
    assert cfg.radio.noise_w_per_hz == pytest.approx(10**-20.4, rel=1e-12)  # -174 dBm/Hz (CHANGES C1)
    assert cfg.radio.gain_tx == pytest.approx(10**0.5, rel=1e-12)  # 5 dBi
    assert cfg.cars.speed_low_mps == pytest.approx((10 / 3.6, 20 / 3.6, 25 / 3.6, 40 / 3.6))
    assert cfg.links.backhaul_bw == 1e9
    assert cfg.links.wan_delay == (0.025, 0.100)


def test_seed_ranges(cfg: Config) -> None:
    assert dict(cfg.fast_mode.seed_split) == {
        "nsga2_and_demos": range(1000, 2000),
        "training": range(2000, 3000),
        "evaluation": range(3000, 4000),
    }


def test_overlapping_seed_ranges_are_refused(raw: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="overlap") as err:
        edited(raw, assign("fast_mode.seed_split.training", "1500-2999"))
    assert err.value.key.startswith("fast_mode.seed_split")


def test_scenario_fog_classes(cfg: Config) -> None:
    assert dict(cfg.scenarios["S1"].fog) == {z: "F-GPU" for z in ("z1", "z2", "z3", "z4", "z5")}
    assert dict(cfg.scenarios["S6"].fog) == {
        "z1": "F-GPU",
        "z2": "F-CPU",
        "z3": "F-GPU",
        "z4": "F-CPU",
        "z5": "F-GPU",
    }
    assert cfg.scenarios["S5"].mobility == "high"
    assert cfg.scenarios["S4"].deadlines == "tight"


def test_hardware_classes(cfg: Config) -> None:
    fgpu, fcpu, cloud = cfg.hardware["F-GPU"], cfg.hardware["F-CPU"], cfg.hardware["cloud"]
    assert (fgpu.c_gpu, fgpu.c_cpu, dict(fgpu.queue)) == (2, 2, {"gpu": 32, "cpu": 16})
    assert fcpu.slot_types == ("cpu",)
    assert cloud.is_cloud and not fgpu.is_cloud
    assert cfg.energy.pue(cloud) == 1.1 and cfg.energy.pue(fcpu) == 1.5
    assert cfg.disabled_task_types == ("T5",)


# --- Immutability -------------------------------------------------------------------------------------------


def test_config_is_frozen(cfg: Config) -> None:
    # Typed Any on purpose: the point is that these writes fail at run time.
    tof: Any = cfg.tof
    task_types: Any = cfg.task_types
    e_ref: Any = cfg.derived.e_ref
    with pytest.raises(dataclasses.FrozenInstanceError):
        tof.alpha = 1.0
    with pytest.raises(TypeError):
        task_types["T9"] = cfg.task_types["T1"]
    with pytest.raises(TypeError):
        e_ref["T1"] = 0.0


# --- Errors name the key (PLAN M0.2) ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "radio",
        "layout.zone_radius_m",
        "layout.joining_zones",
        "hardware.F-GPU.c_gpu",
        "hardware.F-CPU.queue.cpu",
        "task_types.T1.deadline.tight",
        "task_types.T2.input_kb",
        "time_table_placeholder.factors.p95",
        "energy.e_down_nj_per_bit",
        "tof.strongest_class",
        "scenarios.S2.rho_fog",
        "dqfd.per.eps_agent",
        "fedavg.T_fl.testbed",
    ],
)
def test_a_missing_key_raises_an_error_naming_it(raw: dict[str, Any], key: str) -> None:
    with pytest.raises(ConfigError) as err:
        edited(raw, delete(key))
    assert err.value.key == key
    assert key in str(err.value)


@pytest.mark.parametrize(
    ("key", "typo"),
    [("hardware.F-GPU.c_gpu", "c_gpuu"), ("scheduler", "schedular"), ("rl.batch", "batch_size")],
)
def test_a_misspelled_key_names_the_expected_key_and_the_typo(
    raw: dict[str, Any], key: str, typo: str
) -> None:
    with pytest.raises(ConfigError) as err:
        edited(raw, rename(key, typo))
    assert err.value.key == key
    assert typo in str(err.value)


@pytest.mark.parametrize("key", ["hardware.F-GPU.c_tpu", "reward.w4", "extra_section"])
def test_an_unknown_key_is_refused(raw: dict[str, Any], key: str) -> None:
    with pytest.raises(ConfigError, match="unknown") as err:
        edited(raw, assign(key, 1))
    assert err.value.key == key


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("hardware.F-GPU.c_gpu", "two"),
        ("hardware.F-GPU.c_gpu", True),  # a bool is not a count
        ("hardware.F-GPU.c_gpu", -1),
        ("tof.alpha", "0.8"),
        ("tof.alpha", 0.0),
        ("links.backhaul_delay", [0.005, 0.001]),  # low > high
        ("links.wan_bw", 500e6),  # computed (18): only "calibrated" is allowed
        ("energy.e_bar", 11.7),
        ("task_types.T2.deadline.tight", "1.1xE"),
        ("task_types.T1.cls", "firm"),
        ("scenarios.S1.deadlines", "medium"),
        ("scenarios.S1.fog", ["cloud"]),
        ("tof.strongest_class", "cloud"),
        ("energy.reference_class", "F-TPU"),
        ("layout.joining_zones", ["z9"]),
        ("time_table_placeholder.T1.F-CPU", {"gpu": 0.008}),  # F-CPU has no GPU slot
        ("timing.quantiles", {"low": 0.95, "high": 0.05}),
        ("reward.w", 0.0),  # 11.4: w in (0, 1]
        ("reward.w", 1.2),
        ("reward.w", "from_knee"),  # a number now (QUESTIONS Q32 stand-in)
    ],
)
def test_a_bad_value_raises_an_error_naming_its_key(raw: dict[str, Any], key: str, value: object) -> None:
    with pytest.raises(ConfigError) as err:
        edited(raw, assign(key, value))
    assert err.value.key.startswith(key), err.value.key


def test_task_shares_must_sum_to_one(raw: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="sum") as err:
        edited(raw, assign("task_types.T1.share", 0.70))
    assert err.value.key == "task_types"


def test_every_task_type_needs_a_row_on_the_strongest_class(raw: dict[str, Any]) -> None:
    with pytest.raises(ConfigError) as err:
        edited(raw, delete("time_table_placeholder.T4.F-GPU"))
    assert err.value.key == "time_table_placeholder.T4.F-GPU"


def test_the_knee_weight_stand_in_is_read_from_the_config(cfg: Config) -> None:
    # QUESTIONS Q32: w* comes from NSGA-II in M5; until then the config's stand-in, 0.5.
    assert cfg.reward.w == 0.5
