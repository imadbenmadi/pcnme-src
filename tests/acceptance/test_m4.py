"""M4 acceptance tests (PLAN M4; final-design 11.1, 11.3, 11.4; N18).

- The input vector: 48 long, in the order of 11.1, every value in its range: on hand-built contexts (a
  Hypothesis test) and on every decision of real Greedy runs.
- N18: every drop reward is lower than every delivery reward, for random w in (0, 1] and random energies.
- Transition timing on a scripted two-zone scenario computed by hand.
- The input-range report (Greedy, S2, 60 s, 3 seeds): tools/input_ranges.py; its slow test is below.
"""

# ruff: noqa: E501  (the references are full test names, which cannot be wrapped)

from __future__ import annotations

import importlib
import math
from pathlib import Path

import pytest

from pcnme.agent.inputs import INPUT_NAMES, N_INPUTS
from pcnme.config import Config, load_config
from pcnme.experiments.city import City
from tests.agent.test_inputs import check_ranges

TESTS = {
    "order and length (11.1)": "tests.agent.test_inputs::test_48_inputs_in_the_order_of_11_1",
    "every value in its range (Hypothesis)": "tests.agent.test_inputs::test_every_input_is_in_its_range",
    "N18: every drop below every delivery (Hypothesis)": "tests.agent.test_reward::test_every_drop_is_worse_than_every_delivery",
    "transition timing by hand (two zones)": "tests.agent.test_transitions::test_transition_timing_two_tasks_at_one_anchor_by_hand",
    "a re-decision is the next step (two zones)": "tests.agent.test_transitions::test_a_re_decision_is_the_next_step_of_the_same_anchor",
    "s' never from another anchor (two zones)": "tests.agent.test_transitions::test_s_prime_never_comes_from_another_anchor",
}


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


@pytest.mark.parametrize("ref", TESTS.values())
def test_the_acceptance_test_exists(ref: str) -> None:
    module, name = ref.split("::")
    assert callable(getattr(importlib.import_module(module), name)), ref


def test_n18_and_the_scripted_timing_pass(cfg: Config) -> None:
    from tests.agent.test_reward import test_every_drop_is_worse_than_every_delivery
    from tests.agent.test_transitions import test_transition_timing_two_tasks_at_one_anchor_by_hand

    test_every_drop_is_worse_than_every_delivery(cfg)  # type: ignore[call-arg]
    test_transition_timing_two_tasks_at_one_anchor_by_hand()


def ranges_of_a_run(cfg: Config, scenario: str, seed: int, seconds: float) -> int:
    city = City(cfg, scenario, "greedy", seed).run(seconds)
    n = 0
    for zone in city.zones.values():
        for d in zone.decisions:
            assert len(d.inputs) == N_INPUTS
            check_ranges(dict(zip(INPUT_NAMES, d.inputs, strict=True)))
            assert d.masked[d.action] is None  # the action taken was open
            n += 1
        assert len(zone.transitions.replay) == len(zone.decisions) and zone.transitions.discarded == 0
    return n


def test_every_decision_of_a_short_city_run_is_in_range(cfg: Config) -> None:
    assert ranges_of_a_run(cfg, "S2", 2000, 5.0) > 1000


@pytest.mark.slow
@pytest.mark.parametrize("seed", [2000, 2001, 2002])
def test_every_decision_of_greedy_s2_60_s_is_in_range(cfg: Config, seed: int) -> None:
    assert ranges_of_a_run(cfg, "S2", seed, 60.0) > 15_000


@pytest.mark.slow
@pytest.mark.parametrize("scenario", ["S4", "S6"])
def test_tight_deadlines_and_cpu_zones_stay_in_range(cfg: Config, scenario: str) -> None:
    # S4: tight D (T2 a boulder); S6: F-CPU zones (T2 cannot run there: "cannot run").
    assert ranges_of_a_run(cfg, scenario, 2000, 20.0) > 0
    assert math.isfinite(cfg.derived.e_bar)


def _tool():  # type: ignore[no-untyped-def]
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[2] / "tools" / "input_ranges.py"
    spec = importlib.util.spec_from_file_location("input_ranges", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["input_ranges"] = mod  # dataclasses look their module up while the class is built
    spec.loader.exec_module(mod)
    return mod


def test_the_range_report_knows_each_inputs_cap_and_flags_constant_ones() -> None:
    tool = _tool()
    caps = {n: tool.cap(n) for n in INPUT_NAMES}
    assert caps["slack"] == caps["here_wait"] == caps["n1_age"] == caps["cloud_network"] == 2.0
    assert caps["exit"] == caps["w"] == caps["cloud_energy"] == caps["n2_present"] == caps["hard"] == 1.0
    assert caps["next_stays"] == caps["input_size"] == caps["D_norm"] == 1.0
    zeros, ones = (0.0,) * N_INPUTS, tuple(caps[n] for n in INPUT_NAMES)
    rows = {r.name: r for r in tool.table([zeros, ones])}
    assert rows["slack"].mean == 1.0 and rows["slack"].zero == 0.5 and rows["slack"].at_cap == 0.5
    assert rows["slack"].flag == ""
    assert {r.flag for r in tool.table([zeros])} == {"always 0"}
    assert {r.flag for r in tool.table([ones])} == {"always at cap"}
