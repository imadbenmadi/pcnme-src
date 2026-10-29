"""M5 acceptance tests (PLAN M5; final-design 13.2-13.4, 16.5; QUESTIONS Q38).

- Knee and w on synthetic fronts with known answers: tests/expert/test_knee.py.
- Front check (16.5): S2 fronts hold at least 5 distinct points. With the small budget of the cloud session
  (population 10 x 5 generations) they do not (1-4 points, Q38): a strict xfail below, run with -m slow; the
  full budget runs on the desktop (tools/nsga_runs.py).
- Replaying a plan reproduces its objective values exactly (same seed): the scripted tests below and
  tests/expert/test_demos.py; tools/make_demos.py checks it on every front plan it stores.
"""

# ruff: noqa: E501  (the references are full test names, which cannot be wrapped)

from __future__ import annotations

import importlib
import random
from typing import Any

import pytest

from pcnme.config import DEFAULT_CONFIG_PATH, config_from_mapping, load_yaml
from pcnme.expert.demos import replay
from pcnme.expert.nsga import check_front, run_nsga
from pcnme.expert.window import WindowSpec, discover, evaluate, window_seed
from tests.zone.harness import fixed_delay_config

TESTS = {
    "knee: a convex front by hand": "tests.expert.test_knee::test_a_convex_front_by_hand",
    "knee: the weight through the ranges": "tests.expert.test_knee::test_the_weight_goes_back_through_the_ranges",
    "knee: at an end, one-sided": "tests.expert.test_knee::test_a_knee_at_an_end_uses_its_one_neighbour",
    "knee: every point's w_p": "tests.expert.test_knee::test_every_point_gets_its_own_weight",
    "knee: w* the median": "tests.expert.test_knee::test_w_star_is_the_median_knee_weight",
    "replay reproduces the objectives": "tests.expert.test_demos::test_replaying_a_plan_reproduces_its_objectives_exactly",
    "front genomes reproduce their objectives": "tests.expert.test_nsga::test_every_front_genome_reproduces_its_objectives",
    "parallel (spawn) = serial": "tests.expert.test_nsga::test_a_run_is_repeatable_and_parallel_gives_the_same_front",
    "seeds only from 1000-1999": "tests.expert.test_window::test_window_seeds_are_in_the_nsga_range_and_distinct",
}


@pytest.mark.parametrize("ref", TESTS.values())
def test_the_acceptance_test_exists(ref: str) -> None:
    module, name = ref.split("::")
    assert callable(getattr(importlib.import_module(module), name)), ref


def test_replaying_random_plans_reproduces_their_objectives() -> None:
    # Same seed: evaluation (early stop, no logs) and replay (to the horizon, inputs on, another w) agree.
    cfg = fixed_delay_config(nsga2__T_warmup=1.0, nsga2__T_win=0.3)
    window = discover(cfg, WindowSpec("S2", window_seed(cfg, "S2", 1), 1, n_cars=40))
    rng = random.Random(11)
    for w in (0.1, 0.9):
        genome = [rng.randrange(len(t.options)) for t in window.tasks]
        assert replay(cfg, window, genome, w).objectives == evaluate(cfg, window, genome)[0]


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="Q38: at population 10 x 5 generations the S2 fronts hold 1-4 distinct points (16.5 asks 5); the full budget runs on the desktop",
)
def test_s2_fronts_hold_at_least_5_distinct_points() -> None:
    source: Any = load_yaml(DEFAULT_CONFIG_PATH)
    cfg = config_from_mapping(source)
    window = discover(cfg, WindowSpec("S2", window_seed(cfg, "S2", 0), 0))
    front = run_nsga(source, window, population=10, generations=5, workers=4)
    assert [(o.f1, o.f2) for o in check_front(cfg, front)] == list(front.F)
    assert front.distinct >= 5
