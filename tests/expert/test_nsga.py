"""NSGA-II over one window (PLAN M5.2; final-design 13.2.5-13.2.6, 18).

The run tests use the short window of test_window.py and a tiny budget (population 4, 2 generations).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pcnme.config import Config, config_from_mapping, load_yaml
from pcnme.expert.nsga import (
    Front,
    ValidOptionMutation,
    WindowProblem,
    check_front,
    make_algorithm,
    nsga_seed,
    run_nsga,
    sort_front,
)
from pcnme.expert.window import Window, WindowSpec, discover, window_seed
from tests.zone.harness import CONFIG


@pytest.fixture(scope="module")
def source() -> dict[str, Any]:
    data: dict[str, Any] = copy.deepcopy(load_yaml(CONFIG))
    data["nsga2"]["T_warmup"] = 1.0
    data["nsga2"]["T_win"] = 0.3
    return data


@pytest.fixture(scope="module")
def cfg(source: dict[str, Any]) -> Config:
    return config_from_mapping(source)


@pytest.fixture(scope="module")
def window(cfg: Config) -> Window:
    return discover(cfg, WindowSpec("S2", window_seed(cfg, "S2", 0), 0, n_cars=40))


@pytest.fixture(scope="module")
def front(source: dict[str, Any], window: Window) -> Front:
    return run_nsga(source, window, population=4, generations=2)


# --- the mutation (13.2.6) --------------------------------------------------------------------------------


def problem(widths: list[int]) -> SimpleNamespace:
    xu = np.array(widths, dtype=float)
    return SimpleNamespace(n_var=len(widths), xl=np.zeros(len(widths)), xu=xu)


def test_the_mutation_moves_a_gene_to_another_valid_option() -> None:
    p = problem([4] * 50)  # 5 options per gene
    X = np.random.default_rng(0).integers(0, 5, size=(2000, 50))
    Y = ValidOptionMutation()._do(p, X, random_state=np.random.default_rng(1))
    changed = Y != X
    assert ((Y >= 0) & (Y <= 4)).all()
    # 1 / n per gene: 2000 x 50 genes x 1/50 = 2000 expected moves
    assert 1800 < changed.sum() < 2200
    moved_to = Y[changed] - X[changed]
    assert set(np.unique(moved_to % 5)) == {1, 2, 3, 4}  # every other option, never the same


def test_a_gene_with_one_option_never_moves() -> None:
    p = problem([0, 3])
    X = np.zeros((500, 2), dtype=int)
    Y = ValidOptionMutation()._do(p, X, random_state=np.random.default_rng(2))
    assert (Y[:, 0] == 0).all() and (Y[:, 1] != 0).any()


# --- the problem and the seed -------------------------------------------------------------------------------


def test_one_integer_gene_per_window_task_bounded_by_its_options(window: Window) -> None:
    pr = WindowProblem(window, None)
    assert pr.n_var == len(window.tasks) and pr.n_obj == 2
    assert list(pr.xu) == [len(t.options) - 1 for t in window.tasks] and not pr.xl.any()


def test_the_nsga_seed_comes_from_the_windows_seed(cfg: Config, window: Window) -> None:
    other = Window(WindowSpec("S2", window.spec.seed + 1, 0, 40), window.tasks, (), window.greedy)
    assert nsga_seed(cfg, window) == nsga_seed(cfg, window) != nsga_seed(cfg, other)


# --- a run (13.2.5-13.2.6, 18) ------------------------------------------------------------------------------


def test_the_front_is_non_dominated_and_sorted(front: Front) -> None:
    F = front.F
    assert tuple(sorted(F)) == F and front.distinct == len(set(F))
    for a in F:
        assert not any(b[0] <= a[0] and b[1] <= a[1] and b != a for b in F)
    assert front.evaluations <= 4 * 2 and len(front.genomes) == len(F)


def test_every_front_genome_reproduces_its_objectives(cfg: Config, front: Front) -> None:
    # 13.2.3, common random numbers: re-running a plan gives its objectives exactly.
    assert [(o.f1, o.f2) for o in check_front(cfg, front)] == list(front.F)


def test_a_run_is_repeatable_and_parallel_gives_the_same_front(
    source: dict[str, Any], window: Window, front: Front
) -> None:
    # The pool is spawn-started (Windows): results in the population's order, the same front.
    parallel = run_nsga(source, window, population=4, generations=2, workers=2)
    assert (parallel.F, parallel.genomes) == (front.F, front.genomes)


def test_the_budget_defaults_to_the_config(source: dict[str, Any], window: Window) -> None:
    small = copy.deepcopy(source)
    small["nsga2"]["population"] = 3
    small["nsga2"]["generations"] = 1
    f = run_nsga(small, window)
    assert f.evaluations == 3


def test_the_operators_follow_the_config(cfg: Config) -> None:
    a = make_algorithm(cfg, 7)
    assert a.pop_size == 7 and a.eliminate_duplicates
    assert float(a.mating.crossover.prob.value) == cfg.nsga2.crossover_prob == 0.9
    assert isinstance(a.mating.mutation, ValidOptionMutation)


def test_the_front_is_sorted_by_f1_then_f2() -> None:
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    F = np.array([[0.5, 1.0], [0.2, 2.0], [0.9, 0.1], [0.2, 1.5]])
    genomes, points = sort_front(X, F)
    assert points == ((0.2, 1.5), (0.2, 2.0), (0.5, 1.0), (0.9, 0.1))
    assert genomes == ((4, 4), (2, 2), (1, 1), (3, 3))
