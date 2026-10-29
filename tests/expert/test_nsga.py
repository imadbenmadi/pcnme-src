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
from pcnme.expert.window import Window, WindowSpec, discover, evaluate, window_seed
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


# --- the seeded initial population (CHANGES C9) -------------------------------------------------------------


def test_energy_greedy_takes_the_cheapest_open_option(cfg: Config) -> None:
    from dataclasses import replace

    from pcnme.expert.window import energy_greedy
    from tests.agent.test_inputs import make_ctx

    ctx = make_ctx(cfg, "T4")  # here, empty: Ê_here = E_ref is the cheapest; the cloud costs more
    assert energy_greedy(ctx) == "here"
    views = list(ctx.views)
    views[0] = replace(views[0], masked="full")
    assert energy_greedy(replace(ctx, views=tuple(views))) == "cloud"  # the only other open option
    closed = tuple(replace(v, masked=v.masked or "full") for v in ctx.views)
    assert energy_greedy(replace(ctx, views=closed)) is None


def test_the_energy_greedy_plan_is_taken_by_its_window_tasks(cfg: Config, window: Window) -> None:
    from pcnme.expert.window import energy_greedy, plan_of

    g, planned = plan_of(cfg, window, energy_greedy)
    assert len(g) == len(window.tasks) and g != window.greedy_genome
    o, run = evaluate(cfg, window, g)
    assert o == planned  # the rule's own run: greedy before, after and at re-decisions, as in any plan
    # the evaluation is the rule's own run: every gene is taken, except where every option was masked
    for p in run.placements.values():
        assert all(p.first[tid] is None for tid in p.replaced)
    assert sum(len(p.applied) for p in run.placements.values()) > 0.9 * len(window.tasks)
    assert o == evaluate(cfg, window, g)[0]


def test_the_initial_population_holds_both_plans_their_mutants_and_random_plans(
    cfg: Config, window: Window
) -> None:
    from pcnme.expert.nsga import initial_population

    pr = WindowProblem(window, None)
    energy = tuple((g + 1) % len(t.options) for g, t in zip(window.greedy_genome, window.tasks, strict=True))
    X = initial_population(cfg, window, pr, 10, energy)
    rows = [tuple(r) for r in X]
    assert len(rows) == len(set(rows)) == 10
    assert rows[0] == window.greedy_genome and rows[1] == energy
    assert ((pr.xl <= X) & (pr.xu >= X)).all()
    near = lambda r, b: sum(x != y for x, y in zip(r, b, strict=True))  # noqa: E731
    # rows 2-4 are mutated copies (alternating greedy, energy): a few genes away from their base
    assert all(0 < near(rows[i], (window.greedy_genome, energy)[i % 2]) <= 5 for i in (2, 3, 4))
    # rows 5-9 are random: far from both
    assert all(near(rows[i], window.greedy_genome) > len(rows[0]) // 3 for i in range(5, 10))
    assert (initial_population(cfg, window, pr, 10, energy) == X).all()  # seeded
    assert (X[5:] == pr.xu).any() and (X[5:] == 0).any()  # random rows use every option
    same = initial_population(cfg, window, pr, 6, window.greedy_genome)  # both plans equal: kept once
    assert len({tuple(r) for r in same}) == 6 and tuple(same[1]) != window.greedy_genome


def test_no_front_point_is_dominated_by_the_greedy_plan(source: dict[str, Any], window: Window) -> None:
    # C9 + elitism: the greedy plan is in the first population.
    g = (window.greedy.f1, window.greedy.f2)
    f = run_nsga(source, window, population=4, generations=2)
    assert not any(g[0] <= p[0] and g[1] <= p[1] and g != p for p in f.F)
