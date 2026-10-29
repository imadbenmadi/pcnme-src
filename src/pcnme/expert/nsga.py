"""NSGA-II over one window, in pymoo (final-design 13.2.5-13.2.6, 18; PLAN M5.2). THIS TOUCHES THE AGENT: its
fronts become the expert's demonstrations (13.4).

Plain NSGA-II (pymoo's NSGA2): binary tournament selection on rank and crowding, uniform crossover on the
genes with probability 0.9, and a mutation that moves a task to another of its valid options, with
probability 1 / the number of genes per gene (no arithmetic on node ids, 13.2.6). The initial population is
random over the valid options; duplicates are removed (pymoo's default). No constraint: drops are in the
objectives (13.2.5).

pymoo draws from its own generator, seeded from the "nsga" stream of the window's seed (QUESTIONS Q4), so a
window gives the same front on every run. The evaluations of a generation run in parallel on a spawn-started
process pool (Windows has no fork); each worker builds the config from the raw mapping once (a Config is not
picklable) and keeps the window. Results come back in the population's order, so the front does not depend on
the number of workers.
"""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

from pcnme.config import Config, config_from_mapping
from pcnme.expert.window import USAGE, Objectives, Window, evaluate
from pcnme.sim.rng import RandomStreams

ConfigSource = Mapping[str, Any]  # the raw config/default.yaml mapping (picklable)


class ValidOptionMutation(Mutation):
    """13.2.6: each gene, with probability 1 / the number of genes, moves to another of its task's valid
    options, uniformly."""

    def __init__(self) -> None:
        super().__init__(prob=1.0)

    def _do(self, problem: Problem, X: np.ndarray, *args: Any, random_state: Any = None, **kw: Any) -> Any:
        X = np.asarray(X, dtype=int).copy()
        n_var = problem.n_var
        hit = random_state.random(X.shape) < 1.0 / n_var
        width = (problem.xu - problem.xl).astype(int)  # the other options of each gene
        for i, j in zip(*np.nonzero(hit & (width > 0)[None, :]), strict=True):
            step = 1 + int(random_state.integers(width[j]))  # 1 .. width: never the current option
            X[i, j] = problem.xl[j] + (X[i, j] - problem.xl[j] + step) % (width[j] + 1)
        return X


_WORKER: dict[str, Any] = {}  # per worker process: the config and the window (set by its initializer)


def _init_worker(source: ConfigSource, window: Window) -> None:
    _WORKER["cfg"] = config_from_mapping(source)
    _WORKER["window"] = window


def _evaluate_one(genome: Sequence[int]) -> tuple[float, float]:
    o, _ = evaluate(_WORKER["cfg"], _WORKER["window"], [int(g) for g in genome])
    return o.f1, o.f2


class WindowProblem(Problem):
    """One gene per window task, an integer in [0, number of its options - 1]; two objectives (13.2.4)."""

    def __init__(self, window: Window, pool: Any | None) -> None:
        sizes = np.array([len(t.options) for t in window.tasks])
        super().__init__(n_var=len(window.tasks), n_obj=2, xl=np.zeros(len(sizes)), xu=sizes - 1, vtype=int)
        self.pool = pool

    def _evaluate(self, X: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        genomes = [tuple(int(g) for g in row) for row in np.asarray(X)]
        if self.pool is None:
            results = [_evaluate_one(g) for g in genomes]
        else:
            results = self.pool.map(_evaluate_one, genomes)
        out["F"] = np.array(results, dtype=float)


@dataclass(frozen=True, slots=True)
class Front:
    """The final non-dominated set of one window: genomes, their (f1, f2), and what it cost."""

    window: Window
    genomes: tuple[tuple[int, ...], ...]
    F: tuple[tuple[float, float], ...]
    evaluations: int
    seconds: float

    @property
    def distinct(self) -> int:
        """Distinct points of the front (16.5: at least 5 in S2)."""
        return len(set(self.F))


def nsga_seed(cfg: Config, window: Window) -> int:
    """pymoo's seed: the first draw of the "nsga" stream of the window's seed, keyed by scenario and
    window."""
    s = window.spec
    g = RandomStreams.from_config(cfg, s.seed, USAGE).generator("nsga", s.scenario, s.index)
    return int(g.integers(2**31 - 1))


def run_nsga(
    source: ConfigSource,
    window: Window,
    *,
    population: int | None = None,
    generations: int | None = None,
    workers: int = 1,
) -> Front:
    """NSGA-II on one window. Population and generations default to the config (18: 100 x 100)."""
    cfg = config_from_mapping(source)
    pop = population if population is not None else cfg.nsga2.population
    gens = generations if generations is not None else cfg.nsga2.generations
    algorithm = make_algorithm(cfg, pop)
    start = time.perf_counter()
    if workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(source, window)) as pool:
            res = minimize(
                WindowProblem(window, pool), algorithm, ("n_gen", gens), seed=nsga_seed(cfg, window)
            )
    else:
        _init_worker(source, window)
        res = minimize(WindowProblem(window, None), algorithm, ("n_gen", gens), seed=nsga_seed(cfg, window))
    seconds = time.perf_counter() - start
    genomes, points = sort_front(res.X, res.F)
    return Front(window, genomes, points, int(res.algorithm.evaluator.n_eval), seconds)


def make_algorithm(cfg: Config, population: int) -> NSGA2:
    """13.2.6, 18: tournament selection (NSGA2's default), uniform crossover with the config's probability,
    the valid-option mutation, random valid initial plans, no duplicates."""
    return NSGA2(
        pop_size=population,
        sampling=IntegerRandomSampling(),
        crossover=UniformCrossover(prob=cfg.nsga2.crossover_prob),
        mutation=ValidOptionMutation(),
        eliminate_duplicates=True,
    )


def sort_front(
    X: np.ndarray, F: np.ndarray
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[float, float], ...]]:
    """The front's genomes and points by f1, then f2."""
    X, F = np.atleast_2d(X).astype(int), np.atleast_2d(F)
    order = np.lexsort((F[:, 1], F[:, 0]))
    return (
        tuple(tuple(int(g) for g in X[i]) for i in order),
        tuple((float(F[i, 0]), float(F[i, 1])) for i in order),
    )


def check_front(cfg: Config, front: Front) -> list[Objectives]:
    """Re-evaluate every front genome: it must give its objectives exactly (common random numbers)."""
    return [evaluate(cfg, front.window, g)[0] for g in front.genomes]
