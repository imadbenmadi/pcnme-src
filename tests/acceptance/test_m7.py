"""M7 acceptance tests (PLAN M7; final-design 16.1, 16.3, 16.4).

- Every method runs S1 for a short episode on 2 evaluation seeds and keeps the M3 and M4 invariants: every
  task closes once or is given up by its car; every decision has 48 inputs in their ranges, its action was
  open (masked actions are never chosen), and every decision became one transition, none discarded.
- Equal training budgets: every learning method uses exactly the same budget in decisions (16.1).
- `pcnme report` reproduces a hand-checked interval on synthetic data, and the runner refuses seeds outside
  their ranges: the tests listed in TESTS (checked to exist).
"""

# ruff: noqa: E501  (the references are full test names, which cannot be wrapped)

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from pcnme.agent.inputs import INPUT_NAMES, N_INPUTS
from pcnme.config import DEFAULT_CONFIG_PATH, config_from_mapping, load_yaml
from pcnme.experiments.city import City
from pcnme.experiments.runner import RunSpec, method_config, placement, train
from pcnme.methods.learners import LEARNING, METHODS
from tests.agent.test_inputs import check_ranges

TESTS = {
    "report: a hand-checked CI on synthetic data": "tests.metrics.test_report::test_the_report_reproduces_a_hand_checked_interval",
    "stats: the hand-checked interval": "tests.metrics.test_stats::test_a_hand_checked_interval",
    "stats: t quantiles (16.3: 2.78, 2.09)": "tests.metrics.test_stats::test_t_quantiles_match_the_tables",
    "runner: evaluation seeds only": "tests.experiments.test_runner::test_the_runner_refuses_seeds_outside_the_evaluation_range",
    "runner: training seeds only": "tests.experiments.test_runner::test_training_seeds_are_checked_too",
    "knee reference on its window": "tests.experiments.test_runner::test_the_knee_reference_replays_its_plan_on_the_evaluation_window",
    "every named ablation on a tiny run": "tests.experiments.test_ablations::test_every_named_ablation_runs_in_its_own_directory",
    "the agent's ablations on a tiny run": "tests.experiments.test_ablations::test_the_expert_ablations_on_a_tiny_run",
    "sweeps on a tiny run": "tests.experiments.test_ablations::test_the_state_period_sweep_and_the_broker_load",
}
SEEDS = (3000, 3001)
BUDGET = 120  # decisions: the same for every learning method
SHORT = {"seconds": 1.0, "n_cars": 20}


@pytest.fixture(scope="module")
def source() -> dict[str, Any]:
    src = load_yaml(DEFAULT_CONFIG_PATH)
    # small enough for a test: short NSGA-II windows (the knee reference) and two greedy windows (D3QN +
    # greedy expert's demonstrations; the held-out split needs 2)
    return {**src, "nsga2": {**src["nsga2"], "T_warmup": 0.5, "T_win": 0.3}}


@pytest.mark.parametrize("ref", TESTS.values())
def test_the_acceptance_test_exists(ref: str) -> None:
    module, name = ref.split("::")
    assert callable(getattr(importlib.import_module(module), name)), ref


def spec_of(method: str, demos: tuple[Path, ...] = ()) -> RunSpec:
    return RunSpec(
        "S1", method, SEEDS, budget=BUDGET, train_seconds=1.0, windows=2, pretrain_updates=20, demos=demos,
        nsga_population=4, nsga_generations=1, **SHORT,  # type: ignore[arg-type]
    )  # fmt: skip


@pytest.fixture(scope="module")
def knee_demos(source: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, ...]:
    """PCNME's teacher for the test: 2 short windows, each front = the greedy and the energy-greedy plans."""
    from pcnme.expert.demos import demonstrate_front, write
    from pcnme.expert.window import WindowSpec, discover, energy_greedy, plan_of, window_seed

    cfg = config_from_mapping(source)
    out = []
    for k in range(2):
        w = discover(cfg, WindowSpec("S1", window_seed(cfg, "S1", k), k, SHORT["n_cars"]))  # type: ignore[arg-type]
        energy, eo = plan_of(cfg, w, energy_greedy)
        front = demonstrate_front(
            cfg, w, [w.greedy_genome, energy], [(w.greedy.f1, w.greedy.f2), (eo.f1, eo.f2)]
        )
        if not front:  # the two plans coincide: one point, no weight; replay greedy as the knee
            from dataclasses import replace

            from pcnme.expert.demos import replay

            front = [replace(replay(cfg, w, w.greedy_genome, cfg.reward.w), is_knee=True)]
        out.append(write(tmp_path_factory.mktemp("demos") / f"S1-w{k}.parquet", front))
    return tuple(out)


@pytest.mark.parametrize("method", METHODS)
def test_every_method_runs_s1_and_keeps_the_m3_and_m4_invariants(
    source: dict[str, Any], knee_demos: tuple[Path, ...], method: str
) -> None:
    if method in LEARNING:
        pytest.importorskip("torch")
    if method == "knee-reference":
        pytest.importorskip("pymoo")
    spec = spec_of(method, knee_demos if method == "pcnme" else ())
    cfg = method_config(config_from_mapping(source), method)
    trained = train(cfg, spec)
    if method in LEARNING:
        assert trained is not None and trained.budget.used == trained.budget.total == BUDGET  # 16.1
    decisions = 0
    for seed in SEEDS:
        how, _ = placement(source, cfg, spec, trained, seed)
        c = City(cfg, "S1", how, seed, SHORT["n_cars"], usage="evaluation", method_name=method).run(1.0)
        closed = [r.task.task_id for r in c.records]
        given_up = {d.task.task_id for d in c.car_drops}
        assert len(closed) == len(set(closed)) and set(c.created) == set(closed) | given_up  # M3
        for zone in c.zones.values():
            for d in zone.decisions:
                assert len(d.inputs) == N_INPUTS
                check_ranges(dict(zip(INPUT_NAMES, d.inputs, strict=True)))  # M4
                assert d.masked[d.action] is None  # a masked action is never chosen
                decisions += 1
            assert len(zone.transitions.replay) == len(zone.decisions) and zone.transitions.discarded == 0
    assert decisions > 50
