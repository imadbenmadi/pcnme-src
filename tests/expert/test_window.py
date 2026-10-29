"""NSGA-II windows and their evaluation (PLAN M5.1; final-design 13.2, 13.4.3, 16.3; QUESTIONS Q37).

A short window keeps the tests fast: T_warmup 1 s, T_win 0.3 s, S2 with 40 cars (the per-zone load is the
scenario's: λ_car scales with the fleet, 18).
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from pcnme.agent.reward import AttemptReward
from pcnme.car.car import Confirm, Task
from pcnme.config import Config
from pcnme.expert.window import (
    Window,
    WindowSpec,
    discover,
    evaluate,
    horizon,
    objectives,
    simulate,
    window_seed,
)
from pcnme.ids import TaskId
from pcnme.sim.rng import SeedError
from pcnme.zone.zone import TaskRecord
from tests.zone.harness import fixed_delay_config


@pytest.fixture(scope="module")
def cfg() -> Config:
    return fixed_delay_config(nsga2__T_warmup=1.0, nsga2__T_win=0.3)


@pytest.fixture(scope="module")
def spec(cfg: Config) -> WindowSpec:
    return WindowSpec("S2", window_seed(cfg, "S2", 0), 0, n_cars=40)


@pytest.fixture(scope="module")
def window(cfg: Config, spec: WindowSpec) -> Window:
    return discover(cfg, spec)


def random_genome(window: Window, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(len(t.options)) for t in window.tasks]


# --- seeds (16.3) -------------------------------------------------------------------------------------------


def test_window_seeds_are_in_the_nsga_range_and_distinct(cfg: Config) -> None:
    seeds = {window_seed(cfg, s, k) for s in cfg.scenarios for k in range(20)}
    assert len(seeds) == 6 * 20 and min(seeds) == 1000 and max(seeds) == 1519
    assert window_seed(cfg, "S2", 3) == 1103
    with pytest.raises(ValueError):
        window_seed(cfg, "S2", 100)


def test_a_training_seed_is_refused_for_a_window(cfg: Config) -> None:
    with pytest.raises(SeedError):
        simulate(cfg, WindowSpec("S2", 2000, 0, n_cars=40), {}, stop=lambda r: True)


# --- the window (13.2.1, 13.2.2) ----------------------------------------------------------------------------


def test_the_window_holds_the_pebbles_whose_upload_was_in_during_t_win(cfg: Config, window: Window) -> None:
    assert len(window.tasks) > 50
    assert all(1.0 <= t.t_rx < 1.3 for t in window.tasks)
    assert [t.t_rx for t in window.tasks] == sorted(t.t_rx for t in window.tasks)
    assert {t.task_type for t in window.tasks} <= {"T1", "T3", "T4", "T2"}  # loose: T2 is a pebble (S2)


def test_the_genes_are_the_anchors_non_empty_options(window: Window) -> None:
    # 13.2.2: 4 active zones, 3 neighbours each: position n4 is empty, so a gene never points at it.
    assert {t.options for t in window.tasks} == {("here", "n1", "n2", "n3", "cloud")}
    with pytest.raises(ValueError):
        window.genes([5] * len(window.tasks))
    with pytest.raises(ValueError):
        window.genes([0])


def test_the_window_tasks_do_not_depend_on_the_plan(cfg: Config, window: Window) -> None:
    _, run = evaluate(cfg, window, random_genome(window, 1))
    from pcnme.expert.window import window_tasks

    assert tuple(window_tasks(cfg, run)) == window.tasks


# --- evaluation (13.2.3) ------------------------------------------------------------------------------------


def test_the_greedy_plan_reproduces_the_greedy_run(cfg: Config, window: Window) -> None:
    obj, _ = evaluate(cfg, window, window.greedy_genome)
    assert obj == window.greedy


def test_an_evaluation_is_exactly_repeatable(cfg: Config, window: Window) -> None:
    g = random_genome(window, 2)
    assert evaluate(cfg, window, g)[0] == evaluate(cfg, window, g)[0]


def test_the_early_stop_and_the_logs_do_not_change_the_objectives(cfg: Config, window: Window) -> None:
    g = random_genome(window, 3)
    stopped, _ = evaluate(cfg, window, g)
    full = simulate(cfg, window.spec, window.genes(g), logging=True)  # to the horizon, every log on
    assert full.city.clock.now == pytest.approx(horizon(cfg))
    assert objectives(cfg, full.records(t.key for t in window.tasks)) == stopped


def test_the_warm_up_is_greedy_in_every_plan(cfg: Config, window: Window) -> None:
    # 13.2.3: T_warmup of greedy decisions: every decision before the window is the same in every plan.
    def before(genome: list[int]) -> list[tuple[str, object, int]]:
        _, run = evaluate(cfg, window, genome)
        return [
            (z, d.task_id, d.action)
            for z, zone in run.city.zones.items()
            for d in zone.decisions
            if d.time < 1.0
        ]

    a = before(random_genome(window, 4))
    assert a and a == before(window.greedy_genome)


def test_open_genes_are_applied_masked_ones_go_to_greedy_and_re_decisions_are_greedys(
    cfg: Config, window: Window
) -> None:
    # 13.4.3, Q37. All-cloud: a hard T1 cannot make the cloud round trip (about 137 ms > 100 ms): masked.
    cloud = [t.options.index("cloud") for t in window.tasks]
    _, run = evaluate(cfg, window, cloud)
    applied = {(z, tid) for z, p in run.placements.items() for tid in p.applied}
    replaced = {(z, tid) for z, p in run.placements.items() for tid in p.replaced}
    keys = {t.key for t in window.tasks}
    assert applied | replaced == keys and not applied & replaced and replaced
    kinds = {t.key: t.task_type for t in window.tasks}
    assert all(kinds[k] in ("T1", "T3") for k in replaced)  # hard: cannot meet the deadline in the cloud
    assert any(kinds[k] in ("T4", "T2") for k in applied)
    for z, zone in run.city.zones.items():
        for d in zone.decisions:
            if (z, d.task_id) in applied and d.attempt == 1:
                assert d.action == 5


def test_a_plan_changes_the_objectives(cfg: Config, window: Window) -> None:
    here = [0] * len(window.tasks)
    assert evaluate(cfg, window, here)[0] != window.greedy


# --- the objectives (13.2.4) by hand ------------------------------------------------------------------------


def rec(cfg: Config, kind: str, outcome: str, latency: float | None, energy: float) -> TaskRecord:
    t = cfg.task_types[kind]
    task = Task(TaskId("c1", 0, 1), "c1", kind, t.cls, t.deadline["loose"], t.l_max["loose"], 0.0, 0.0, 0.0)
    r = TaskRecord(task=task, anchor="z1", outcome=outcome)
    if latency is not None:
        r.confirm = Confirm(task.task_id, "z1", 1, latency)
    r.rewards = {1: AttemptReward(1, "delivered", 0.0, energy * 0.4, None, True)}
    r.rewards[2] = replace(r.rewards[1], attempt=2, energy=energy * 0.6)  # E is over every attempt
    return r


def test_the_objectives_by_hand(cfg: Config) -> None:
    e_ref, e_bar = cfg.derived.e_ref, cfg.derived.e_bar
    rs = [
        rec(cfg, "T1", "on_time", 0.05, 1.5 * e_ref["T1"]),  # c = 0.5, ê = 1.5
        rec(cfg, "T4", "late", 1.5, 100 * e_ref["T4"]),  # c = 1.5 + 0.5 = 2, ê capped at ē
        rec(cfg, "T1", "drop_lost", None, 0.0),  # C_drop = 5, ē
    ]
    o = objectives(cfg, rs)
    assert (o.n, o.drops) == (3, 1)
    assert o.f1 == pytest.approx((0.5 + 2.0 + 5.0) / 3)
    assert o.f2 == pytest.approx((1.5 + e_bar + e_bar) / 3)
    assert objectives(cfg, []).n == 0


def test_the_horizon_by_hand(cfg: Config) -> None:
    # T_warmup 1 + T_win 0.3 + the longest D + L_max (T2 loose: 3 + 3 s) + T_ack 0.2 + G 0.5 = 8.0 s
    assert horizon(cfg) == pytest.approx(8.0)


def test_boulders_are_not_window_tasks(cfg: Config) -> None:
    # 13.2.1: pebbles only. In S4 (tight) T2 is a boulder (18): it goes to the cloud by rule, with no gene.
    w = discover(cfg, WindowSpec("S4", window_seed(cfg, "S4", 0), 0, n_cars=40))
    assert w.tasks and all(t.task_type != "T2" for t in w.tasks)


def test_a_re_decision_never_takes_the_gene(cfg: Config) -> None:
    # Q37: only a window task's first decision takes its gene; a re-decision of it is greedy's.
    from pcnme.expert.window import PlanPlacement
    from tests.agent.test_inputs import make_ctx

    ctx = make_ctx(cfg, "T4")  # an empty F-GPU here: greedy picks here
    p = PlanPlacement(cfg, "z1", {ctx.task.task_id: "cloud"})
    assert p.choose(ctx) == "cloud" and p.applied == {ctx.task.task_id}
    assert p.choose(ctx) == "here"
    assert p.first == {ctx.task.task_id: "cloud"}


def test_a_rule_makes_only_the_first_decision_of_its_tasks(cfg: Config) -> None:
    # C9: the energy-greedy plan's rule decides a window task's first decision; re-decisions are greedy's.
    from pcnme.expert.window import PlanPlacement
    from tests.agent.test_inputs import make_ctx

    ctx = make_ctx(cfg, "T4")
    p = PlanPlacement(cfg, "z1", {}, rule=lambda c: "cloud", ruled=frozenset({ctx.task.task_id}))
    assert p.choose(ctx) == "cloud" and p.choose(ctx) == "here"
    other = PlanPlacement(cfg, "z1", {}, rule=lambda c: "cloud", ruled=frozenset())
    assert other.choose(ctx) == "here"  # not a ruled task: greedy
