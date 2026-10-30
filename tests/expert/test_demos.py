"""Demonstrations (PLAN M5.4, M5 acceptance; final-design 13.4, 11.3, 11.4; QUESTIONS Q39, Q40).

The short window of test_window.py (T_warmup 1 s, T_win 0.3 s, S2, 40 cars).
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from pcnme.agent.inputs import INPUT_NAMES
from pcnme.config import Config
from pcnme.expert.demos import Demonstration, demonstrate_front, replay, rows, with_w, write
from pcnme.expert.window import Window, WindowSpec, discover, evaluate, window_seed
from pcnme.zone.decision import OPTIONS
from tests.zone.harness import fixed_delay_config

pq = pytest.importorskip("pyarrow.parquet")


@pytest.fixture(scope="module")
def cfg() -> Config:
    return fixed_delay_config(nsga2__T_warmup=1.0, nsga2__T_win=0.3)


@pytest.fixture(scope="module")
def window(cfg: Config) -> Window:
    return discover(cfg, WindowSpec("S2", window_seed(cfg, "S2", 0), 0, n_cars=40))


@pytest.fixture(scope="module")
def genome(window: Window) -> list[int]:
    rng = random.Random(7)
    return [rng.randrange(len(t.options)) for t in window.tasks]


@pytest.fixture(scope="module")
def demo(cfg: Config, window: Window, genome: list[int]) -> Demonstration:
    return replay(cfg, window, genome, 0.3)


def test_replaying_a_plan_reproduces_its_objectives_exactly(
    cfg: Config, window: Window, genome: list[int], demo: Demonstration
) -> None:
    # PLAN M5 acceptance: the same seed, the same objective values, whatever w the replay carries.
    assert demo.objectives == evaluate(cfg, window, genome)[0]
    assert replay(cfg, window, window.greedy_genome, 0.8).objectives == window.greedy


def test_the_inputs_carry_the_plans_w(demo: Demonstration) -> None:
    i = INPUT_NAMES.index("w")
    assert demo.transitions and all(d.transition.s[i] == 0.3 for d in demo.transitions)


def test_the_rewards_use_the_plans_w(cfg: Config, window: Window, genome: list[int]) -> None:
    # Q40: a delivered attempt's reward is B - w L/D - (1 - w) ê: the same transition under two w differs.
    a = {
        (t.transition.task_id, t.transition.attempt): t.transition.r
        for t in replay(cfg, window, genome, 0.2).transitions
    }
    b = {
        (t.transition.task_id, t.transition.attempt): t.transition.r
        for t in replay(cfg, window, genome, 0.9).transitions
    }
    assert a.keys() == b.keys() and any(a[k] != b[k] for k in a)


def test_only_window_tasks_and_every_one_of_their_decisions(
    cfg: Config, window: Window, genome: list[int]
) -> None:
    d = replay(cfg, window, genome, 0.5)
    keys = {t.key for t in window.tasks}
    got = {(x.transition.anchor, x.transition.task_id) for x in d.transitions}
    assert got <= keys
    # every window task with a decision (not all-masked) has its first decision stored
    firsts = {(x.transition.anchor, x.transition.task_id) for x in d.transitions if x.transition.attempt == 1}
    assert len(firsts) >= len(keys) - sum(1 for _ in [])  # at most the all-masked ones are missing
    assert len(firsts) > 0.9 * len(keys)


def test_expert_flags_mark_the_plans_own_actions(
    window: Window, genome: list[int], demo: Demonstration
) -> None:
    # 13.4.3: expert = a first decision that took its open gene; a masked gene went to greedy (no flag);
    # re-decisions are greedy's (no flag).
    genes = window.genes(genome)
    experts = [d for d in demo.transitions if d.expert]
    plain = [d for d in demo.transitions if not d.expert]
    assert experts and plain
    for d in experts:
        t = d.transition
        assert t.attempt == 1 and OPTIONS[t.a] == genes[(t.anchor, t.task_id)]
    for d in plain:
        t = d.transition
        assert t.attempt > 1 or OPTIONS[t.a] != genes[(t.anchor, t.task_id)]


def test_the_greedy_plan_is_all_expert(cfg: Config, window: Window) -> None:
    # The greedy genome's genes are greedy's own open choices: every first decision is the plan's.
    d = replay(cfg, window, window.greedy_genome, 0.5)
    assert all(x.expert for x in d.transitions if x.transition.attempt == 1)


def test_every_distinct_front_plan_is_replayed_with_its_own_w(cfg: Config, window: Window) -> None:
    # 13.4.4: one demonstration per distinct front point, each with its w_p; the knee is marked. The front
    # labels below are given (demonstrate_front takes the w's from them); the replays are real.
    from pcnme.expert.knee import knee, weights

    rng = random.Random(3)
    genomes = [[rng.randrange(len(t.options)) for t in window.tasks] for _ in range(3)]
    F = [(0.2, 1.4), (0.3, 1.1), (0.6, 1.0)]
    demos = demonstrate_front(cfg, window, [*genomes, genomes[0]], [*F, F[0]])  # a duplicate point: once
    wp, kn = weights(F), knee(F)
    assert kn is not None
    assert [d.plan for d in demos] == [0, 1, 2]
    assert [d.w for d in demos] == [wp[p] for p in F]
    assert [d.is_knee for d in demos] == [p == kn.point for p in F]
    assert [d.objectives for d in demos] == [evaluate(cfg, window, g)[0] for g in genomes]


def test_a_one_point_front_gives_no_demonstrations(cfg: Config, window: Window) -> None:
    g = list(window.greedy_genome)
    assert demonstrate_front(cfg, window, [g, g], [(0.3, 1.0), (0.3, 1.0)]) == []


def test_w_must_be_a_weight(cfg: Config) -> None:
    with pytest.raises(ValueError):
        with_w(cfg, 0.0)
    assert with_w(cfg, 0.7).reward.w == 0.7


def test_the_demonstrations_are_stored_to_parquet(demo: Demonstration, tmp_path: Path) -> None:
    path = write(tmp_path / "demos" / "S2-w0.parquet", [demo])
    t = pq.read_table(path)
    assert t.num_rows == len(demo.transitions) == len(rows([demo]))
    names = t.column_names
    assert [n for n in names if n.startswith("x_")] == [f"x_{n}" for n in INPUT_NAMES]
    assert [n for n in names if n.startswith("xn_")] == [f"xn_{n}" for n in INPUT_NAMES]
    assert {f"valid_{o}" for o in OPTIONS} | {f"valid_next_{o}" for o in OPTIONS} <= set(names)
    r = t.to_pylist()
    assert sum(x["expert"] for x in r) == sum(d.expert for d in demo.transitions)
    assert all(x[f"valid_{x['action']}"] for x in r)  # the action taken was open
    trunc = [x for x in r if x["truncated"] and x["dt_s"] is None]
    assert all(x["xn_w"] is None for x in trunc)


def test_a_re_decision_of_a_planned_task_is_not_expert(demo: Demonstration) -> None:
    from dataclasses import replace

    from pcnme.expert.demos import is_expert

    t = demo.transitions[0].transition
    assert is_expert(t, {t.task_id}) and not is_expert(t, set())
    assert not is_expert(replace(t, attempt=2), {t.task_id})


def test_a_truncated_transition_has_an_empty_s_prime_row(demo: Demonstration) -> None:
    from dataclasses import replace

    from pcnme.agent.nstep import build
    from pcnme.expert.demos import DemoTransition

    t = replace(demo.transitions[0].transition, s_next=None, valid_next=None, dt=None, truncated=True)
    (smp,) = build([t], 10, 0.9, 0.1)
    d = replace(demo, transitions=(DemoTransition(t, False, smp),))
    (row,) = rows([d])
    assert row["truncated"] and row["dt_s"] is None
    assert all(row[f"xn_{n}"] is None for n in INPUT_NAMES) and row["valid_next_here"] is None
    assert (row["has1"], row["hasn"], row["m"], row["xm_w"]) == (False, False, 0, None)


def test_the_stored_samples_load_back_as_they_were(demo: Demonstration, tmp_path: Path) -> None:
    # 13.5: the n-step part is computed over the anchor's whole chain at replay and read back for training.
    from pcnme.expert.demos import load_samples

    path = write(tmp_path / "S2-w0.parquet", [demo])
    loaded = load_samples([path])
    assert [s for _, s in loaded] == [d.sample for d in demo.transitions]
    assert {g for g, _ in loaded} == {("S2", 0)}
    assert any(s.hasn and s.m == 10 for _, s in loaded)  # full 10-step chains
    assert all(s.expert == d.expert for (_, s), d in zip(loaded, demo.transitions, strict=True))


# --- 16.4: the expert filter and the random-plan expert (QUESTIONS Q49) -------------------------------------


def test_the_expert_filter_by_hand() -> None:
    from pcnme.expert.demos import beats_greedy

    # w 0.5: knee cost 0.5 x 0.2 + 0.5 x 1.0 = 0.6; greedy 0.5 x 0.3 + 0.5 x 1.1 = 0.7: 14.3 % better
    assert beats_greedy((0.2, 1.0), (0.3, 1.1), 0.5, 0.14)
    assert not beats_greedy((0.2, 1.0), (0.3, 1.1), 0.5, 0.15)
    assert beats_greedy((0.4, 1.2), (0.3, 1.1), 0.5, 0.0)  # margin 0: no filter, even a worse knee stays


def test_the_rows_carry_the_objectives(demo: Demonstration, window: Window) -> None:
    r = rows([demo])[0]
    assert (r["f1"], r["f2"]) == (demo.objectives.f1, demo.objectives.f2)
    assert (r["greedy_f1"], r["greedy_f2"]) == (window.greedy.f1, window.greedy.f2)


def test_loading_filters_windows_whose_knee_does_not_beat_greedy(
    cfg: Config, window: Window, tmp_path: Path
) -> None:
    from dataclasses import replace

    from pcnme.expert.demos import load_samples

    good = replace(replay(cfg, window, window.greedy_genome, 0.5), is_knee=True)  # the knee = greedy: 0 %
    path = write(tmp_path / "w0.parquet", [good])
    assert len(load_samples([path])) == len(good.transitions)
    assert len(load_samples([path], 0.0)) == len(good.transitions)
    assert load_samples([path], 0.01) == []  # not 1 % better than greedy: the window is left out
    # only the knee counts: another front plan far better than greedy does not keep the window
    from pcnme.expert.window import Objectives

    other = replace(good, plan=1, is_knee=False, objectives=Objectives(0.01, 0.01, good.objectives.n, 0))
    both = write(tmp_path / "w0b.parquet", [good, other])
    assert load_samples([both], 0.01) == []


def test_a_random_plan_teacher(cfg: Config) -> None:
    from pcnme.expert.demos import random_plan

    w = discover(cfg, WindowSpec("S2", window_seed(cfg, "S2", 0), 0, n_cars=40))
    g1, g2 = random_plan(cfg, w), random_plan(cfg, w)
    assert g1 == g2 and len(g1) == len(w.tasks)  # from the window's own stream: the same plan every time
    assert all(0 <= g < len(t.options) for g, t in zip(g1, w.tasks, strict=True))
    assert g1 != w.greedy_genome
