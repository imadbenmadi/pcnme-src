"""The learning baselines' training (PLAN M7; final-design 16.1, 18): the budget in decisions, the ε schedule,
the D3QN family."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pcnme.agent.budget import Budget, epsilon  # noqa: E402
from pcnme.agent.network import make_network  # noqa: E402
from pcnme.agent.nstep import build  # noqa: E402
from pcnme.config import Config, EpsilonSchedule, load_config  # noqa: E402
from pcnme.methods.learners import METHODS, deployed_d3qn, train_d3qn, training_seeds  # noqa: E402
from tests.agent.test_losses import chain  # noqa: E402

ZONES = ("z1", "z2", "z3", "z4")


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


def test_the_baselines_epsilon_schedule_by_hand(cfg: Config) -> None:
    s = cfg.rl.epsilon_baselines
    assert (s.start, s.end, s.fraction) == (1.0, 0.01, 0.2)  # 18
    assert epsilon(s, 0, 100) == 1.0
    assert epsilon(s, 10, 100) == pytest.approx(1.0 - 0.99 * 0.5)  # half-way down the first 20
    assert epsilon(s, 20, 100) == 0.01 == epsilon(s, 99, 100)
    assert epsilon(EpsilonSchedule(1.0, 0.1, 0.0), 0, 100) == 0.1


def test_the_budget_counts_decisions_and_stops_at_its_total() -> None:
    b = Budget(3)
    for _ in range(5):
        b.spend()
    assert b.used == 3 and b.exhausted


def test_agents_sharing_a_budget_stop_learning_together(cfg: Config) -> None:
    from pcnme.agent.online import OnlineAgent
    from tests.agent.test_inputs import make_ctx

    b = Budget(4)
    net = make_network(cfg, 1)
    a1, a2 = OnlineAgent(cfg, "z1", net, budget=b), OnlineAgent(cfg, "z2", net, budget=b)
    ctx = replace(make_ctx(cfg, "T1"), inputs=(0.0,) * 48)
    for _ in range(2):
        a1.choose(ctx)
        a2.choose(ctx)
    assert b.exhausted and not a1.learning and not a2.learning
    a1._add(build(chain(), 2, 0.9, 0.1) * 5)
    assert a1.updates == 0  # no learning after the budget


def test_the_schedule_decays_with_the_shared_budget(cfg: Config) -> None:
    from pcnme.agent.online import OnlineAgent
    from tests.agent.test_inputs import make_ctx

    b = Budget(100)
    a = OnlineAgent(cfg, "z1", make_network(cfg, 1), budget=b, schedule=cfg.rl.epsilon_baselines)
    assert a.current_epsilon() == 1.0
    ctx = replace(make_ctx(cfg, "T1"), inputs=(0.0,) * 48)
    for _ in range(20):
        a.choose(ctx)
    assert a.current_epsilon() == pytest.approx(0.01)


def test_training_stops_at_the_budget_in_decisions(cfg: Config) -> None:
    t = train_d3qn(cfg, "d3qn-scratch", "S2", ZONES, 400, seconds=2.0, n_cars=40)
    assert t.budget.used == 400 and t.budget.exhausted and len(t.episodes) >= 1
    assert set(t.states) == set(ZONES) and t.expert == () and t.schedule == cfg.rl.epsilon_baselines
    assert t.episodes[-1].updates <= 400 // 4 + 4  # one update per 4 transitions


def test_pretrained_methods_need_demonstrations_and_use_them(cfg: Config) -> None:
    from pcnme.expert.demos import greedy_demonstrations
    from pcnme.expert.window import WindowSpec, discover, window_seed
    from tests.zone.harness import fixed_delay_config

    small = fixed_delay_config(nsga2__T_warmup=1.0, nsga2__T_win=0.3)
    small = replace(small, dqfd=replace(small.dqfd, eval_every=20, patience=2))
    with pytest.raises(ValueError, match="demonstrations"):
        train_d3qn(small, "d3qn-greedy", "S2", ZONES, 100, seconds=2.0, n_cars=40)
    demos = []
    for k in range(2):
        w = discover(small, WindowSpec("S2", window_seed(small, "S2", k), k, n_cars=40))
        demos += [(("S2", k), t.sample) for t in greedy_demonstrations(small, w).transitions]
    t = train_d3qn(
        small, "d3qn-greedy", "S2", ZONES, 200, demos=demos, seconds=2.0, n_cars=40, pretrain_updates=60
    )
    assert t.budget.used == 200 and len(t.expert) == len(demos) and t.agreement is not None
    assert t.schedule is None  # pretrained: ε online
    deployed = deployed_d3qn(small, t, ZONES, 3000)
    assert all(len(a.replay.expert) == len(demos) for a in deployed.values())
    for z in ZONES:
        sd = deployed[z].online.state_dict()
        assert all(torch.equal(sd[k], t.states[z][k]) for k in sd)


def test_training_uses_training_seeds_only(cfg: Config) -> None:
    assert next(iter(training_seeds(cfg))) == 2000
    with pytest.raises(ValueError, match="training range"):
        training_seeds(cfg, 3000)


def test_the_method_names_are_the_metrics_tags() -> None:
    # docs/METRICS.md's method tags
    tags = "pcnme d3qn-scratch d3qn-greedy vfc-ppo tang-wong qeco greedy random knee-reference"
    assert set(METHODS) == set(tags.split())
