"""The expert data online (final-design 16.4: kept with priorities, shrinking share + fading margin, cleared
after pretraining; Hester et al., Zhang et al.) and pretraining only. THIS TOUCHES THE AGENT."""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from pcnme.agent.budget import Budget  # noqa: E402
from pcnme.agent.network import make_network  # noqa: E402
from pcnme.agent.nstep import build  # noqa: E402
from pcnme.agent.online import OnlineAgent  # noqa: E402
from pcnme.config import Config  # noqa: E402
from tests.agent.test_dqfd import cfg, toy  # noqa: E402, F401
from tests.agent.test_losses import chain  # noqa: E402

ZONES = ("z1", "z2", "z3", "z4")


def mode(c: Config, expert: str = "keep", online: bool = True) -> Config:
    return replace(c, ablation=replace(c.ablation, expert=expert, online=online))


def experts(n: int = 4) -> list[object]:
    return [replace(s, expert=True) for s in build(chain(), 2, 0.9, 0.1)][:n]


def test_the_default_keeps_the_expert_data(cfg: Config) -> None:  # noqa: F811
    assert (cfg.ablation.expert, cfg.ablation.online) == ("keep", True)
    a = OnlineAgent(cfg, "z1", make_network(cfg, 1), experts(), budget=Budget(100))  # type: ignore[arg-type]
    a.budget.used = 50  # type: ignore[union-attr]
    a._add(build(chain(), 2, 0.9, 0.1) * 10)
    assert a.replay.expert_scale == 1.0 and a.margin_scale() == 1.0


def test_fading_shrinks_the_share_and_the_margin_with_the_budget(cfg: Config) -> None:  # noqa: F811
    c = mode(cfg, "fade")
    budget = Budget(100)
    a = OnlineAgent(c, "z1", make_network(c, 1), experts(), budget=budget)  # type: ignore[arg-type]
    assert a.margin_scale() == 1.0
    budget.used = 30
    a._add(build(chain(), 2, 0.9, 0.1) * 10)  # updates at 4 and 8 samples
    assert (
        a.updates > 0
        and a.replay.expert_scale == pytest.approx(0.7)
        and a.margin_scale() == pytest.approx(0.7)
    )
    budget.used = 100
    assert a.margin_scale() == 0.0
    deployed = OnlineAgent(c, "z1", make_network(c, 1), experts(), usage="training")  # type: ignore[arg-type]
    assert deployed.margin_scale() == 0.0 and deployed.replay.expert_scale == 0.0  # faded out by deployment


def test_the_scaled_margin_weight_reaches_the_loss(cfg: Config) -> None:  # noqa: F811
    c = mode(cfg, "fade")
    budget = Budget(100, used=75)
    a = OnlineAgent(c, "z1", make_network(c, 1), experts(), budget=budget)  # type: ignore[arg-type]
    assert a.loss_config().dqfd.lambda_margin == pytest.approx(0.25 * c.dqfd.lambda_margin)
    assert a.loss_config().dqfd.lambda_n == c.dqfd.lambda_n
    keep = OnlineAgent(cfg, "z1", make_network(cfg, 1), experts(), budget=budget)  # type: ignore[arg-type]
    assert keep.loss_config() is cfg


def test_training_clears_or_keeps_the_expert_data(cfg: Config) -> None:  # noqa: F811
    from pcnme.methods.learners import deployed_d3qn, train_d3qn

    for m, n_expert in (("keep", len(toy())), ("clear", 0)):
        c = mode(cfg, m)
        t = train_d3qn(c, "pcnme", "S2", ZONES, 80, demos=toy(), seconds=1.0, n_cars=20, pretrain_updates=20)
        assert len(t.expert) == n_expert and t.agreement is not None
        assert all(len(a.replay.expert) == n_expert for a in deployed_d3qn(c, t, ZONES, 3000).values())


def test_pretraining_only_skips_online_learning(cfg: Config) -> None:  # noqa: F811
    from pcnme.methods.learners import deployed_d3qn, train_d3qn

    c = mode(cfg, online=False)
    t = train_d3qn(c, "pcnme", "S2", ZONES, 80, demos=toy(), seconds=1.0, n_cars=20, pretrain_updates=20)
    assert t.episodes == [] and t.budget.used == 0
    assert all(not a.learning for a in deployed_d3qn(c, t, ZONES, 3000).values())
    with pytest.raises(ValueError, match="pretrain"):
        train_d3qn(c, "d3qn-scratch", "S2", ZONES, 80, seconds=1.0, n_cars=20)
