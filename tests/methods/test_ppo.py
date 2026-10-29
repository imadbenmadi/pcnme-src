"""VFC PPO (PLAN M7; final-design 16.1, 18: clip 0.2, lr 3e-4, GAE 0.95, 10 epochs, minibatch 64)."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pcnme.agent.nstep import Sample  # noqa: E402
from pcnme.config import Config, load_config  # noqa: E402
from pcnme.methods.ppo import (  # noqa: E402
    PpoAgent,
    clipped_objective,
    deployed_ppo,
    gae,
    make_actor_critic,
    train_ppo,
)
from tests.agent.test_inputs import make_ctx  # noqa: E402

ZONES = ("z1", "z2", "z3", "z4")
OPTS = ["here", "n1", "n2", "n3", "n4", "cloud"]


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


def test_masked_actions_get_probability_0_and_are_never_chosen(cfg: Config) -> None:
    net = make_actor_critic(cfg, 1)
    valid = torch.tensor([[True, False, True, False, False, False]])
    logp, v = net(torch.randn(1, 48, generator=torch.Generator().manual_seed(0)), valid)
    p = logp.exp()[0].detach()
    assert float(p[~valid[0]].sum()) == 0.0 and float(p.sum()) == pytest.approx(1.0) and v.shape == (1,)
    agent = PpoAgent(cfg, "z1", net, learn=False)
    base = make_ctx(cfg, "T1")
    rng = random.Random(0)
    for _ in range(300):
        mask = [rng.random() < 0.5 for _ in range(6)]
        if not any(mask):
            mask[rng.randrange(6)] = True
        views = tuple(replace(x, masked=None if m else "full") for x, m in zip(base.views, mask, strict=True))
        ctx = replace(base, views=views, inputs=tuple(rng.uniform(-1, 2) for _ in range(48)))
        a = agent.choose(ctx)
        assert a is not None and mask[OPTS.index(a)]
    closed = replace(base, views=tuple(replace(x, masked="full") for x in base.views), inputs=(0.0,) * 48)
    assert agent.choose(closed) is None


def test_gae_by_hand() -> None:
    # step 1: delta = 1 + 0.9 x 0 - 0.5 = 0.5; step 0: delta = 1 + 0.9 x 0.5 - 0.5 = 0.95,
    # A0 = 0.95 + 0.9 x 0.95 x 0.5 = 1.3775 (the time-based discount 0.9 per step here)
    adv, ret = gae(
        np.array([1.0, 1.0]), np.array([0.5, 0.5]), np.array([0.5, 0.0]), np.array([0.9, 0.9]), 0.95
    )
    assert adv.tolist() == pytest.approx([1.3775, 0.5]) and ret.tolist() == pytest.approx([1.8775, 1.0])


def bandit(n: int, seed: int) -> list[tuple[Sample, float]]:
    # Two valid actions; action 0 pays 1, action 1 pays 0; a one-step target (discount 0); old log-prob
    # ln 1/2.
    rng = np.random.default_rng(seed)
    valid = (True, True, False, False, False, False)
    out = []
    for _ in range(n):
        s = tuple(rng.uniform(0, 1, 48))
        a = int(rng.integers(2))
        smp = Sample(s, a, 1.0 - a, valid, False, True, s, valid, 0.0, False, 0, 0.0, None, None, 0.0)
        out.append((smp, float(np.log(0.5))))
    return out


def test_an_update_moves_the_policy_toward_the_paying_action(cfg: Config) -> None:
    agent = PpoAgent(cfg, "z1", make_actor_critic(cfg, 2))
    x = torch.tensor([bandit(1, 9)[0][0].s], dtype=torch.float32)
    valid = torch.tensor([[True, True, False, False, False, False]])
    before = float(agent.net(x, valid)[0].exp()[0, 0])
    for k in range(5):
        agent.rollout = bandit(256, k)
        agent.update()
    after = float(agent.net(x, valid)[0].exp()[0, 0])
    assert agent.updates == 5 and after > before + 0.1  # about 0.50 -> 0.68: clipped steps


def test_samples_without_s_prime_are_left_out(cfg: Config) -> None:
    agent = PpoAgent(cfg, "z1", make_actor_critic(cfg, 3))
    agent.rollout = [(replace(s, has1=False, s1=None, valid1=None), lp) for s, lp in bandit(64, 0)]
    agent.update()
    assert agent.updates == 0  # nothing left to learn from


def test_an_update_every_rollout_decisions(cfg: Config) -> None:
    small = replace(cfg, rl=replace(cfg.rl, ppo=replace(cfg.rl.ppo, rollout=40, minibatch=16)))
    agent = PpoAgent(small, "z1", make_actor_critic(small, 4))
    rows = bandit(100, 1)
    agent._collect([("T", k) for k in range(100)], [s for s, _ in rows])
    assert agent.updates == 2 and len(agent.rollout) == 20  # 40 + 40, 20 waiting


def test_training_stops_at_the_budget_and_deploys(cfg: Config) -> None:
    from pcnme.expert.demos import with_w

    delay_only = with_w(cfg, 1.0)  # Q43: VFC optimises delay only
    t = train_ppo(delay_only, "S2", ZONES, 300, seconds=2.0, n_cars=40)
    assert t.budget.used == 300 and t.method == "vfc-ppo"
    deployed = deployed_ppo(delay_only, t, ZONES, 3000)
    for z in ZONES:
        sd = deployed[z].net.state_dict()
        assert all(torch.equal(sd[k], t.states[z][k]) for k in sd)


def test_the_clipped_objective_by_hand() -> None:
    # ρ 1.5, A 1: min(1.5, 1.2) = 1.2; ρ 0.5, A -1: min(-0.5, -0.8) = -0.8; loss = -mean = -(1.2 - 0.8) / 2
    loss = clipped_objective(torch.tensor([1.5, 0.5]), torch.tensor([1.0, -1.0]), 0.2)
    assert float(loss) == pytest.approx(-0.2)
