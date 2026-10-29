"""DQfD pretraining on the demonstrations (final-design 13.5, 18 Pretraining row; PLAN M6.4). THIS TOUCHES THE
AGENT.

One global network learns from the demonstrations with the DQfD loss (agent/losses.py) and prioritized replay
(agent/replay.py): every demonstration transition goes into the expert part, never overwritten (the margin
loss still applies to the expert-flagged ones only). Adam, learning rate, batch and the hard target copy every
rl.target_update_every updates come from the config. Every dqfd.eval_every updates the held-out expert
agreement is measured: the share of held-out expert transitions on which the network's masked argmax is the
expert's action. Training stops at dqfd.pretrain_updates, or early once the agreement has not risen for
dqfd.patience measurements; the network kept is the one with the best agreement.

Engineering choices: the held-out set is whole windows (no transition of a held-out window is trained on),
a share dqfd.holdout_share of them, at least one, drawn from the agent stream; β is annealed over the update
budget.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from pcnme.agent.losses import Batch, dqfd_loss
from pcnme.agent.network import DuelingQNet, deterministic, make_network, masked_argmax
from pcnme.agent.nstep import Sample
from pcnme.agent.replay import Replay, beta_at
from pcnme.config import Config

Group = tuple[str, int]  # (scenario, window)


@dataclass
class Curve:
    """What M6.6 plots: per measurement, the update and the agreement; per update the loss parts."""

    agreement: list[tuple[int, float]] = field(default_factory=list)
    # per update: the update, J_DQ, J_n, J_E, the total and the batch's mean Q
    losses: list[tuple[int, float, float, float, float, float]] = field(default_factory=list)
    upsample: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class Pretrained:
    network: DuelingQNet
    curve: Curve
    updates: int
    best_agreement: float
    stopped_early: bool
    held_out: set[Group]  # the windows never trained on
    replay: Replay


def split(groups: Sequence[Group], share: float, rng: np.random.Generator) -> set[Group]:
    """The held-out windows: a share of the distinct windows, at least one (and one left to train on)."""
    distinct = sorted(set(groups))
    if len(distinct) < 2:
        raise ValueError("the held-out split needs at least 2 windows of demonstrations")
    k = min(max(1, round(share * len(distinct))), len(distinct) - 1)
    pick = rng.permutation(len(distinct))[:k]
    return {distinct[i] for i in pick}


def agreement(net: torch.nn.Module, samples: Sequence[Sample]) -> float:
    """The share of expert transitions on which the masked argmax is the expert's action."""
    expert = [s for s in samples if s.expert]
    if not expert:
        return 0.0
    with torch.no_grad():
        b = Batch.of(expert)
        return float((masked_argmax(net(b.s), b.valid) == b.a).float().mean())


def pretrain(
    cfg: Config,
    demos: Sequence[tuple[Group, Sample]],
    rng: np.random.Generator,
    seed: int,
    *,
    updates: int | None = None,
) -> Pretrained:
    deterministic()
    held = split([g for g, _ in demos], cfg.dqfd.holdout_share, rng)
    train = [s for g, s in demos if g not in held]
    test = [s for g, s in demos if g in held]
    budget = updates if updates is not None else cfg.dqfd.pretrain_updates
    online = make_network(cfg, seed)
    target = copy.deepcopy(online)
    opt = torch.optim.Adam(online.parameters(), lr=cfg.rl.lr)
    replay = Replay(cfg, rng, expert=train)
    curve = Curve(agreement=[(0, agreement(online, test))])  # the untrained network's, for the curve
    best, best_state, since, u = -1.0, copy.deepcopy(online.state_dict()), 0, 0
    stopped = False
    for u in range(1, budget + 1):
        drawn = replay.draw(cfg.rl.batch, beta_at(cfg, u, budget))
        parts = dqfd_loss(online, target, Batch.of(drawn.samples, drawn.weights), cfg)
        opt.zero_grad()
        parts.total.backward()
        opt.step()
        replay.update(drawn.index, parts.td_errors.tolist())
        curve.losses.append(
            (u, parts.td, parts.n_step, parts.margin, float(parts.total.detach()), parts.mean_q)
        )
        if u % cfg.rl.target_update_every == 0:
            target.load_state_dict(online.state_dict())
        if u % cfg.dqfd.eval_every == 0:
            a = agreement(online, test)
            curve.agreement.append((u, a))
            curve.upsample.append((u, replay.upsample_ratio(drawn)))
            if a > best:
                best, best_state, since = a, copy.deepcopy(online.state_dict()), 0
            else:
                since += 1
                if since >= cfg.dqfd.patience:
                    stopped = True
                    break
    online.load_state_dict(best_state)
    return Pretrained(online, curve, u, best, stopped, held, replay)
