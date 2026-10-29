"""The methods in the agent's seat and how the learning ones are trained (final-design 16.1, 13.5, 18; PLAN
M7). THIS TOUCHES THE AGENT: the baselines use its 48 inputs, 6 actions, masks, reward and time-based
discount.

    pcnme          D3QN, DQfD pretraining on the knee demonstrations, then online (ε online)
    d3qn-scratch   the same network, no pretraining, online with the baselines' ε schedule
    d3qn-greedy    the same pretraining with the greedy plans as teacher, then online (ε online)
    tang-wong      DQN with an LSTM over the last H inputs, dueling + double (methods/lstm.py)
    qeco           its D3QN + LSTM, on our inputs and reward (methods/lstm.py)
    vfc-ppo        PPO actor-critic, masked actions get probability 0 (methods/ppo.py)
    greedy, random the rules of 16.1 (methods/simple.py)
    knee-reference the NSGA-II knee plan replayed, fast mode only (experiments/runner.py)

Every learning method trains on the same budget in decisions (16.1): one Budget shared by its zones' agents;
they learn until it is used up. Training runs on training seeds only, with the neighbour order shuffled per
episode (13.6).
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pcnme.agent.budget import Budget
from pcnme.agent.dqfd import Group, pretrain
from pcnme.agent.network import make_network
from pcnme.agent.nstep import Sample
from pcnme.agent.online import Episode, OnlineAgent, train_online
from pcnme.config import Config, EpsilonSchedule
from pcnme.sim.rng import RandomStreams

D3QN_FAMILY = ("pcnme", "d3qn-scratch", "d3qn-greedy")
LSTM_FAMILY = ("tang-wong", "qeco")
LEARNING = (*D3QN_FAMILY, *LSTM_FAMILY, "vfc-ppo")
RULES = ("greedy", "random")
METHODS = (*LEARNING, *RULES, "knee-reference")  # docs/METRICS.md method tags (16.1)
Demos = Sequence[tuple[Group, Sample]]


@dataclass
class Trained:
    """A trained learning method: each zone's network weights after training, its expert data, the budget
    it used and the training episodes."""

    method: str
    states: dict[str, dict[str, Any]]
    expert: tuple[Sample, ...]
    budget: Budget
    episodes: list[Episode] = field(default_factory=list)
    agreement: float | None = None  # the pretraining's best held-out expert agreement
    schedule: EpsilonSchedule | None = None  # the ε schedule it trained with (None: ε online)


def training_seeds(cfg: Config, start: int | None = None) -> Iterable[int]:
    """Training episodes use the training seed range only (16.3), in order."""
    r = cfg.fast_mode.seed_split["training"]
    first = r.start if start is None else start
    if first not in r:
        raise ValueError(f"training seed {first} is outside the training range {r.start}-{r.stop - 1}")
    return range(first, r.stop)


def _seed(cfg: Config, method: str, use: str, seed: int) -> tuple[int, np.random.Generator]:
    s = RandomStreams.from_config(cfg, seed, "training")
    return int(s.generator("agent", method, use).integers(2**31 - 1)), s.generator(
        "agent", method, f"{use}-rng"
    )


def train_d3qn(
    cfg: Config,
    method: str,
    scenario: str,
    zones: Sequence[str],
    budget_decisions: int,
    *,
    demos: Demos = (),
    seconds: float | None = None,
    seed: int | None = None,
    n_cars: int | None = None,
    pretrain_updates: int | None = None,
    episode_hook: Callable[[Episode], None] | None = None,
) -> Trained:
    """PCNME, D3QN from scratch, or D3QN + greedy expert (16.1). 16.4 ablations: the expert data cleared
    after pretraining (ablation.expert clear), or pretraining only (ablation.online false: no online
    episode, the budget unused)."""
    if method not in D3QN_FAMILY:
        raise ValueError(f"{method} is not a D3QN method")
    if not cfg.ablation.online and method == "d3qn-scratch":
        raise ValueError("pretraining only needs a method that pretrains (pcnme, d3qn-greedy)")
    seeds = iter(training_seeds(cfg, seed))
    first = next(iter(training_seeds(cfg, seed)))
    init, rng = _seed(cfg, method, "init", first)
    agreement = None
    expert: tuple[Sample, ...]
    if method == "d3qn-scratch":
        net, expert = make_network(cfg, init), ()
    else:
        if not demos:
            raise ValueError(
                f"{method} needs demonstrations ({'knee' if method == 'pcnme' else 'greedy'} plans)"
            )
        pre = pretrain(cfg, demos, rng, init, updates=pretrain_updates)
        net, agreement = pre.network, pre.best_agreement
        expert = () if cfg.ablation.expert == "clear" else tuple(s for _, s in demos)
    budget = Budget(budget_decisions)
    schedule = cfg.rl.epsilon_baselines if method == "d3qn-scratch" else None  # 18: baselines 1 -> 0.01
    agents = {
        z: OnlineAgent(cfg, z, net, expert, seed=first, budget=budget, schedule=schedule) for z in zones
    }
    out = Trained(method, {}, expert, budget, agreement=agreement, schedule=schedule)
    length = cfg.fast_mode.episode_length if seconds is None else seconds
    while cfg.ablation.online and not budget.exhausted:
        (ep,) = train_online(cfg, agents, scenario, [next(seeds)], length, n_cars)
        out.episodes.append(ep)
        if episode_hook is not None:
            episode_hook(ep)
    out.states = {z: copy.deepcopy(a.online.state_dict()) for z, a in agents.items()}
    return out


def deployed_d3qn(cfg: Config, trained: Trained, zones: Sequence[str], seed: int) -> dict[str, OnlineAgent]:
    """The trained agents deployed for one evaluation episode: each zone from its own weights, learning
    online with ε online and its expert data kept (13.5; QUESTIONS Q42 stand-in); frozen when pretraining
    only (16.4)."""
    out = {}
    for z in zones:
        net = make_network(cfg, 0)
        net.load_state_dict(trained.states[z])
        out[z] = OnlineAgent(
            cfg, z, net, trained.expert, seed=seed, usage="evaluation", learn=cfg.ablation.online
        )
    return out
