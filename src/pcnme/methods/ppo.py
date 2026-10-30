"""VFC PPO in the agent's seat (final-design 16.1, 18 RL rows; PLAN M7). THIS TOUCHES THE AGENT: the same 48
inputs, 6 actions, masks and time-based discount; masked actions get probability 0.

An actor-critic (48 -> 256 -> 256, a policy head of 6 and a value head) trained with PPO's clipped objective
(Schulman et al.): every rl.ppo.rollout decisions of a zone, rl.ppo.epochs epochs of minibatches of
rl.ppo.minibatch, clip rl.ppo.clip, Adam rl.ppo.lr, value loss weight vf_coef and entropy bonus ent_coef.
Advantages by GAE over the zone's decisions in order, with the time-based discount per step: δ_t = r_t +
γ^(Δt/τ) V(s_{t+1}) - V(s_t), A_t = δ_t + γ^(Δt/τ) λ A_{t+1}. VFC optimises delay only (16.1): its reward
is the 11.4 reward with w = 1 (QUESTIONS Q43 stand-in), set by the caller through the config.

Engineering choices: one trunk shared by both heads; the action is sampled from the agent's own stream; a
transition without s' (Q36) has no advantage and is left out of the update; advantages normalised per
update; the log-probability is the one the policy had at the decision; a rollout left at the episode end is
used if it holds at least one minibatch, else dropped.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn

from pcnme.agent.budget import Budget
from pcnme.agent.inputs import N_INPUTS, Decision
from pcnme.agent.network import N_ACTIONS, deterministic
from pcnme.agent.nstep import Sample
from pcnme.agent.online import OnlineChain, train_online
from pcnme.agent.transitions import Transition
from pcnme.config import Config
from pcnme.methods.learners import Trained, _seed, training_seeds
from pcnme.sim.rng import RandomStreams
from pcnme.zone.decision import OPTIONS, DecisionContext, Option

Key = tuple[object, int]


class ActorCritic(nn.Module):
    def __init__(self, hidden: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = N_INPUTS
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        self.body = nn.Sequential(*layers)
        self.policy = nn.Linear(width, N_ACTIONS)
        self.value = nn.Linear(width, 1)

    def forward(self, x: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        """Masked log-probabilities (masked actions: probability 0) and V(s)."""
        z = self.body(x)
        logits = self.policy(z).masked_fill(~valid, float("-inf"))
        return torch.log_softmax(logits, dim=-1), self.value(z).squeeze(-1)


def make_actor_critic(cfg: Config, seed: int) -> ActorCritic:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ActorCritic(cfg.rl.hidden)


def clipped_objective(ratio: Tensor, adv: Tensor, clip: float) -> Tensor:
    """PPO's clipped surrogate, as a loss: -mean(min(ρ A, clip(ρ, 1 - ε, 1 + ε) A))."""
    return -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()


def gae(
    r: np.ndarray, v: np.ndarray, v1: np.ndarray, g: np.ndarray, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and returns over consecutive steps, γ^(Δt/τ) per step (g)."""
    adv = np.zeros_like(r)
    nxt = 0.0
    for t in range(len(r) - 1, -1, -1):
        delta = r[t] + g[t] * v1[t] - v[t]
        nxt = delta + g[t] * lam * nxt
        adv[t] = nxt
    return adv, adv + v


@dataclass
class PpoAgent:
    """One zone's PPO agent: a Placement that learns from its zone's transitions, on-policy."""

    cfg: Config
    zone_id: str
    network: ActorCritic
    seed: int = 2000
    usage: str = "training"
    learn: bool = True
    budget: Budget | None = None
    updates: int = 0
    losses: list[tuple[int, float, float, float, float]] = field(
        default_factory=list
    )  # update, policy, value, entropy, total

    def __post_init__(self) -> None:
        deterministic()
        streams = RandomStreams.from_config(self.cfg, self.seed, self.usage)
        self._rng = streams.generator("agent", self.zone_id, "ppo")
        self.net = copy.deepcopy(self.network)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.rl.ppo.lr)
        rl = self.cfg.rl
        self.chain = OnlineChain(1, rl.gamma_per_tau, rl.tau)
        self._logp: dict[Key, float] = {}
        self._last_logp = 0.0
        self.rollout: list[tuple[Sample, float]] = []

    @property
    def learning(self) -> bool:
        return self.learn and (self.budget is None or not self.budget.exhausted)

    def choose(self, ctx: DecisionContext) -> Option | None:
        valid = [v.masked is None for v in ctx.views]
        if not any(valid):
            return None
        with torch.no_grad():
            logp, _ = self.net(torch.tensor([ctx.inputs], dtype=torch.float32), torch.tensor([valid]))
        p = logp.exp()[0].numpy().astype(np.float64)
        p = p / p.sum()
        a = int(self._rng.choice(N_ACTIONS, p=p))
        self._last_logp = float(logp[0, a])
        if self.budget is not None:
            self.budget.spend()
        return OPTIONS[a]

    def attach(self, zone: object) -> None:
        buf = zone.transitions  # type: ignore[attr-defined]
        buf.on_decision_hook = self._decision
        buf.on_replay_hook = self._transition

    def _decision(self, d: Decision) -> None:
        key = (d.task_id, d.attempt)
        self.chain.decision(key)
        self._logp[key] = self._last_logp

    def _transition(self, t: Transition) -> None:
        start = self.chain.next
        samples = self.chain.transition(t)
        self._collect(self.chain.order[start : start + len(samples)], samples)

    def end_episode(self) -> None:
        keys = [k for k in self.chain.order[self.chain.next :] if k in self.chain.done]
        self._collect(keys, self.chain.flush())
        if self.learning and len(self.rollout) >= self.cfg.rl.ppo.minibatch:
            self.update()
        self.rollout.clear()
        self._logp.clear()

    def _collect(self, keys: Sequence[Key], samples: Sequence[Sample]) -> None:
        for k, s in zip(keys, samples, strict=True):
            self.rollout.append((s, self._logp.pop(k, 0.0)))
            if self.learning and len(self.rollout) >= self.cfg.rl.ppo.rollout:
                self.update()
                self.rollout.clear()

    def update(self) -> None:
        """PPO on the rollout (Schulman et al.), the samples with an s' only (Q36)."""
        ppo = self.cfg.rl.ppo
        rows = [(s, lp) for s, lp in self.rollout if s.has1 and s.s1 is not None and s.valid1 is not None]
        if len(rows) < 2:
            return
        s = torch.tensor([x.s for x, _ in rows], dtype=torch.float32)
        valid = torch.tensor([x.valid for x, _ in rows])
        s1 = torch.tensor([x.s1 for x, _ in rows], dtype=torch.float32)
        valid1 = torch.tensor([x.valid1 for x, _ in rows])
        a = torch.tensor([x.a for x, _ in rows])
        old = torch.tensor([lp for _, lp in rows], dtype=torch.float32)
        with torch.no_grad():
            _, v = self.net(s, valid)
            _, v1 = self.net(s1, valid1)
        r = np.array([x.r for x, _ in rows])
        g = np.array([x.g1 for x, _ in rows])
        adv_np, ret_np = gae(
            r, v.numpy().astype(np.float64), v1.numpy().astype(np.float64), g, ppo.gae_lambda
        )
        adv = torch.tensor(adv_np, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv
        ret = torch.tensor(ret_np, dtype=torch.float32)
        n = len(rows)
        for _ in range(ppo.epochs):
            order = self._rng.permutation(n)
            for start in range(0, n, ppo.minibatch):
                idx = torch.tensor(order[start : start + ppo.minibatch])
                logp, value = self.net(s[idx], valid[idx])
                lp_a = logp.gather(1, a[idx].unsqueeze(1)).squeeze(1)
                policy = clipped_objective(torch.exp(lp_a - old[idx]), adv[idx], ppo.clip)
                value_loss = ((value - ret[idx]) ** 2).mean()
                probs = logp.exp()
                entropy = -(probs * logp.masked_fill(~valid[idx], 0.0)).sum(dim=1).mean()
                loss = policy + ppo.vf_coef * value_loss - ppo.ent_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        self.updates += 1
        parts = (float(policy.detach()), float(value_loss.detach()), float(entropy.detach()))
        self.losses.append((self.updates, *parts, float(loss.detach())))


def train_ppo(
    cfg: Config,
    scenario: str,
    zones: Sequence[str],
    budget_decisions: int,
    *,
    seconds: float | None = None,
    seed: int | None = None,
    n_cars: int | None = None,
) -> Trained:
    """VFC PPO on the shared budget (16.1). `cfg` must carry its reward (w = 1, Q43)."""
    seeds = iter(training_seeds(cfg, seed))
    first = next(iter(training_seeds(cfg, seed)))
    init, _ = _seed(cfg, "vfc-ppo", "init", first)
    net = make_actor_critic(cfg, init)
    budget = Budget(budget_decisions)
    agents = {z: PpoAgent(cfg, z, net, seed=first, budget=budget) for z in zones}
    out = Trained("vfc-ppo", {}, (), budget)
    length = cfg.fast_mode.episode_length if seconds is None else seconds
    while not budget.exhausted:
        (ep,) = train_online(cfg, agents, scenario, [next(seeds)], length, n_cars)  # type: ignore[arg-type]
        out.episodes.append(ep)
    out.states = {z: copy.deepcopy(a.net.state_dict()) for z, a in agents.items()}
    return out


def deployed_ppo(cfg: Config, trained: Trained, zones: Sequence[str], seed: int) -> dict[str, PpoAgent]:
    out = {}
    for z in zones:
        net = make_actor_critic(cfg, 0)
        net.load_state_dict(trained.states[z])
        out[z] = PpoAgent(cfg, z, net, seed=seed, usage="evaluation")
    return out
