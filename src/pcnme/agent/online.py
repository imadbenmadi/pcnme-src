"""Online learning in every zone (final-design 13.5, 13.6, 11.2-11.3, 18 RL row; PLAN M6.5). THIS TOUCHES THE
AGENT.

Every zone keeps its own copy of the network and keeps learning from its own closed attempts; the expert
transitions stay in its replay for good, in the part that is never overwritten (13.5). The agent acts
ε-greedily (rl.epsilon_online) with the masked argmax: it never takes a masked action (11.2); an all-masked
task makes no decision. Its own transitions become samples in decision order once the n-step chain behind
them is final (OnlineChain), one gradient update per 1 / rl.updates_per_transition new transitions, a hard
target copy every rl.target_update_every updates. In training the neighbour order is shuffled per episode
(13.6; City(shuffle_neighbours=True)).

Engineering choices: β is β_final online; the agent's random draws (ε, replay) come from its own streams of a
training seed, so they do not depend on the episodes' seeds; at an episode end the chain is flushed (the last
transitions' n-step parts use the steps that have an s', Q36).
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from pcnme.agent.budget import Budget, epsilon
from pcnme.agent.inputs import Decision
from pcnme.agent.losses import Batch, dqfd_loss
from pcnme.agent.network import deterministic, masked_argmax
from pcnme.agent.nstep import Sample, build
from pcnme.agent.replay import Drawn, Replay
from pcnme.agent.transitions import Transition
from pcnme.config import Config, EpsilonSchedule
from pcnme.sim.rng import RandomStreams
from pcnme.zone.decision import OPTIONS, DecisionContext, Option
from pcnme.zone.zone import TaskRecord

if TYPE_CHECKING:
    from pcnme.fl.fedavg import FedState

Key = tuple[object, int]


class OnlineChain:
    """One anchor's decisions in order; a transition's sample is built once the next n are final."""

    def __init__(self, n: int, gamma_per_tau: float, tau: float) -> None:
        self.n, self.gamma, self.tau = n, gamma_per_tau, tau
        self.order: list[Key] = []
        self.done: dict[Key, Transition] = {}
        self.next = 0  # the first decision whose sample is not out yet

    def decision(self, key: Key) -> None:
        self.order.append(key)

    def transition(self, t: Transition) -> list[Sample]:
        self.done[(t.task_id, t.attempt)] = t
        out = []
        while self.next < len(self.order):
            keys = self.order[self.next : self.next + self.n]
            if len(keys) < self.n or not all(k in self.done for k in keys):
                break
            out.append(build([self.done[k] for k in keys], self.n, self.gamma, self.tau)[0])
            self.done.pop(self.order[self.next])
            self.next += 1
        return out

    def flush(self) -> list[Sample]:
        """The episode ended: every remaining final transition, its n-step part over what is there."""
        out = []
        while self.next < len(self.order):
            k = self.order[self.next]
            if k in self.done:
                run = []
                for kk in self.order[self.next : self.next + self.n]:
                    if kk not in self.done:
                        break
                    run.append(self.done[kk])
                out.append(build(run, self.n, self.gamma, self.tau)[0])
            self.next += 1
        self.order, self.done, self.next = [], {}, 0
        return out


@dataclass
class OnlineAgent:
    """One zone's agent: a Placement (choose) that learns from its zone's transitions."""

    cfg: Config
    zone_id: str
    network: torch.nn.Module  # the D3QN, or an LSTM baseline's network
    expert: Sequence[Sample] = ()
    seed: int = 2000  # the agent's own streams: a seed of `usage`'s range
    usage: str = "training"  # evaluation for a deployed agent (16.3: evaluation seeds only there)
    learn: bool = True
    epsilon: float | None = None  # default: rl.epsilon_online
    budget: Budget | None = None  # the training budget in decisions, shared by every zone (16.1)
    schedule: EpsilonSchedule | None = None  # the baselines' ε schedule over the budget (18)
    updates: int = 0
    new_samples: int = 0
    losses: list[tuple[int, float, float, float, float, float]] = field(default_factory=list)
    global_version: int = 0  # the FedAvg version it last installed (13.7.5; 0: its own model)

    def __post_init__(self) -> None:
        deterministic()
        rl = self.cfg.rl
        streams = RandomStreams.from_config(self.cfg, self.seed, self.usage)
        self._eps_rng = streams.generator("agent", self.zone_id, "epsilon")
        self.replay = Replay(self.cfg, streams.generator("agent", self.zone_id, "replay"), self.expert)
        self.online = copy.deepcopy(self.network)
        self.target = copy.deepcopy(self.network)
        self.opt = torch.optim.Adam(self.online.parameters(), lr=rl.lr)
        self.chain = OnlineChain(self.cfg.dqfd.n_step, rl.gamma_per_tau, rl.tau)
        self.eps = rl.epsilon_online if self.epsilon is None else self.epsilon
        self._loss_cfg: tuple[float, Config] = (1.0, self.cfg)
        self._fade()
        self._uploaded_at = 0  # new_samples at the last FedAvg upload
        self._installed_at = 0  # updates at the last install
        # 13.7 FedAvg + local fine-tuning: the personal copy that decides, its target and optimizer
        self.personal: torch.nn.Module | None = None
        self.personal_target: torch.nn.Module | None = None
        self._personal_opt: torch.optim.Optimizer | None = None
        self._personal_steps = 0
        self.finetune_steps = 0  # fine-tuning steps at the last install

    # --- acting (11.2) -------------------------------------------------------------------------------------

    def choose(self, ctx: DecisionContext) -> Option | None:
        valid = [v.masked is None for v in ctx.views]
        if not any(valid):
            return None  # 7.3: every option masked, no decision
        if self._eps_rng.random() < self.current_epsilon():
            open_ = [i for i, v in enumerate(valid) if v]
            a = open_[int(self._eps_rng.integers(len(open_)))]
        else:
            with torch.no_grad():
                net = self.online if self.personal is None else self.personal  # 13.7 table: who decides
                q = net(torch.tensor([self.state(ctx)], dtype=torch.float32))
                a = int(masked_argmax(q, torch.tensor([valid]))[0])
        if self.budget is not None:
            self.budget.spend()  # a decision (16.1: the budget counts decisions)
        return OPTIONS[a]

    def state(self, ctx: DecisionContext) -> tuple[float, ...]:
        """What the network sees at a decision: the 48 inputs (an LSTM baseline adds its history)."""
        return ctx.inputs

    def current_epsilon(self) -> float:
        if self.schedule is not None and self.budget is not None:
            return epsilon(self.schedule, self.budget.used, self.budget.total)
        return self.eps

    @property
    def learning(self) -> bool:
        """Learning stops once the shared budget is used up (16.1)."""
        return self.learn and (self.budget is None or not self.budget.exhausted)

    # --- learning (13.5) -----------------------------------------------------------------------------------

    def attach(self, zone: object) -> None:
        """A new episode's zone: its decisions and transitions feed this agent."""
        buf = zone.transitions  # type: ignore[attr-defined]
        buf.on_decision_hook = self._decision
        buf.on_replay_hook = self._transition

    def _decision(self, d: Decision) -> None:
        self.chain.decision((d.task_id, d.attempt))

    def _transition(self, t: Transition) -> None:
        self._add(self.chain.transition(t))

    def end_episode(self) -> None:
        self._add(self.chain.flush())

    def _add(self, samples: Sequence[Sample]) -> None:
        per = self.cfg.rl.updates_per_transition
        for s in samples:
            self.replay.add(s)
            self.new_samples += 1
            if self.learning and self.new_samples * per >= self.updates + 1:
                self.update()

    # --- FedAvg (13.7) --------------------------------------------------------------------------------------

    @property
    def new_experiences(self) -> int:
        """Samples added since the last upload (13.7.2: none, nothing sent)."""
        return self.new_samples - self._uploaded_at

    def mark_uploaded(self) -> None:
        self._uploaded_at = self.new_samples

    def shared_state(self) -> dict[str, torch.Tensor]:
        """The model it uploads: the shared model (13.7 table)."""
        return self.online.state_dict()

    def install(self, weights: dict[str, torch.Tensor], version: int) -> None:
        """13.7.5: install the global model between two decisions and reset the target network. The
        optimizer starts afresh too (engineering choice: its moments belong to the replaced weights)."""
        self.online.load_state_dict(weights)
        self.target.load_state_dict(weights)
        self.opt = torch.optim.Adam(self.online.parameters(), lr=self.cfg.rl.lr)
        self.global_version, self._installed_at = version, self.updates
        if self.cfg.ablation.federated == "finetune":
            self._personalize()

    def _personalize(self) -> None:
        """13.7 FedAvg + local fine-tuning: a personal copy of the new global model, fine-tuned K steps on
        the zone's replay buffer, then trained online with the shared one. Fast mode does not time the
        K steps (QUESTIONS Q51): the copy takes over at once, so decisions never wait."""
        self.personal = copy.deepcopy(self.online)  # the online network holds the new weights now
        self.personal_target = copy.deepcopy(self.online)
        self._personal_opt = torch.optim.Adam(self.personal.parameters(), lr=self.cfg.rl.lr)
        self._personal_steps = 0
        self.finetune_steps = self.cfg.fedavg.finetune_k if len(self.replay) else 0
        for _ in range(self.finetune_steps):
            self._personal_step(self.replay.draw(self.cfg.rl.batch, self.cfg.dqfd.per.beta_final))

    def _personal_step(self, drawn: Drawn) -> None:
        """One step of the personal copy on a drawn batch; the priorities stay the shared model's."""
        assert self.personal is not None and self.personal_target is not None and self._personal_opt
        parts = dqfd_loss(
            self.personal, self.personal_target, Batch.of(drawn.samples, drawn.weights), self.loss_config()
        )
        self._personal_opt.zero_grad()
        parts.total.backward()
        self._personal_opt.step()
        self._personal_steps += 1
        if self._personal_steps % self.cfg.rl.target_update_every == 0:
            self.personal_target.load_state_dict(self.personal.state_dict())

    @property
    def model_version(self) -> tuple[int, int]:
        """5.5: (global version, local update count since it was installed), logged with every decision."""
        return self.global_version, self.updates - self._installed_at

    # --- the expert data online (16.4: kept / shrinking share + fading margin / cleared) ------------------

    def margin_scale(self) -> float:
        """keep and clear: 1. fade: 1 - the share of the training budget used, linearly to 0; a deployed agent
        (no budget) has faded out."""
        if self.cfg.ablation.expert != "fade":
            return 1.0
        if self.budget is None:
            return 0.0
        return max(0.0, 1.0 - self.budget.used / self.budget.total)

    def loss_config(self) -> Config:
        """The config of the loss: the margin weight λ_E x margin_scale() when fading."""
        f = self.margin_scale()
        if f == 1.0:
            return self.cfg
        if self._loss_cfg[0] != f:
            d = self.cfg.dqfd
            self._loss_cfg = (f, replace(self.cfg, dqfd=replace(d, lambda_margin=d.lambda_margin * f)))
        return self._loss_cfg[1]

    def _fade(self) -> None:
        """The expert part's sampling scale follows margin_scale(), rebuilt every 1 % of the budget."""
        f = self.margin_scale()
        if f != self.replay.expert_scale and (abs(f - self.replay.expert_scale) >= 0.01 or f == 0.0):
            self.replay.scale_expert(f)

    def update(self) -> None:
        rl, per = self.cfg.rl, self.cfg.dqfd.per
        self._fade()
        drawn = self.replay.draw(rl.batch, per.beta_final)
        parts = dqfd_loss(
            self.online, self.target, Batch.of(drawn.samples, drawn.weights), self.loss_config()
        )
        self.opt.zero_grad()
        parts.total.backward()
        self.opt.step()
        self.replay.update(drawn.index, parts.td_errors.tolist())
        if self.personal is not None:
            self._personal_step(drawn)  # 13.7: every transition trains both copies
        self.updates += 1
        self.losses.append(
            (self.updates, parts.td, parts.n_step, parts.margin, float(parts.total.detach()), parts.mean_q)
        )
        if self.updates % rl.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())


@dataclass(frozen=True, slots=True)
class Episode:
    """One training episode's numbers (M6.6): on-time share and mean ê of its closed tasks, updates so far."""

    seed: int
    on_time: float
    mean_e_hat: float
    tasks: int
    updates: int
    mean_loss: float


def episode_stats(cfg: Config, recs: Sequence[TaskRecord]) -> tuple[float, float]:
    """16.3: the share of tasks delivered on time, and the mean ê (min(E / E_ref, ē) over every attempt;
    ē for a drop, as in 13.2.4)."""
    if not recs:
        return 0.0, 0.0
    e_bar = cfg.derived.e_bar
    e = [
        min(sum(a.energy for a in r.rewards.values()) / cfg.derived.e_ref[r.task.task_type], e_bar)
        if r.outcome in ("on_time", "late")
        else e_bar
        for r in recs
    ]
    return sum(r.outcome == "on_time" for r in recs) / len(recs), float(np.mean(e))


def train_online(
    cfg: Config,
    agents: dict[str, OnlineAgent],
    scenario: str,
    seeds: Sequence[int],
    seconds: float,
    n_cars: int | None = None,
    fed: FedState | None = None,
) -> list[Episode]:
    """Online episodes in fast mode: the same agents across episodes, the neighbour order shuffled in each;
    with federated learning (13.7) the FedAvg rounds go on across the episodes (one FedState)."""
    from pcnme.experiments.city import City
    from pcnme.fl.fedavg import FedState

    fed = fed if fed is not None else FedState()

    out = []
    for seed in seeds:
        before = {z: a.updates for z, a in agents.items()}
        city = City(
            cfg, scenario, lambda c, s, z: agents[z], seed, n_cars, shuffle_neighbours=True, fed=fed
        ).run(seconds)
        recs = [r for r in city.records if r.task.created < seconds]
        on_time, mean_e = episode_stats(cfg, recs)
        losses = [x[4] for a in agents.values() for x in a.losses if x[0] > before[a.zone_id]]
        out.append(
            Episode(
                seed,
                on_time,
                mean_e,
                len(recs),
                sum(a.updates for a in agents.values()),
                float(np.mean(losses)) if losses else 0.0,
            )
        )
    return out
