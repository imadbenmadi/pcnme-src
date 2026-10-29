"""Tang & Wong and QECO in the agent's seat (final-design 16.1, 18 RL rows; PLAN M7). THIS TOUCHES THE AGENT:
the same 48 inputs, 6 actions, masks, reward and time-based discount as PCNME.

Both are a dueling double DQN with an LSTM over the anchor's last H = 10 decision inputs (zero-padded at the
start of an episode). On PCNME's inputs and reward their papers' designs nearly coincide (QUESTIONS Q44
stand-in): Tang & Wong feed the LSTM's last output to the dueling heads; QECO joins the LSTM's last output
with the current 48 inputs and passes them through a hidden layer first. Both learn as in their papers from
random play, without pretraining or expert data: 1-step double-DQN targets with the time-based discount,
uniform replay of rl.replay_agent own transitions, one update per 4 transitions, a hard target copy every
1 000 updates, the baselines' ε schedule over the shared budget (18). Engineering choices: LSTM width 256
(the hidden width of 18); the replay is the prioritized one with α = 0 and β = 0 (uniform, weights 1).
"""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import torch
from torch import Tensor, nn

from pcnme.agent.budget import Budget
from pcnme.agent.inputs import N_INPUTS, Decision
from pcnme.agent.network import N_ACTIONS
from pcnme.agent.nstep import Sample, discount
from pcnme.agent.online import OnlineAgent, train_online
from pcnme.agent.transitions import Transition
from pcnme.config import Config
from pcnme.methods.learners import LSTM_FAMILY, Trained, _seed, training_seeds
from pcnme.zone.decision import DecisionContext

Key = tuple[object, int]


class LstmQNet(nn.Module):
    """(B, H x 48) -> LSTM over the H steps -> dueling heads. variant: tang-wong or qeco (Q44)."""

    def __init__(self, variant: str, history: int, hidden: int = 256, n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        if variant not in LSTM_FAMILY:
            raise ValueError(f"unknown LSTM baseline {variant}")
        self.variant, self.history = variant, history
        self.lstm = nn.LSTM(N_INPUTS, hidden, batch_first=True)
        width = hidden
        if variant == "qeco":
            self.mix = nn.Sequential(nn.Linear(hidden + N_INPUTS, hidden), nn.ReLU())
        self.advantage = nn.Linear(width, n_actions)
        self.value = nn.Linear(width, 1)

    def forward(self, x: Tensor) -> Tensor:
        seq = x.view(x.shape[0], self.history, N_INPUTS)
        out, _ = self.lstm(seq)
        z = out[:, -1, :]
        if self.variant == "qeco":
            z = self.mix(torch.cat([z, seq[:, -1, :]], dim=1))
        a = self.advantage(z)
        return self.value(z) + a - a.mean(dim=-1, keepdim=True)


def make_lstm(cfg: Config, variant: str, seed: int) -> LstmQNet:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return LstmQNet(variant, cfg.rl.lstm_history_h, cfg.rl.hidden[0])


def plain_dqn(cfg: Config) -> Config:
    """1-step double DQN with uniform replay: no n-step, no margin, α = 0, β = 0."""
    d = cfg.dqfd
    per = replace(d.per, alpha=0.0, beta0=0.0, beta_final=0.0)
    return replace(cfg, dqfd=replace(d, lambda_n=0.0, lambda_margin=0.0, per=per))


@dataclass
class LstmAgent(OnlineAgent):
    """One zone's LSTM baseline: its state is the anchor's last H decision inputs."""

    histories: dict[Key, tuple[float, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cfg = plain_dqn(self.cfg)
        super().__post_init__()
        self._recent: deque[tuple[float, ...]] = deque(maxlen=self.cfg.rl.lstm_history_h)

    def _window(self, current: tuple[float, ...]) -> tuple[float, ...]:
        h = self.cfg.rl.lstm_history_h
        steps = [*self._recent, current][-h:]
        pad = [(0.0,) * N_INPUTS] * (h - len(steps))
        return tuple(x for step in (*pad, *steps) for x in step)

    def state(self, ctx: DecisionContext) -> tuple[float, ...]:
        return self._window(ctx.inputs)

    def attach(self, zone: object) -> None:
        super().attach(zone)
        self._recent.clear()  # a new episode: the history starts empty
        self.histories.clear()

    def _decision(self, d: Decision) -> None:
        self.histories[(d.task_id, d.attempt)] = self._window(d.inputs)
        self._recent.append(d.inputs)

    def _transition(self, t: Transition) -> None:
        key = (t.task_id, t.attempt)
        s = self.histories.pop(key, None)
        if s is None:
            return
        s1 = self.histories.get(t.next_key) if t.next_key is not None else None
        has1 = s1 is not None and t.dt is not None and t.valid_next is not None
        rl = self.cfg.rl
        g1 = discount(t.dt, rl.gamma_per_tau, rl.tau) if has1 and t.dt is not None else 0.0
        smp = Sample(s, t.a, t.r, t.valid, False, has1, s1, t.valid_next, g1, False, 0, 0.0, None, None, 0.0)
        self._add([smp])

    def end_episode(self) -> None:
        self.histories.clear()


def train_lstm(
    cfg: Config,
    method: str,
    scenario: str,
    zones: Sequence[str],
    budget_decisions: int,
    *,
    seconds: float | None = None,
    seed: int | None = None,
    n_cars: int | None = None,
) -> Trained:
    """Tang & Wong or QECO from random play on the shared budget (16.1)."""
    if method not in LSTM_FAMILY:
        raise ValueError(f"{method} is not an LSTM baseline")
    seeds = iter(training_seeds(cfg, seed))
    first = next(iter(training_seeds(cfg, seed)))
    init, _ = _seed(cfg, method, "init", first)
    net = make_lstm(cfg, method, init)
    budget = Budget(budget_decisions)
    sched = cfg.rl.epsilon_baselines
    agents = {z: LstmAgent(cfg, z, net, (), seed=first, budget=budget, schedule=sched) for z in zones}
    out = Trained(method, {}, (), budget, schedule=sched)
    length = cfg.fast_mode.episode_length if seconds is None else seconds
    while not budget.exhausted:
        (ep,) = train_online(cfg, agents, scenario, [next(seeds)], length, n_cars)  # type: ignore[arg-type]
        out.episodes.append(ep)
    out.states = {z: copy.deepcopy(a.online.state_dict()) for z, a in agents.items()}
    return out


def deployed_lstm(cfg: Config, trained: Trained, zones: Sequence[str], seed: int) -> dict[str, LstmAgent]:
    """Each zone from its trained weights, learning online with ε online (Q42)."""
    out = {}
    for z in zones:
        net = make_lstm(cfg, trained.method, 0)
        net.load_state_dict(trained.states[z])
        out[z] = LstmAgent(cfg, z, net, (), seed=seed, usage="evaluation")
    return out
