"""The replay buffer (final-design 13.5, 18 RL and Pretraining rows; PLAN M6.3). THIS TOUCHES THE AGENT.

Two parts: the agent's own transitions (a ring of rl.replay_agent = 100 000, the oldest overwritten) and the
expert transitions, kept for good in a part that is never overwritten. Prioritized replay over both (Schaul
et al.): priority p = |δ| + ε, with the bonus ε_d on expert and ε_a on own transitions; a transition is drawn
with probability P(i) = p_i^α / Σ p^α, and its loss weighted by w_i = (N P(i))^(-β) / max_batch w, β annealed
from β0 to 1. Engineering choices: a new transition (and every expert one at load) gets the largest priority
so far (1 at the start), so it is drawn at least once soon; the batch is drawn stratified (one draw per equal
slice of the total); the weights are normalised by the batch's largest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pcnme.agent.nstep import Sample
from pcnme.config import Config


class SumTree:
    """A binary sum tree over a fixed number of leaves: set a leaf, read the total, find by prefix sum."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.leaf0 = 1
        while self.leaf0 < size:
            self.leaf0 *= 2
        self.tree = np.zeros(2 * self.leaf0)

    def set(self, i: int, value: float) -> None:
        j = self.leaf0 + i
        self.tree[j] = value
        j //= 2
        while j >= 1:
            self.tree[j] = self.tree[2 * j] + self.tree[2 * j + 1]
            j //= 2

    def get(self, i: int) -> float:
        return float(self.tree[self.leaf0 + i])

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def find(self, u: float) -> int:
        """The leaf whose prefix-sum slice holds u (0 <= u < total)."""
        j = 1
        while j < self.leaf0:
            left = self.tree[2 * j]
            if u < left:
                j = 2 * j
            else:
                u -= left
                j = 2 * j + 1
        return min(j - self.leaf0, self.size - 1)


@dataclass(frozen=True, slots=True)
class Drawn:
    index: tuple[int, ...]  # buffer slots (expert first, then the agent ring)
    samples: tuple[Sample, ...]
    weights: tuple[float, ...]


class Replay:
    def __init__(self, cfg: Config, rng: np.random.Generator, expert: Sequence[Sample] = ()) -> None:
        per = cfg.dqfd.per
        self.alpha, self.eps_demo, self.eps_agent = per.alpha, per.eps_demo, per.eps_agent
        self.expert = tuple(expert)
        self.capacity = cfg.rl.replay_agent
        self.agent: list[Sample] = []
        self._next = 0  # the ring's next slot
        self.tree = SumTree(len(self.expert) + self.capacity)
        self.max_p = 1.0
        self.rng = rng
        for i in range(len(self.expert)):
            self.tree.set(i, self.max_p**self.alpha)

    def __len__(self) -> int:
        return len(self.expert) + len(self.agent)

    def add(self, s: Sample) -> int:
        """An own transition into the ring (the oldest one goes when it is full). The slot it took."""
        slot = len(self.expert) + self._next
        if len(self.agent) < self.capacity:
            self.agent.append(s)
        else:
            self.agent[self._next] = s
        self._next = (self._next + 1) % self.capacity
        self.tree.set(slot, self.max_p**self.alpha)
        return slot

    def at(self, slot: int) -> Sample:
        e = len(self.expert)
        return self.expert[slot] if slot < e else self.agent[slot - e]

    def is_expert(self, slot: int) -> bool:
        return slot < len(self.expert)

    def draw(self, batch: int, beta: float) -> Drawn:
        """A stratified prioritized batch and its importance weights."""
        n = len(self)
        if n == 0:
            raise ValueError("an empty replay")
        total = self.tree.total
        edges = np.linspace(0.0, total, batch + 1)
        u = self.rng.uniform(edges[:-1], edges[1:])
        slots = [self.tree.find(float(x)) for x in np.minimum(u, np.nextafter(total, 0.0))]
        p = np.array([self.tree.get(i) for i in slots]) / total
        w = (n * p) ** (-beta)
        w = w / w.max()
        return Drawn(tuple(slots), tuple(self.at(i) for i in slots), tuple(float(x) for x in w))

    def update(self, slots: Sequence[int], td_errors: Sequence[float]) -> None:
        """p = |δ| + ε (ε_d for expert, ε_a for own transitions), stored as p^α."""
        for i, d in zip(slots, td_errors, strict=True):
            p = abs(float(d)) + (self.eps_demo if self.is_expert(i) else self.eps_agent)
            self.max_p = max(self.max_p, p)
            self.tree.set(i, p**self.alpha)

    def upsample_ratio(self, drawn: Drawn) -> float:
        """16.3: how often expert transitions are drawn against uniform sampling (DQfD's Fig. 1)."""
        share = len(self.expert) / len(self)
        if share == 0:
            return 0.0
        return sum(self.is_expert(i) for i in drawn.index) / len(drawn.index) / share


def beta_at(cfg: Config, update: int, updates: int) -> float:
    """β annealed linearly from β0 to β_final over `updates` gradient updates (18)."""
    per = cfg.dqfd.per
    f = min(1.0, update / updates) if updates > 0 else 1.0
    return per.beta0 + f * (per.beta_final - per.beta0)
