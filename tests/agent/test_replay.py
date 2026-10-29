"""The prioritized replay buffer (PLAN M6.3; final-design 13.5, 18): α 0.4, β0 0.6 -> 1, ε_d 1, ε_a 0.001."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pcnme.agent.nstep import Sample
from pcnme.agent.replay import Replay, SumTree, beta_at
from pcnme.config import Config, load_config


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


def small(cfg: Config, capacity: int) -> Config:
    return replace(cfg, rl=replace(cfg.rl, replay_agent=capacity))


def sample(tag: float, expert: bool = False) -> Sample:
    s = (tag,) + (0.0,) * 47
    return Sample(s, 0, tag, (True,) * 6, expert, False, None, None, 0.0, False, 0, 0.0, None, None, 0.0)


def test_the_sum_tree() -> None:
    t = SumTree(5)
    for i, v in enumerate((1.0, 2.0, 3.0, 4.0, 0.0)):
        t.set(i, v)
    assert t.total == 10.0 and t.get(2) == 3.0
    assert [t.find(u) for u in (0.0, 0.99, 1.0, 2.99, 3.0, 5.99, 6.0, 9.99)] == [0, 0, 1, 1, 2, 2, 3, 3]
    t.set(3, 0.0)
    assert t.total == 6.0 and t.find(5.99) == 2


def test_the_expert_part_is_never_overwritten(cfg: Config) -> None:
    r = Replay(small(cfg, 3), np.random.default_rng(0), [sample(-1, True), sample(-2, True)])
    for k in range(10):
        r.add(sample(k))
    assert len(r) == 5 and [x.r for x in r.expert] == [-1, -2]
    assert sorted(x.r for x in r.agent) == [7, 8, 9]  # the ring kept the newest 3
    assert r.is_expert(0) and r.is_expert(1) and not r.is_expert(2)


def test_draws_follow_p_to_the_alpha(cfg: Config) -> None:
    # Two own transitions with |δ| 9 and 0: p = 9.001 and 0.001, so P = p^0.4 / Σ p^0.4.
    r = Replay(small(cfg, 2), np.random.default_rng(1))
    a, b = r.add(sample(0)), r.add(sample(1))
    r.update([a, b], [9.0, 0.0])
    pa, pb = 9.001**0.4, 0.001**0.4
    counts = Counter()
    for _ in range(400):
        counts.update(r.draw(8, 1.0).index)
    share = counts[a] / (counts[a] + counts[b])
    assert share == pytest.approx(pa / (pa + pb), abs=0.03)


def test_priorities_get_the_expert_or_the_agent_bonus(cfg: Config) -> None:
    r = Replay(small(cfg, 4), np.random.default_rng(2), [sample(-1, True)])
    own = r.add(sample(0))
    r.update([0, own], [0.0, 0.0])
    assert r.tree.get(0) == pytest.approx(1.0**0.4)  # ε_d = 1
    assert r.tree.get(own) == pytest.approx(0.001**0.4)  # ε_a = 0.001


def test_new_transitions_get_the_largest_priority_so_far(cfg: Config) -> None:
    r = Replay(small(cfg, 4), np.random.default_rng(3))
    a = r.add(sample(0))
    r.update([a], [4.0])
    b = r.add(sample(1))
    assert r.tree.get(b) == pytest.approx(4.001**0.4) == pytest.approx(r.tree.get(a))


def test_the_importance_weights_by_hand(cfg: Config) -> None:
    # P = (0.75, 0.25) after p^α; N = 2; w = (N P)^-β / max: β = 1: (1/1.5, 1/0.5) / 2 = (1/3, 1).
    r = Replay(small(cfg, 2), np.random.default_rng(4))
    r.tree.set(r.add(sample(0)), 3.0)
    r.tree.set(r.add(sample(1)), 1.0)
    d = r.draw(2, 1.0)  # stratified: one draw in [0, 2), one in [2, 4): slots 0 and 1
    assert d.index == (0, 1) and d.weights == pytest.approx((1 / 3, 1.0))
    assert r.draw(2, 0.0).weights == (1.0, 1.0)  # β = 0: no correction


def test_the_upsample_ratio(cfg: Config) -> None:
    r = Replay(small(cfg, 4), np.random.default_rng(5), [sample(-1, True)])
    for k in range(3):
        r.add(sample(k))
    d = r.draw(8, 1.0)
    share = sum(r.is_expert(i) for i in d.index) / 8
    assert r.upsample_ratio(d) == pytest.approx(share / 0.25)


def test_draws_repeat_with_the_seed(cfg: Config) -> None:
    def run(seed: int) -> tuple[int, ...]:
        r = Replay(small(cfg, 50), np.random.default_rng(seed))
        for k in range(50):
            r.add(sample(k))
        r.update(list(range(50)), [k / 10 for k in range(50)])
        return r.draw(16, 0.6).index

    assert run(7) == run(7) and run(7) != run(8)


def test_beta_is_annealed_to_one(cfg: Config) -> None:
    assert beta_at(cfg, 0, 100) == pytest.approx(0.6)
    assert beta_at(cfg, 50, 100) == pytest.approx(0.8)
    assert beta_at(cfg, 500, 100) == pytest.approx(1.0) == beta_at(cfg, 3, 0)


def test_an_empty_replay_cannot_draw(cfg: Config) -> None:
    with pytest.raises(ValueError):
        Replay(small(cfg, 2), np.random.default_rng(0)).draw(1, 1.0)
