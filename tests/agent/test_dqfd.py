"""DQfD pretraining (PLAN M6.4; final-design 13.5, 18). Synthetic demonstrations keep the tests fast: the
expert takes action 0 when input 0 is above 0.5, else action 1, and is rewarded for it."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pcnme.agent.dqfd import agreement, pretrain, split  # noqa: E402
from pcnme.agent.nstep import Sample  # noqa: E402
from pcnme.config import Config, load_config  # noqa: E402
from tests.agent.test_losses import Table  # noqa: E402


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    c = load_config(default_config_path)
    return replace(
        c,
        rl=replace(c.rl, lr=1e-3, batch=32, target_update_every=50),
        dqfd=replace(c.dqfd, eval_every=50, patience=4, holdout_share=0.25),
    )


def toy(n_windows: int = 4, per: int = 150, seed: int = 0) -> list[tuple[tuple[str, int], Sample]]:
    rng = np.random.default_rng(seed)
    out = []
    for w in range(n_windows):
        for _ in range(per):
            x = rng.uniform(0, 1, 48)
            a = 0 if x[0] > 0.5 else 1
            s = Sample(tuple(x), a, 1.0, (True, True) + (False,) * 4, True, False, None, None, 0.0,
                       False, 0, 0.0, None, None, 0.0)  # fmt: skip
            out.append((("S2", w), s))
    return out


def test_the_held_out_split_is_whole_windows(cfg: Config) -> None:
    groups = [("S2", w) for w in range(10) for _ in range(3)]
    held = split(groups, 0.2, np.random.default_rng(1))
    assert len(held) == 2 and held <= set(groups)
    assert split(groups, 0.2, np.random.default_rng(1)) == held
    assert len(split([("S2", 0), ("S2", 1)], 0.2, np.random.default_rng(0))) == 1  # at least one
    assert len(split([("S2", w) for w in range(5)], 0.3, np.random.default_rng(0))) == 2  # round(1.5)
    assert len(split([("S2", 0), ("S2", 1)], 0.5, np.random.default_rng(0))) == 1  # one left to train on
    with pytest.raises(ValueError):
        split([("S2", 0)], 0.2, np.random.default_rng(0))


def test_agreement_by_hand() -> None:
    def smp(q: tuple[float, ...], a: int, expert: bool = True) -> Sample:
        s = q + (0.0,) * (48 - len(q))
        return Sample(s, a, 0.0, (True, True, True) + (False,) * 3, expert, False, None, None, 0.0,
                      False, 0, 0.0, None, None, 0.0)  # fmt: skip

    xs = [smp((1, 0, 0), 0), smp((0, 1, 0), 0), smp((0, 0, 5, 9), 2), smp((5, 0, 0), 1, expert=False)]
    # argmax among the valid actions: 0, 1, 2 (action 3 = 9 is masked): 2 of the 3 expert rows agree
    assert agreement(Table(), xs) == pytest.approx(2 / 3)
    assert agreement(Table(), [xs[3]]) == 0.0


def test_pretraining_raises_the_held_out_expert_agreement(cfg: Config) -> None:
    out = pretrain(cfg, toy(), np.random.default_rng(2), seed=3, updates=600)
    (u0, untrained), *_ = out.curve.agreement
    assert u0 == 0 and untrained < 0.7  # about chance (2 valid actions)
    assert out.best_agreement >= 0.9
    assert out.updates <= 600 and out.curve.losses[0][0] == 1
    assert agreement(out.network, [s for _, s in toy(seed=9)]) == pytest.approx(out.best_agreement, abs=0.1)


def test_it_stops_early_when_the_agreement_stops_rising(cfg: Config) -> None:
    flat = replace(cfg, rl=replace(cfg.rl, lr=0.0))  # nothing learns: the agreement never rises
    out = pretrain(flat, toy(), np.random.default_rng(2), seed=3, updates=10_000)
    assert out.stopped_early and out.updates == 50 * (1 + cfg.dqfd.patience)


def test_pretraining_repeats_with_the_seed(cfg: Config) -> None:
    a = pretrain(cfg, toy(), np.random.default_rng(5), seed=6, updates=100)
    b = pretrain(cfg, toy(), np.random.default_rng(5), seed=6, updates=100)
    assert a.curve.losses == b.curve.losses and a.curve.agreement == b.curve.agreement
    assert all(torch.equal(x, y) for x, y in zip(a.network.parameters(), b.network.parameters(), strict=True))


def test_the_best_network_is_kept(cfg: Config) -> None:
    out = pretrain(cfg, toy(), np.random.default_rng(2), seed=3, updates=600)
    held = split([g for g, _ in toy()], cfg.dqfd.holdout_share, np.random.default_rng(2))
    test = [s for g, s in toy() if g in held]
    assert agreement(out.network, test) == pytest.approx(out.best_agreement)


def test_no_held_out_transition_is_trained_on_and_priorities_move(cfg: Config) -> None:
    # with a 1-step target (g1 = 0: y = r), the TD errors are not 0, so the priorities change
    data = [(g, replace(x, has1=True, s1=x.s, valid1=x.valid, g1=0.0)) for g, x in toy()]
    out = pretrain(cfg, data, np.random.default_rng(2), seed=3, updates=100)
    assert len(out.replay.expert) == sum(g not in out.held_out for g, _ in data) == 450
    leaves = [out.replay.tree.get(i) for i in range(len(out.replay.expert))]
    assert len(set(leaves)) > 10  # updated from the TD errors, not all at the initial 1
