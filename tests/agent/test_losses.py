"""The n-step samples and the DQfD loss (PLAN M6.2, M6 acceptance; final-design 11.3.6-11.3.7, 13.5; Q36).

A stub network whose Q-values are the first 6 inputs makes every target computable by hand. γ = 0.9 per
τ = 100 ms (18).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pcnme.agent.losses import Batch, double_target, dqfd_loss, margin_loss  # noqa: E402
from pcnme.agent.nstep import build, discount  # noqa: E402
from pcnme.agent.transitions import Transition  # noqa: E402
from pcnme.config import Config, load_config  # noqa: E402

G = 0.9
ALL = (True,) * 6


class Table(torch.nn.Module):
    """Q(s, ·) = the first 6 numbers of s; one dummy parameter so L2 has something to sum."""

    def __init__(self, w: float = 0.0) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([w]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :6] + 0.0 * self.w


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


def state(*q: float) -> tuple[float, ...]:
    return tuple(q) + (0.0,) * (48 - len(q))


def chain() -> list[Transition]:
    # One anchor's 4 decisions at 0, 0.1, 0.25 and 0.3 s; rewards 1, 2, 3, 4; the last has no s' (Q36).
    times, rewards = (0.0, 0.1, 0.25, 0.3), (1.0, 2.0, 3.0, 4.0)
    s = [state(float(k), 0, 0, 0, 0, 0) for k in range(4)]
    out = []
    for k in range(4):
        last = k == 3
        out.append(
            Transition(
                anchor="z1",
                task_id=f"T{k}",
                attempt=1,
                time=times[k],
                s=s[k],
                a=0,
                valid=ALL,
                r=rewards[k],
                s_next=None if last else s[k + 1],
                valid_next=None if last else ALL,
                dt=None if last else times[k + 1] - times[k],
                next_key=None if last else (f"T{k + 1}", 1),
                truncated=last,
            )
        )
    return out


# --- the discount and the n-step samples by hand ------------------------------------------------------------


def test_the_time_based_discount() -> None:
    assert discount(0.1, G, 0.1) == pytest.approx(0.9)
    assert discount(0.05, G, 0.1) == pytest.approx(0.9**0.5)
    assert discount(0.0, G, 0.1) == 1.0


def test_n_step_samples_by_hand() -> None:
    s0, s1, s2, s3 = build(chain(), 2, G, 0.1)
    # sample 0: r0 + γ^(0.1/0.1) r1 = 1 + 0.9 x 2 = 2.8; bootstraps from s2 at t = 0.25: γ^2.5
    assert (s0.hasn, s0.m) == (True, 2) and s0.rn == pytest.approx(2.8) and s0.gn == pytest.approx(0.9**2.5)
    assert s0.sn == chain()[2].s and (s0.has1, s0.g1) == (True, pytest.approx(0.9))
    # sample 1: 2 + γ^1.5 x 3, bootstrap from s3 at 0.3: γ^2
    assert (
        s1.rn == pytest.approx(2 + 0.9**1.5 * 3)
        and s1.gn == pytest.approx(0.9**2.0)
        and s1.sn == chain()[3].s
    )
    # sample 2: the next one (the last) has no s', so the chain stops after 1 step: r2, bootstrap γ^0.5
    # from s3
    assert (s2.m, s2.rn) == (1, 3.0) and s2.gn == pytest.approx(0.9**0.5)
    # sample 3: no s' at all: out of both bootstrapped targets (Q36)
    assert (s3.has1, s3.hasn, s3.m, s3.s1, s3.sn) == (False, False, 0, None, None)


def test_a_broken_chain_is_refused() -> None:
    c = chain()
    with pytest.raises(ValueError, match="chain"):
        build([c[0], c[2]], 2, G, 0.1)


# --- targets ------------------------------------------------------------------------------------------------


def test_the_double_target_uses_the_online_argmax_among_valid_actions() -> None:
    online, target = Table(), Table()
    s1 = torch.tensor([state(1, 9, 3, 0, 0, 0)])
    valid = torch.tensor([[True, False, True, False, False, False]])  # 9 is masked
    # online argmax among valid: action 2 (3 > 1); the target's value there: 3
    assert double_target(online, target, s1, valid).tolist() == [3.0]


def test_a_hand_computed_n_step_target(cfg: Config) -> None:
    # PLAN M6 acceptance. Sample 0 of the chain, n = 2: y_n = 2.8 + γ^2.5 x Q_target(s2, a*); Q(s2) =
    # (2, 0, ...):
    # a* = 0, Q_target = 2: y_n = 2.8 + 0.9^2.5 x 2. Q(s0, a=0) = 0, so J_n = y_n^2 (MSE, weight 1).
    samples = build(chain(), 2, G, 0.1)
    b = Batch.of([samples[0]])
    zero = replace(cfg, dqfd=replace(cfg.dqfd, lambda_margin=0.0, lambda_l2=0.0))
    parts = dqfd_loss(Table(), Table(), b, zero)
    y_n = 2.8 + 0.9**2.5 * 2.0
    y_1 = 1.0 + 0.9 * 1.0  # r0 + γ x Q_target(s1, 0) = 1 + 0.9 x 1
    assert parts.n_step == pytest.approx(y_n**2) and parts.td == pytest.approx(y_1**2)
    assert float(parts.total.detach()) == pytest.approx(y_1**2 + cfg.dqfd.lambda_n * y_n**2)


def test_rows_without_s_prime_add_nothing_to_the_bootstrapped_terms(cfg: Config) -> None:
    samples = build(chain(), 2, G, 0.1)
    full = dqfd_loss(Table(), Table(), Batch.of(samples), cfg)
    without = dqfd_loss(Table(), Table(), Batch.of(samples[:3]), cfg)
    assert full.td == pytest.approx(without.td) and full.n_step == pytest.approx(without.n_step)
    assert float(full.td_errors[3]) == 0.0


# --- the margin loss (expert transitions only) --------------------------------------------------------------


def test_the_margin_loss_by_hand() -> None:
    q = torch.tensor([[1.0, 2.0, 0.5, 0.0, 0.0, 0.0]])
    a = torch.tensor([0])
    every = torch.tensor([[True, True, True, False, False, False]])
    # max(1 + 0, 2 + 0.8, 0.5 + 0.8) - 1 = 1.8
    assert margin_loss(q, a, every, 0.8).tolist() == [pytest.approx(1.8)]
    no_1 = torch.tensor([[True, False, True, False, False, False]])  # action 1 masked: max(1, 1.3) - 1
    assert margin_loss(q, a, no_1, 0.8).tolist() == [pytest.approx(0.3)]
    assert margin_loss(torch.tensor([[3.0, 2.0, 0, 0, 0, 0]]), a, every, 0.8).tolist() == [0.0]


def test_the_margin_loss_is_zero_on_non_expert_transitions(cfg: Config) -> None:
    # PLAN M6 acceptance.
    samples = [replace(x, expert=False) for x in build(chain(), 2, G, 0.1)]
    samples = [replace(x, s=state(0, 5, 5, 5, 5, 5)) for x in samples]  # a = 0 is far below the others
    assert dqfd_loss(Table(), Table(), Batch.of(samples), cfg).margin == 0.0
    flagged = [replace(samples[0], expert=True), *samples[1:]]
    assert dqfd_loss(Table(), Table(), Batch.of(flagged), cfg).margin == pytest.approx(5.8)


# --- the total, the weights and L2 --------------------------------------------------------------------------


def test_the_total_combines_the_terms_with_the_config_lambdas(cfg: Config) -> None:
    samples = [replace(x, expert=True) for x in build(chain(), 2, G, 0.1)]
    p = dqfd_loss(Table(2.0), Table(), Batch.of(samples), cfg)
    d = cfg.dqfd
    assert p.l2 == pytest.approx(4.0)
    expected = p.td + d.lambda_n * p.n_step + d.lambda_margin * p.margin + d.lambda_l2 * p.l2
    assert float(p.total.detach()) == pytest.approx(expected)


def test_importance_weights_scale_the_per_row_terms(cfg: Config) -> None:
    samples = build(chain(), 2, G, 0.1)[:1]
    one = dqfd_loss(Table(), Table(), Batch.of(samples, [1.0]), cfg)
    half = dqfd_loss(Table(), Table(), Batch.of(samples, [0.5]), cfg)
    assert half.td == pytest.approx(one.td / 2) and half.n_step == pytest.approx(one.n_step / 2)


def test_huber_when_the_config_says_so(cfg: Config) -> None:
    samples = build(chain(), 2, G, 0.1)[:1]
    huber = replace(cfg, rl=replace(cfg.rl, loss="huber"))
    y_1 = 1.9  # see the hand-computed test: the error is 1.9 > 1, Huber = 1.9 - 0.5
    assert dqfd_loss(Table(), Table(), Batch.of(samples), huber).td == pytest.approx(y_1 - 0.5)


def test_the_td_error_for_the_priorities(cfg: Config) -> None:
    samples = build(chain(), 2, G, 0.1)
    p = dqfd_loss(Table(), Table(), Batch.of(samples), cfg)
    assert float(p.td_errors[0]) == pytest.approx(1.9)  # |1 + 0.9 x 1 - 0|


def test_the_lambdas_are_read_from_the_config(cfg: Config) -> None:
    samples = [replace(x, expert=True) for x in build(chain(), 2, G, 0.1)]
    other = replace(cfg, dqfd=replace(cfg.dqfd, lambda_n=2.0, lambda_margin=3.0, lambda_l2=0.5))
    p = dqfd_loss(Table(1.0), Table(), Batch.of(samples), other)
    assert float(p.total.detach()) == pytest.approx(p.td + 2.0 * p.n_step + 3.0 * p.margin + 0.5 * p.l2)
