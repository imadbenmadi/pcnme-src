"""The DQfD loss (final-design 13.5, 11.3.7; PLAN M6.2). THIS TOUCHES THE AGENT.

    loss = J_DQ + λ1 J_n + λ2 J_E + λ3 L2

J_DQ: the 1-step double-DQN loss, y = r + γ^(Δt/τ) Q_target(s', argmax_{a' valid in s'} Q(s', a')). J_n: the
same with the n-step target. J_E: the large-margin loss on expert transitions only, max_{a valid} [Q(s, a) +
l(a_E, a)] - Q(s, a_E), l = margin for a != a_E, 0 for a_E. L2: the sum of the squared weights (weight decay).
Rows without s' (Q36) or without an n-step target add nothing to J_DQ or J_n.

Engineering choices: each term is the mean over the rows it applies to (0 when none); the prioritized-replay
importance weights multiply every per-row term (J_DQ, J_n and J_E); L2 covers every parameter; the TD error
for the priorities is the 1-step one, else the n-step one, else 0; MSE, or Huber when the config says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pcnme.agent.network import masked_argmax
from pcnme.agent.nstep import Sample
from pcnme.config import Config


@dataclass(frozen=True, slots=True)
class Batch:
    s: Tensor  # (B, 48)
    a: Tensor  # (B,) long
    r: Tensor
    valid: Tensor  # (B, 6) bool
    expert: Tensor  # (B,) bool
    has1: Tensor
    s1: Tensor
    valid1: Tensor
    g1: Tensor
    hasn: Tensor
    rn: Tensor
    sn: Tensor
    validn: Tensor
    gn: Tensor
    weights: Tensor  # importance weights (1 without prioritized replay)

    @classmethod
    def of(
        cls, samples: Sequence[Sample], weights: Sequence[float] | None = None, device: str = "cpu"
    ) -> Batch:
        zero_s = (0.0,) * len(samples[0].s)
        some = (True,) + (False,) * (len(samples[0].valid) - 1)  # a placeholder for rows without s'

        def f(xs: Sequence[object], dtype: torch.dtype = torch.float32) -> Tensor:
            return torch.tensor(xs, dtype=dtype, device=device)

        return cls(
            s=f([x.s for x in samples]),
            a=f([x.a for x in samples], torch.long),
            r=f([x.r for x in samples]),
            valid=f([x.valid for x in samples], torch.bool),
            expert=f([x.expert for x in samples], torch.bool),
            has1=f([x.has1 for x in samples], torch.bool),
            s1=f([x.s1 if x.s1 is not None else zero_s for x in samples]),
            valid1=f([x.valid1 if x.valid1 is not None and x.has1 else some for x in samples], torch.bool),
            g1=f([x.g1 for x in samples]),
            hasn=f([x.hasn for x in samples], torch.bool),
            rn=f([x.rn for x in samples]),
            sn=f([x.sn if x.sn is not None else zero_s for x in samples]),
            validn=f([x.validn if x.validn is not None and x.hasn else some for x in samples], torch.bool),
            gn=f([x.gn for x in samples]),
            weights=f(list(weights) if weights is not None else [1.0] * len(samples)),
        )


@dataclass(frozen=True, slots=True)
class LossParts:
    total: Tensor
    td: float
    n_step: float
    margin: float
    l2: float
    td_errors: Tensor  # (B,) for the priorities


def double_target(online: nn.Module, target: nn.Module, s1: Tensor, valid1: Tensor) -> Tensor:
    """11.3.7: Q_target(s', argmax over the actions valid in s' of Q_online(s', a'))."""
    with torch.no_grad():
        a_star = masked_argmax(online(s1), valid1)
        return target(s1).gather(1, a_star.unsqueeze(1)).squeeze(1)


def _mean(x: Tensor, rows: Tensor) -> Tensor:
    return x[rows].mean() if bool(rows.any()) else x.sum() * 0.0


def margin_loss(q: Tensor, a: Tensor, valid: Tensor, margin: float) -> Tensor:
    """13.5: max over valid a of [Q(s, a) + l(a_E, a)] - Q(s, a_E), per row."""
    gap = torch.full_like(q, margin)  # l(a_E, a)
    gap.scatter_(1, a.unsqueeze(1), 0.0)
    best = (q + gap).masked_fill(~valid, float("-inf")).max(dim=1).values
    return best - q.gather(1, a.unsqueeze(1)).squeeze(1)


def dqfd_loss(online: nn.Module, target: nn.Module, b: Batch, cfg: Config) -> LossParts:
    d = cfg.dqfd
    q = online(b.s)
    q_sa = q.gather(1, b.a.unsqueeze(1)).squeeze(1)
    y1 = b.r + b.g1 * double_target(online, target, b.s1, b.valid1)
    yn = b.rn + b.gn * double_target(online, target, b.sn, b.validn)
    err = nn.functional.huber_loss if cfg.rl.loss == "huber" else nn.functional.mse_loss
    j_dq = _mean(b.weights * err(q_sa, y1, reduction="none"), b.has1)
    j_n = _mean(b.weights * err(q_sa, yn, reduction="none"), b.hasn)
    j_e = _mean(b.weights * margin_loss(q, b.a, b.valid, d.margin), b.expert)
    l2 = torch.stack([(p * p).sum() for p in online.parameters()]).sum()
    total = j_dq + d.lambda_n * j_n + d.lambda_margin * j_e + d.lambda_l2 * l2
    with torch.no_grad():
        td = torch.where(b.has1, y1 - q_sa, torch.where(b.hasn, yn - q_sa, torch.zeros_like(q_sa))).abs()
    td_v, n_v, e_v, l2_v = (float(x.detach()) for x in (j_dq, j_n, j_e, l2))
    return LossParts(total, td_v, n_v, e_v, l2_v, td)
