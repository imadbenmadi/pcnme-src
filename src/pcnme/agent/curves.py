"""Training curves (PLAN M6.6; final-design 16.3; docs/METRICS.md `learning`). THIS TOUCHES THE AGENT only in
what is logged.

Two tables, written like every run's logs (metrics/runlog.py: Parquet for the numbers, InfluxDB line protocol
for Grafana): `learning`, one row per zone and phase every `every` updates (the means of td_loss, n_step_loss,
margin_loss, total_loss and mean_q over those updates; epsilon; in pretraining the held-out expert agreement
and the demonstration up-sample ratio when measured), and `episodes`, one row per online episode (on-time
share and mean ê, 16.3; updates so far). tools/plot_curves.py draws them.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pcnme.agent.dqfd import Pretrained
from pcnme.agent.online import Episode, OnlineAgent
from pcnme.metrics.runlog import RunLog

Loss = tuple[int, float, float, float, float, float]  # update, J_DQ, J_n, J_E, total, mean Q


def loss_rows(losses: Sequence[Loss], every: int) -> list[dict[str, float]]:
    """The means over each block of `every` updates (the last block may be shorter)."""
    out = []
    for start in range(0, len(losses), every):
        block = np.array([x[1:] for x in losses[start : start + every]])
        td, n, e, total, q = block.mean(axis=0)
        out.append(
            {
                "update": float(losses[min(start + every, len(losses)) - 1][0]),
                "td_loss": float(td),
                "n_step_loss": float(n),
                "margin_loss": float(e),
                "total_loss": float(total),
                "mean_q": float(q),
            }
        )
    return out


def log_pretraining(log: RunLog, pre: Pretrained, every: int) -> None:
    agree = dict(pre.curve.agreement)
    upsample = dict(pre.curve.upsample)
    for row in loss_rows(pre.curve.losses, every):
        u = int(row["update"])
        log.add("learning", float(u), zone="global", phase="pretrain", **row,
                expert_agreement=agree.get(u), demo_upsample_ratio=upsample.get(u), epsilon=0.0)  # fmt: skip
    if 0 in agree:  # the untrained network's agreement
        log.add("learning", 0.0, zone="global", phase="pretrain", update=0.0, expert_agreement=agree[0])


def log_online(log: RunLog, agents: dict[str, OnlineAgent], episodes: Sequence[Episode], every: int) -> None:
    for z, a in agents.items():
        for row in loss_rows(a.losses, every):
            log.add("learning", row["update"], zone=z, phase="online", **row, epsilon=a.eps)
    for k, e in enumerate(episodes):
        log.add("episodes", float(k), episode=k, seed=e.seed, on_time=e.on_time, mean_e_hat=e.mean_e_hat,
                tasks=e.tasks, updates=e.updates, mean_loss=e.mean_loss)  # fmt: skip


def write(log: RunLog, out_dir: Path, info: dict[str, object]) -> Path:
    """results/<run_id>/: learning.parquet, episodes.parquet, run.json, points.lp (METRICS.md)."""
    return log.write(out_dir, info)
