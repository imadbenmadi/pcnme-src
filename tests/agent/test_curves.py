"""Training curves (PLAN M6.6; docs/METRICS.md learning and episode)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pq = pytest.importorskip("pyarrow.parquet")

from pcnme.agent.curves import log_online, log_pretraining, loss_rows, write  # noqa: E402
from pcnme.agent.dqfd import pretrain  # noqa: E402
from pcnme.agent.network import make_network  # noqa: E402
from pcnme.agent.online import OnlineAgent, train_online  # noqa: E402
from pcnme.config import Config  # noqa: E402
from pcnme.metrics.runlog import RunLog  # noqa: E402
from tests.agent.test_dqfd import cfg, toy  # noqa: E402, F401

ROOT = Path(__file__).resolve().parents[2]


def test_loss_rows_are_block_means() -> None:
    losses = [(u, float(u), 2.0 * u, 0.0, 3.0 * u, -float(u)) for u in range(1, 6)]
    rows = loss_rows(losses, 2)
    assert [r["update"] for r in rows] == [2.0, 4.0, 5.0]
    assert rows[0]["td_loss"] == 1.5 and rows[0]["n_step_loss"] == 3.0 and rows[0]["total_loss"] == 4.5
    assert rows[2]["mean_q"] == -5.0  # the short last block


def test_the_curves_are_written_and_plotted(cfg: Config, tmp_path: Path) -> None:  # noqa: F811
    log = RunLog("S2", "pcnme", 2000)
    pre = pretrain(cfg, toy(), np.random.default_rng(2), seed=3, updates=200)
    log_pretraining(log, pre, 50)
    net = make_network(cfg, 4)
    agents = {z: OnlineAgent(cfg, z, net) for z in ("z1", "z2", "z3", "z4")}
    episodes = train_online(cfg, agents, "S2", [2000, 2001], 2.0, n_cars=40)
    log_online(log, agents, episodes, 50)
    d = write(log, tmp_path, {"what": "curves"})
    learning = pq.read_table(d / "learning.parquet").to_pylist()
    pre_rows = [r for r in learning if r["phase"] == "pretrain"]
    agree = sorted(
        (r["update"], r["expert_agreement"]) for r in pre_rows if r["expert_agreement"] is not None
    )
    assert agree == sorted(pre.curve.agreement) and agree[0][0] == 0  # the untrained network's included
    assert {r["zone"] for r in learning if r["phase"] == "online"} <= {"z1", "z2", "z3", "z4"}
    eps = pq.read_table(d / "episodes.parquet").to_pylist()
    fields = ("seed", "on_time", "mean_e_hat", "tasks", "updates", "mean_loss")
    assert [tuple(e[f] for f in fields) for e in eps] == [
        tuple(getattr(e, f) for f in fields) for e in episodes
    ]
    assert (d / "points.lp").read_text(encoding="utf-8").count("learning,") == len(learning)
    out = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "plot_curves.py"), str(d)], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert (d / "curves.png").stat().st_size > 10_000
