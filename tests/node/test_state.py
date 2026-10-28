"""Published node state and the anchor's wait estimate (PLAN M1.8; final-design 8.5; QUESTIONS Q8, Q9)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pcnme.node.state import Pending, SlotState, wait_estimate
from pcnme.node.timetable import Exponential
from tests.node.helpers import row, tiny

if TYPE_CHECKING:
    from tests.node.conftest import Node

LONG = 1e6


# --- W1: what a node publishes (8.5) ------------------------------------------------------------------------


def test_the_snapshot_holds_the_8_5_fields_per_slot_type(make_node: Callable[..., Node]) -> None:
    # The config's F-GPU node: 2 T1 run on the GPU, then a queued soft T4 and two queued hard T1.
    node = make_node()
    for task in ("T1", "T1", "T4", "T1", "T1"):
        node.submit(task, LONG)
    node.run(0.002)
    snap = node.executor.snapshot()
    assert set(snap) == {"gpu", "cpu"}
    gpu = snap["gpu"]
    assert (gpu.c, gpu.capacity, gpu.n_hard, gpu.n_soft, gpu.u) == (2, 32, 2, 1, 2)
    assert gpu.b_hard == pytest.approx(0.016) and gpu.b_soft == pytest.approx(0.025)
    assert gpu.r == pytest.approx(2 * (0.008 - 0.002))  # remaining part of the two running means
    assert snap["cpu"] == SlotState(c=2, capacity=16, n_hard=0, n_soft=0, b_hard=0.0, b_soft=0.0, r=0.0, u=0)


def test_a_node_without_a_gpu_publishes_only_its_cpu(make_node: Callable[..., Node]) -> None:
    assert set(make_node("F-CPU").executor.snapshot()) == {"cpu"}


# --- W2, W3: the estimate -----------------------------------------------------------------------------------

BUSY = SlotState(c=2, capacity=32, n_hard=1, n_soft=1, b_hard=0.008, b_soft=0.025, r=0.012, u=2)


def test_the_wait_is_zero_while_a_slot_is_free() -> None:
    free = SlotState(c=2, capacity=32, n_hard=0, n_soft=0, b_hard=0.0, b_soft=0.0, r=0.006, u=1)
    assert wait_estimate(free, "hard") == 0.0
    assert wait_estimate(free, "soft") == 0.0


def test_the_pooled_formula_otherwise() -> None:
    # 8.5: Ŵ_hard = (R + B_hard) / c, Ŵ_soft = (R + B_hard + B_soft) / c
    assert wait_estimate(BUSY, "hard") == pytest.approx((0.012 + 0.008) / 2)
    assert wait_estimate(BUSY, "soft") == pytest.approx((0.012 + 0.008 + 0.025) / 2)


def test_pending_sends_take_the_free_slot_and_join_the_backlog_of_their_class() -> None:
    # One slot free (u = 1 of 2), but the anchor has one pending hard send (8 ms) to this q: u + pending = c,
    # so the estimate is the formula with that send in B_hard (R1, 9.2).
    one_free = SlotState(c=2, capacity=32, n_hard=0, n_soft=0, b_hard=0.0, b_soft=0.0, r=0.006, u=1)
    pending = [Pending(cls="hard", mean=0.008)]
    assert wait_estimate(one_free, "hard", pending) == pytest.approx((0.006 + 0.008) / 2)
    # A pending soft send counts in B_soft: only soft estimates see it.
    soft_pending = [Pending(cls="soft", mean=0.025), Pending(cls="hard", mean=0.008)]
    assert wait_estimate(BUSY, "hard", soft_pending) == pytest.approx((0.012 + 0.008 + 0.008) / 2)
    assert wait_estimate(BUSY, "soft", soft_pending) == pytest.approx((0.012 + 0.016 + 0.050) / 2)


# --- W5: one formula for the executor and the anchor (Q8) ---------------------------------------------------


def test_the_executors_own_estimate_is_the_8_5_formula_on_its_snapshot(
    make_node: Callable[..., Node],
) -> None:
    node = make_node()
    rng = np.random.default_rng(2000)
    tasks = ["T1", "T3", "T4", "T2"]
    for step in range(60):
        node.submit(tasks[int(rng.integers(4))], LONG)
        node.run(0.004 * (step + 1))
        snap = node.executor.snapshot()
        for q, state in snap.items():
            for cls in ("hard", "soft"):
                assert node.executor.wait_estimate(q, cls) == pytest.approx(wait_estimate(state, cls))


# --- W6: c_q = 1 exact in the mean (8.5) --------------------------------------------------------------------


def poisson_single_slot(run: float, rho: float, n: int, exponential: bool) -> tuple[np.ndarray, np.ndarray]:
    """One slot, one hard class with equal budgets (EDF = FIFO), Poisson arrivals: (Ŵ, real wait) per task."""
    n_ = tiny([row("M", "cpu", run, distribution=Exponential(run) if exponential else None)])
    gaps = np.random.default_rng(2001).exponential(run / rho, n)
    estimates, arrival = [], {}
    t = 0.0
    for gap in gaps:
        t += float(gap)
        n_.clock.run_until(t)
        estimates.append(n_.ex.wait_estimate("cpu", "hard"))
        req = n_.submit("M", "hard", LONG)
        arrival[req.key] = t
    n_.clock.run_until(t + 1e4)
    starts = {rep.key: rep.start for rep in n_.reports}
    waits = np.array([starts[k] - a for k, a in arrival.items()])  # type: ignore[operator]
    return np.array(estimates), waits


def test_one_slot_with_fixed_runs_the_estimate_is_exact() -> None:
    # M/D/1, FIFO: the work ahead of a newcomer is exactly R + B_hard, so Ŵ equals its real wait, task by
    # task.
    est, waits = poisson_single_slot(run=0.010, rho=0.8, n=5_000, exponential=False)
    assert np.allclose(est, waits, rtol=0, atol=1e-12)


def test_one_slot_with_exponential_runs_the_literal_r_underestimates() -> None:
    # Q9: with exponential runs the expected remaining time stays Ê whatever has elapsed, but R counts only
    # max(0, Ê - elapsed). This pins the size of that gap at ρ = 0.8, for Imed's answer to Q9.
    est, waits = poisson_single_slot(run=0.010, rho=0.8, n=50_000, exponential=True)
    ratio = est.mean() / waits.mean()
    assert 0.6 < ratio < 0.95, ratio
