"""The water-filling wait estimate (CHANGES C4; final-design 8.5).

Ŵ is the W with sum_i max(0, W - r_i) = B: the backlog B, poured onto the c_q slots in the order they free
up, fills them all to W. r_i = each slot's remaining time (0 when free); still 0 while a slot is free.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pcnme.node.executor import Accepted
from pcnme.node.state import Pending, SlotState, wait_estimate, water_fill

if TYPE_CHECKING:
    from tests.node.conftest import Node


def state(
    remaining: tuple[float, ...], b_hard: float = 0.0, b_soft: float = 0.0, u: int | None = None
) -> SlotState:
    busy = sum(1 for r in remaining if r > 0) if u is None else u
    return SlotState(
        c=len(remaining),
        capacity=32,
        n_hard=0,
        n_soft=0,
        b_hard=b_hard,
        b_soft=b_soft,
        remaining=remaining,
        u=busy,
    )


def test_water_fill_by_hand() -> None:
    # r = (2, 700) ms, B = 8 ms: the first slot frees at 2 ms and takes all 8 ms of backlog -> W = 10 ms
    # (it stays below 700 ms, so the second slot never shares it).
    assert water_fill((0.002, 0.700), 0.008) == pytest.approx(0.010)
    # r = (6, 10) ms, B = 8 ms: level 14 ms on one slot would pass 10 ms, so both fill: (8 + 6 + 10) / 2 = 12
    # ms.
    assert water_fill((0.006, 0.010), 0.008) == pytest.approx(0.012)
    # Order does not matter; B = 0 waits for the first slot to free.
    assert water_fill((0.010, 0.006), 0.008) == pytest.approx(0.012)
    assert water_fill((0.004, 0.009), 0.0) == pytest.approx(0.004)


def test_the_level_solves_the_defining_equation() -> None:
    rng = np.random.default_rng(2000)
    for _ in range(500):
        r = tuple(float(x) for x in rng.uniform(0.0, 1.0, int(rng.integers(1, 6))))
        b = float(rng.uniform(0.0, 2.0))
        w = water_fill(r, b)
        assert sum(max(0.0, w - ri) for ri in r) == pytest.approx(b, abs=1e-12)
        assert w >= min(r) - 1e-15


def test_one_slot_is_the_old_formula() -> None:
    # c_q = 1: W = r + B = R + B (8.5), so "exact in the mean" for c_q = 1 is unchanged.
    assert water_fill((0.030,), 0.025) == pytest.approx(0.055)


def test_equal_remaining_times_give_the_pooled_formula() -> None:
    assert water_fill((0.008, 0.008), 0.020) == pytest.approx((0.016 + 0.020) / 2)


def test_still_zero_while_a_slot_is_free_and_pending_sends_count() -> None:
    free = state((0.006, 0.0))
    assert wait_estimate(free, "hard") == 0.0
    # One pending hard send (8 ms) takes the free slot: W solves max(0, W - 0) + max(0, W - 6) = 8 -> 7 ms.
    assert wait_estimate(free, "hard", [Pending(cls="hard", mean=0.008)]) == pytest.approx(0.007)


def test_a_pending_send_cannot_be_spread_over_two_free_slots() -> None:
    # Both slots free, one pending send: the newcomer takes the other free slot, so Ŵ = 0. Water-filling alone
    # would pour the 8 ms over both slots and give 4 ms: the free-slot rule (8.5) comes first.
    assert wait_estimate(state((0.0, 0.0)), "hard", [Pending(cls="hard", mean=0.008)]) == 0.0


def test_stealing_uses_the_water_filling_estimate_too() -> None:
    # 1 GPU + 2 CPU slots. At 0 the CPU runs "long" (720 ms) and "short" (25 ms); the GPU runs "g" (mean
    # 200 ms, really 20 ms). X (GPU 100 ms, CPU 60 ms) joins the CPU queue (CPU F = 25 + 60 = 85 ms < GPU
    # 200 + 100). At 20 ms the GPU frees: X's CPU finish = 20 + water((5, 700), 0) + 60 = 85 ms, earlier than
    # the GPU's 20 + 100 = 120 ms, so no steal; X runs on the CPU at 25 ms. The pooled estimate,
    # 20 + (705 / 2) + 60 = 432 ms, would have stolen it to the GPU.
    from tests.node.helpers import row, tiny

    rows = [
        row("g", "gpu", 0.200, run=0.020),
        row("long", "cpu", 0.720),
        row("short", "cpu", 0.025),
        row("X", "gpu", 0.100),
        row("X", "cpu", 0.060),
    ]
    n = tiny(rows, c_gpu=1, c_cpu=2)
    n.submit("g", "hard", 10.0)
    n.submit("long", "hard", 10.0)
    n.submit("short", "hard", 10.0)
    x = n.submit("X", "hard", 10.0)
    n.clock.run_until(5.0)
    rep = next(r for r in n.reports if r.key == x.key)
    assert (rep.slot, rep.start) == ("cpu", pytest.approx(0.025))


def test_class_backlogs() -> None:
    s = state((0.002, 0.700), b_hard=0.008, b_soft=0.025)
    assert wait_estimate(s, "hard") == pytest.approx(0.010)  # 2 + 8
    assert wait_estimate(s, "soft") == pytest.approx(0.035)  # 2 + 8 + 25, still below 700


def test_the_snapshot_publishes_each_slots_remaining_time(make_node: Callable[..., Node]) -> None:
    node = make_node()
    node.submit("T2", 1e6)  # 720 ms on one GPU slot
    node.submit("T1", 1e6)  # 8 ms on the other
    node.run(0.002)
    gpu = node.executor.snapshot()["gpu"]
    assert gpu.remaining == pytest.approx((0.006, 0.718))  # sorted, one entry per slot
    assert gpu.r == pytest.approx(0.724)  # the old sum R, kept for logs
    assert node.executor.snapshot()["cpu"].remaining == (0.0, 0.0)


def test_with_a_t2_on_one_gpu_slot_the_new_estimate_matches_the_real_wait(
    make_node: Callable[..., Node],
) -> None:
    # The web review's case (CHANGES C4): the config's F-GPU node at S2 load (74 tasks/s, the whole mix),
    # fixed run times. For every hard task accepted on the GPU while a T2 runs there: the new Ŵ (from the
    # accept reply) against the real wait, and the old pooled (R + B_hard) / c on the same snapshot.
    node = make_node()
    cfg = node.cfg
    rng = np.random.default_rng(2002)
    types = list(cfg.task_types.values())
    shares = np.array([t.share for t in types])
    rate = cfg.derived.arrival_rate_per_zone["S2"]
    t = 0.0
    samples: list[tuple[object, float, float, float]] = []  # (key, arrival, new Ŵ, old Ŵ)
    while t < 30.0:
        t += float(rng.exponential(1 / rate))
        node.run(t)
        task = types[int(rng.choice(len(types), p=shares))]
        gpu = node.executor.snapshot()["gpu"]
        t2_running = any(r > 0.050 for r in gpu.remaining)  # only a T2 runs longer than 50 ms on the GPU
        old = 0.0 if gpu.u < gpu.c else (gpu.r + gpu.b_hard) / gpu.c
        req = node.request(task.name, task.deadline["loose"])
        reply = node.executor.submit(req)
        if task.cls == "hard" and t2_running and isinstance(reply, Accepted) and reply.slot == "gpu":
            samples.append((req.key, t, reply.wait, old))
    node.run(t + 10.0)
    start = {r.key: r.start for r in node.reports if r.start is not None}
    real = np.array([start[k] - a for k, a, _, _ in samples if k in start])
    new = np.array([w for k, _, w, _ in samples if k in start])
    old = np.array([o for k, _, _, o in samples if k in start])
    assert len(real) > 300
    assert new.mean() == pytest.approx(real.mean(), rel=0.15)
    assert old.mean() / real.mean() > 1.6  # the pooled estimate was about 2x too high
