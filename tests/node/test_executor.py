"""Executor: slots, queues, slot-type choice, refusals (PLAN M1.5; final-design 8.1-8.3, 8.5, 6C, 12.2).

An F-GPU node from the config: c_GPU = 2, c_CPU = 2, Q_GPU = 32, Q_CPU = 16, r = 4. Run times are the
placeholder means (deterministic), so every number below is computed by hand.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from pcnme.node.executor import Accepted, Refused, choose_slot

if TYPE_CHECKING:
    from tests.node.conftest import Node

T1_GPU, T1_CPU = 0.008, 0.090  # config time_table_placeholder
T4_GPU, T4_CPU = 0.025, 0.200
LONG = 1e6  # a budget no test reaches


# --- R6, R7, R10: slot-type choice by earliest estimated finish (8.3) ---------------------------------------


def test_an_empty_node_takes_the_faster_slot_type_with_no_wait(make_node: Callable[..., Node]) -> None:
    node = make_node()
    assert node.submit("T1", 0.100) == Accepted(slot="gpu", wait=0.0)


def test_the_cpu_is_chosen_once_the_gpu_would_finish_later(make_node: Callable[..., Node]) -> None:
    # n T1 tasks submitted at t = 0. Tasks 1-2 start on the 2 GPU slots. For task n >= 3 the GPU estimate is
    # Ŵ = (R + B_hard) / 2 = (2 x 8 + (n - 3) x 8) / 2 ms = 4 (n - 1) ms, so F_GPU = 4 (n - 1) + 8 ms, against
    # F_CPU = 0 + 90 ms (CPU free). The GPU wins while 4 (n - 1) + 8 <= 90, so n <= 21: task 22 goes to a CPU.
    node = make_node()
    replies = [node.submit("T1", 0.100) for _ in range(22)]
    slots = [r.slot for r in replies if isinstance(r, Accepted)]
    assert slots == ["gpu"] * 21 + ["cpu"]
    assert [r.wait for r in replies[:4]] == pytest.approx([0.0, 0.0, 0.008, 0.012])  # type: ignore[union-attr]
    assert replies[20].wait == pytest.approx(0.080)  # type: ignore[union-attr]  # task 21: 4 x 20 ms


def test_equal_estimates_go_to_the_gpu() -> None:
    # Engineering choice (not in the design): ties in F_q go to the GPU slot type.
    assert choose_slot({"gpu": 0.050, "cpu": 0.050}) == "gpu"
    assert choose_slot({"gpu": 0.051, "cpu": 0.050}) == "cpu"


# --- R4, R9: capacity and the hard reserve (8.2.3) ----------------------------------------------------------


def test_a_hard_task_is_refused_full_when_every_queue_is_full(make_node: Callable[..., Node]) -> None:
    # 2 + 32 on the GPU and 2 + 16 on the CPU = 52 places. With a long budget every T1 task finds one (the
    # slot type alternates by estimated finish), so after 52 both queues are full and the 53rd is refused.
    node = make_node()
    accepted = [node.submit("T1", LONG) for _ in range(52)]
    assert all(isinstance(r, Accepted) for r in accepted)
    assert node.executor.queue_length("gpu") == 32 and node.executor.queue_length("cpu") == 16
    assert node.submit("T1", LONG) == Refused(reason="full")


def test_a_soft_task_needs_a_place_below_q_minus_r(make_node: Callable[..., Node]) -> None:
    # T2 is soft and GPU only: 2 run and 28 queue (Q_GPU - r = 28). The 31st T2 has no place: refused "full".
    # A soft T4 can still go to the free CPU slots.
    node = make_node()
    assert node.submit("T2", LONG) == Accepted(slot="gpu", wait=0.0)
    assert node.submit("T2", LONG) == Accepted(slot="gpu", wait=0.0)
    for _ in range(28):
        assert isinstance(node.submit("T2", LONG), Accepted)
    assert node.executor.queue_length("gpu") == 28
    assert node.submit("T2", LONG) == Refused(reason="full")
    assert node.submit("T4", LONG) == Accepted(slot="cpu", wait=0.0)


def test_a_hard_task_can_take_the_reserved_places(make_node: Callable[..., Node]) -> None:
    # After 28 queued soft T2, the GPU queue still admits hard tasks (it holds fewer than Q_GPU = 32). T1 goes
    # to the CPU while that finishes earlier, then to the GPU's reserved places.
    node = make_node()
    for _ in range(30):
        node.submit("T2", LONG)
    slots = [node.submit("T1", LONG) for _ in range(40)]
    assert any(isinstance(r, Accepted) and r.slot == "gpu" for r in slots)
    assert node.executor.queue_length("gpu") == 32  # the 4 reserved places, filled by hard tasks only


def test_a_soft_task_is_refused_full_when_only_reserved_places_are_left(
    make_node: Callable[..., Node],
) -> None:
    node = make_node()
    for _ in range(30):
        node.submit("T2", LONG)  # GPU: no soft place left
    for _ in range(2 + 12):  # CPU: 2 running + 12 queued = Q_CPU - r
        assert isinstance(node.submit("T4", LONG), Accepted)
    assert node.executor.queue_length("cpu") == 12
    assert node.submit("T4", LONG) == Refused(reason="full")


# --- R5, R8: refused late (8.3.3); d_k = arrival + B_k (6C) -------------------------------------------------


def test_a_hard_task_that_cannot_finish_by_its_deadline_is_refused_late(
    make_node: Callable[..., Node],
) -> None:
    node = make_node()
    assert node.submit("T1", 0.007) == Refused(reason="late")  # best F = 8 ms > 7 ms
    assert node.submit("T1", 0.008) == Accepted(slot="gpu", wait=0.0)  # F = d_k exactly is in time


def test_a_soft_task_is_refused_late_only_past_deadline_plus_l_max(make_node: Callable[..., Node]) -> None:
    # T4 (soft, loose D = L_max = 1 s): F = 25 ms. Budget 0.01 s -> d_k + L_max = 1.01 s > 25 ms: accepted.
    node = make_node()
    assert node.submit("T4", 0.010) == Accepted(slot="gpu", wait=0.0)
    assert node.submit("T4", 0.020, mode="tight") == Accepted(slot="gpu", wait=0.0)  # L_max = 0.5 s
    assert node.submit("T4", -0.990) == Refused(reason="late")  # d_k + L_max = 10 ms < 25 ms


def test_the_deadline_counts_from_the_arrival_time(make_node: Callable[..., Node]) -> None:
    node = make_node()
    node.run(5.0)
    assert node.submit("T1", 0.008) == Accepted(slot="gpu", wait=0.0)  # d_k = 5.008, F = 5 + 0.008


def test_a_task_no_slot_type_can_run_is_refused_late(make_node: Callable[..., Node]) -> None:
    assert make_node("F-CPU").submit("T2", LONG) == Refused(reason="late")  # T2 is GPU only (8.3 table)


def test_the_cpu_only_node_runs_on_its_cpu_slots(make_node: Callable[..., Node]) -> None:
    node = make_node("F-CPU")
    assert node.submit("T1", LONG) == Accepted(slot="cpu", wait=0.0)


# --- R1, R12, R13, R14: running -----------------------------------------------------------------------------


def test_one_task_per_slot_and_runs_take_their_drawn_time(make_node: Callable[..., Node]) -> None:
    node = make_node()
    for _ in range(3):
        node.submit("T1", LONG)
    assert node.executor.busy("gpu") == 2 and node.executor.queue_length("gpu") == 1
    node.run(0.008)
    done = [(r.outcome, r.finish) for r in node.reports]
    assert done == [("done", pytest.approx(0.008)), ("done", pytest.approx(0.008))]
    assert node.executor.busy("gpu") == 1  # the third started at 8 ms
    node.run(1.0)
    assert node.reports[-1].finish == pytest.approx(0.016)


def test_the_finish_report_carries_the_execution_energy(make_node: Callable[..., Node]) -> None:
    # final-design 12.2: E_exec = PUE_fog x p x T_run = 1.5 x 25 W x 8 ms = 0.3 J on the F-GPU GPU slot.
    node = make_node()
    node.submit("T1", LONG)
    node.run(1.0)
    (r,) = node.reports
    assert (r.slot, r.busy, r.energy) == ("gpu", pytest.approx(0.008), pytest.approx(1.5 * 25 * 0.008))


def test_cloud_execution_energy_uses_the_cloud_pue(make_node: Callable[..., Node]) -> None:
    node = make_node("cloud")
    node.submit("T2", LONG)
    node.run(1.0)
    assert node.reports[0].energy == pytest.approx(1.1 * 150 * 0.200)  # PUE_cloud 1.1, 150 W, 200 ms


# --- R3: order inside the queue (8.2.2) ---------------------------------------------------------------------
# On the F-CPU node (one slot type, c_CPU = 2), both slots first busy with a T4 (200 ms).


def starts(node: Node) -> list[object]:
    return [key for kind, _, key, _ in node.trace if kind == "start"]


def test_a_free_slot_takes_hard_before_soft_then_earliest_deadline(make_node: Callable[..., Node]) -> None:
    # Both CPU slots run a T3 (110 ms), so they free before the soft task has waited A_max = 200 ms (no soft
    # promotion, 8.4 step 2, which test_scheduler.py covers).
    node = make_node("F-CPU")
    for _ in range(2):
        node.submit("T3", LONG)
    soft = node.request("T4", 5.0)
    late_hard = node.request("T1", 3.0)
    early_hard = node.request("T1", 2.0)
    replies = [node.executor.submit(req) for req in (soft, late_hard, early_hard)]
    # Hard waits count only the queued hard tasks: (2 x 110 ms) / 2, then + 90 ms / 2 for the second hard one.
    assert replies == [
        Accepted(slot="cpu", wait=pytest.approx(0.110)),
        Accepted(slot="cpu", wait=pytest.approx(0.110)),
        Accepted(slot="cpu", wait=pytest.approx(0.155)),
    ]
    node.run(10.0)
    assert starts(node)[2:] == [early_hard.key, late_hard.key, soft.key]


def test_ties_go_by_arrival_time_then_task_id(make_node: Callable[..., Node]) -> None:
    # Same d_k = 3.0 s for all three (times and budgets exact in binary), while both slots run a T4 until
    # 200 ms.
    node = make_node("F-CPU")
    for _ in range(2):
        node.submit("T4", LONG)
    node.run(0.0625)
    b = node.request("T1", 2.9375, car="c09")
    a = node.request("T1", 2.9375, car="c02")
    node.executor.submit(b)
    node.executor.submit(a)  # same arrival time and deadline: task id c02 < c09
    node.run(0.125)
    c = node.request("T1", 2.875, car="c01")  # the smallest task id, but it arrived later
    node.executor.submit(c)
    node.run(10.0)
    assert starts(node)[2:] == [a.key, b.key, c.key]


def test_wait_estimates_follow_8_5_on_the_exact_queue(make_node: Callable[..., Node]) -> None:
    # Q8 / Q9: the executor's own Ŵ = 0 with a free slot, else (R + B_hard) / c for hard and
    # (R + B_hard + B_soft) / c for soft; R = the remaining part of the running tasks' mean runs.
    node = make_node()
    ex = node.executor
    assert ex.wait_estimate("gpu", "hard") == 0.0
    node.submit("T1", LONG)
    assert ex.wait_estimate("gpu", "hard") == 0.0  # one slot still free
    node.submit("T1", LONG)
    node.submit("T4", LONG)  # queued soft, 25 ms
    node.submit("T1", LONG)  # queued hard, 8 ms
    node.run(0.002)
    r = 2 * (0.008 - 0.002)
    assert ex.wait_estimate("gpu", "hard") == pytest.approx((r + 0.008) / 2)
    assert ex.wait_estimate("gpu", "soft") == pytest.approx((r + 0.008 + 0.025) / 2)
