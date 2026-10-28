"""Which task a free slot takes (PLAN M1.6; final-design 8.4; QUESTIONS Q10, Q11).

One test or more per rule of 8.4. Most use a small node with explicit rows, because with run times equal to
their means a hard task is never skipped: the hard estimate already counts every queued hard task (8.5, safe
side), so a skip needs a run that takes longer than its mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from pcnme.config import HardwareClass, SlotType, TaskClass
from pcnme.node.executor import Accepted, AttemptKey, Executor, Report, TaskRequest
from pcnme.node.timetable import Deterministic, RunRow, RunTimeTable
from pcnme.sim.clock import EventClock
from pcnme.sim.rng import RandomStreams

A_MAX = 0.200  # config scheduler.soft_promotion_A_max
SPLIT = {"training": range(2000, 3000)}


def row(task: str, slot: SlotType, mean: float, run: float | None = None, low: float | None = None) -> RunRow:
    """A row on the test node "N": mean Ê, p5 = 0.9 Ê unless given, p95 = 1.3 Ê; runs take `run` (or Ê)."""
    return RunRow(
        task=task,
        hw="N",
        slot=slot,
        mean=mean,
        low=0.9 * mean if low is None else low,
        high=1.3 * mean,
        longest_step=0.5 * mean,
        energy_per_run=20.0 * mean,
        distribution=Deterministic(mean if run is None else run),
    )


@dataclass
class Tiny:
    clock: EventClock
    ex: Executor
    reports: list[Report]
    trace: list[tuple[str, float, AttemptKey, str]]
    seq: list[int] = field(default_factory=lambda: [0])

    def req(self, task: str, cls: TaskClass, budget: float, l_max: float = 0.0) -> TaskRequest:
        self.seq[0] += 1
        return TaskRequest(AttemptKey(("c01", 0, self.seq[0]), "z1", 1), task, cls, l_max, budget)

    def submit(self, task: str, cls: TaskClass, budget: float, l_max: float = 0.0) -> TaskRequest:
        r = self.req(task, cls, budget, l_max)
        assert isinstance(self.ex.submit(r), Accepted), r
        return r

    def started(self) -> list[AttemptKey]:
        return [k for kind, _, k, _ in self.trace if kind == "start"]

    def outcome(self, r: TaskRequest) -> str:
        return next(rep.outcome for rep in self.reports if rep.key == r.key)


def tiny(rows: list[RunRow], c_gpu: int = 0, c_cpu: int = 1, promotion: str = "mean") -> Tiny:
    hw = HardwareClass(
        name="N",
        c_gpu=c_gpu,
        c_cpu=c_cpu,
        queue={q: 100 for q, c in (("gpu", c_gpu), ("cpu", c_cpu)) if c > 0},
        p_dyn_w={q: 20.0 for q, c in (("gpu", c_gpu), ("cpu", c_cpu)) if c > 0},
        p_idle_w=0.0,
    )
    clock, reports, trace = EventClock(), list[Report](), list[tuple[str, float, AttemptKey, str]]()
    ex = Executor(
        clock,
        RandomStreams(2000, "training", SPLIT),
        hw,
        RunTimeTable({(r.task, r.hw, r.slot): r for r in rows}),
        reserve_hard=4,
        pue=1.0,
        a_max=A_MAX,
        promotion_estimate=promotion,  # type: ignore[arg-type]
        on_report=reports.append,
        trace=trace.append,
    )
    return Tiny(clock, ex, reports, trace)


# --- S1: skip hopeless heads (8.4 step 1; 8.6 "head of the queue") ------------------------------------------


def test_a_hard_head_is_skipped_when_it_can_no_longer_finish() -> None:
    # One CPU slot. "long" has a mean of 200 ms but really runs 300 ms. H (mean 90 ms, low 81 ms) is accepted
    # with F = 200 + 90 = 290 ms <= its 300 ms budget; when the slot frees at 300 ms, 300 + 81 > 300: skipped.
    n = tiny([row("long", "cpu", 0.200, run=0.300), row("H", "cpu", 0.090), row("next", "cpu", 0.010)])
    n.submit("long", "hard", 10.0)
    h = n.submit("H", "hard", 0.300)
    nxt = n.submit("next", "hard", 10.0)
    n.clock.run_until(5.0)
    assert n.outcome(h) == "skipped"
    skipped = next(r for r in n.reports if r.key == h.key)
    assert (skipped.finish, skipped.busy, skipped.energy, skipped.start) == (
        pytest.approx(0.300),
        0.0,
        0.0,
        None,
    )
    assert n.started()[1] == nxt.key  # the slot took the next task at once


def test_a_hard_head_that_can_still_just_finish_is_not_skipped() -> None:
    # Binary-exact times: "long" (mean 250 ms) really runs 312.5 ms; H (mean 125 ms, low 62.5 ms, d_k = 375
    # ms)
    # is accepted with F = 250 + 125 = 375 ms. At 312.5 ms, t + Ê^low = 375 ms = d_k, not later: it runs. The
    # skip
    # rule uses Ê^low, not the mean (t + Ê = 437.5 ms would be later).
    n = tiny([row("long", "cpu", 0.25, run=0.3125), row("H", "cpu", 0.125, low=0.0625)])
    n.submit("long", "hard", 10.0)
    h = n.submit("H", "hard", 0.375)
    n.clock.run_until(5.0)
    assert n.outcome(h) == "done"


def test_a_soft_head_is_skipped_only_past_deadline_plus_l_max() -> None:
    n = tiny([row("long", "cpu", 0.200, run=1.300), row("S", "cpu", 0.090)])
    n.submit("long", "hard", 10.0)
    ok = n.submit("S", "soft", 0.300, l_max=1.000)  # at 1.3 s: 1.3 + 0.081 <= 0.3 + 1.0? no -> skipped
    n2 = tiny([row("long", "cpu", 0.200, run=1.300), row("S", "cpu", 0.090)])
    n2.submit("long", "hard", 10.0)
    fine = n2.submit("S", "soft", 0.300, l_max=1.100)  # 1.381 <= 1.4: runs, late but within L_max
    n.clock.run_until(5.0)
    n2.clock.run_until(5.0)
    assert n.outcome(ok) == "skipped" and n2.outcome(fine) == "done"


def test_skipping_continues_until_a_head_can_run() -> None:
    n = tiny([row("long", "cpu", 0.200, run=0.300), row("H", "cpu", 0.040)])
    n.submit("long", "hard", 10.0)
    # Accepted at t = 0 with F = 200 + 40 k + 40 ms <= 330 ms (k = 0, 1, 2); at 300 ms, 300 + 36 > 330:
    # hopeless.
    doomed = [n.submit("H", "hard", 0.330) for _ in range(3)]
    keeper = n.submit("H", "hard", 10.0)
    n.clock.run_until(5.0)
    assert [n.outcome(d) for d in doomed] == ["skipped"] * 3
    assert n.outcome(keeper) == "done"


# --- S2: soft promotion (8.4 step 2) ------------------------------------------------------------------------


@pytest.mark.parametrize("promotion", ["mean", "high"])
def test_a_soft_task_that_waited_a_max_runs_before_hard_tasks_that_can_spare_it(promotion: str) -> None:
    # One slot busy until 250 ms. S (soft) waits from 0; H (hard, mean 10 ms, deadline 10 s) arrives at 240
    # ms.
    # At 250 ms S has waited >= A_max = 200 ms, and H still meets 10 s with S on the slot (high 13 ms): S
    # first.
    n = tiny([row("long", "cpu", 0.250), row("S", "cpu", 0.010), row("H", "cpu", 0.010)], promotion=promotion)
    n.submit("long", "hard", 10.0)
    s = n.submit("S", "soft", 10.0, l_max=10.0)
    n.clock.run_until(0.240)
    h = n.submit("H", "hard", 9.76)
    n.clock.run_until(5.0)
    assert n.started()[1:] == [s.key, h.key]


@pytest.mark.parametrize("promotion", ["mean", "high"])
def test_no_promotion_before_a_max(promotion: str) -> None:
    n = tiny([row("long", "cpu", 0.150), row("S", "cpu", 0.010), row("H", "cpu", 0.010)], promotion=promotion)
    n.submit("long", "hard", 10.0)
    s = n.submit("S", "soft", 10.0, l_max=10.0)
    h = n.submit("H", "hard", 10.0)
    n.clock.run_until(5.0)  # slot frees at 150 ms < A_max: hard first
    assert n.started()[1:] == [h.key, s.key]


@pytest.mark.parametrize("promotion", ["mean", "high"])
def test_no_promotion_when_a_queued_hard_task_would_miss(promotion: str) -> None:
    # At 250 ms: S would hold the slot for its high, 130 ms; H (mean 10 ms) must finish by 300 ms. With S
    # first
    # H ends at 250 + 130 + 10 (or 13) > 300 ms: no promotion, H first.
    n = tiny([row("long", "cpu", 0.250), row("S", "cpu", 0.100), row("H", "cpu", 0.010)], promotion=promotion)
    n.submit("long", "hard", 10.0)
    s = n.submit("S", "soft", 10.0, l_max=10.0)
    n.clock.run_until(0.240)
    h = n.submit("H", "hard", 0.060)
    n.clock.run_until(5.0)
    assert n.started()[1:] == [h.key, s.key]


@pytest.mark.parametrize("promotion", ["mean", "high"])
def test_the_promotion_check_list_schedules_the_hard_queue_on_all_slots(promotion: str) -> None:
    # Two slots. At 250 ms slot 1 frees and slot 2 runs until 260 ms. S waited A_max. H1 and H2 (10 ms each,
    # deadline 300 ms) are list-scheduled on the 2 slots with S on slot 1 for 13 ms: H1 on slot 2 from 260 ms
    # (ends 270 or 273), H2 after it (ends 280 or 286) <= 300 ms: S is promoted.
    n = tiny(
        [row("a", "cpu", 0.250), row("b", "cpu", 0.260), row("S", "cpu", 0.010), row("H", "cpu", 0.010)],
        c_cpu=2,
        promotion=promotion,
    )
    n.submit("a", "hard", 10.0)
    n.submit("b", "hard", 10.0)
    s = n.submit("S", "soft", 10.0, l_max=10.0)
    n.clock.run_until(0.245)
    h1, h2 = n.submit("H", "hard", 0.055), n.submit("H", "hard", 0.055)
    n.clock.run_until(5.0)
    assert n.started()[2:] == [s.key, h1.key, h2.key]


def test_promotion_needs_an_explicit_estimate_choice() -> None:
    # Q10 is open: there is no default.
    with pytest.raises(ValueError, match="promotion"):
        tiny([row("x", "cpu", 0.01)], promotion="p99")


# --- S5: work stealing (8.4 step 5) -------------------------------------------------------------------------
# One GPU and one CPU slot. "g" holds the GPU with a mean of 200 ms but really ends at 20 ms; "c" holds the
# CPU
# for 100 ms. X (GPU 50 ms, CPU 60 ms) arrives at 0: GPU F = 200 + 50 = 250 ms, CPU F = 100 + 60 = 160 ms, so
# it
# joins the CPU queue. At 20 ms the GPU frees with an empty queue.


def stealing_node(x_gpu: float, x_gpu_low: float | None = None, g_run: float = 0.020) -> Tiny:
    rows = [
        row("g", "gpu", 0.200, run=g_run),
        row("c", "cpu", 0.100),
        row("X", "gpu", x_gpu, low=x_gpu_low),
        row("X", "cpu", 0.060),
        row("C", "cpu", 0.060),
        row("L", "cpu", 0.500),
    ]
    n = tiny(rows, c_gpu=1, c_cpu=1)
    n.submit("g", "hard", 10.0)
    n.submit("c", "hard", 10.0)
    return n


def slot_of(n: Tiny, r: TaskRequest) -> str:
    return next(rep.slot for rep in n.reports if rep.key == r.key)


def test_an_empty_queue_steals_a_task_that_finishes_earlier_here() -> None:
    # At 20 ms: X's current estimated CPU finish = 20 + (100 - 20) remaining + 60 = 160 ms (Q11); on the GPU
    # it
    # would finish at 20 + 50 = 70 ms < 160: stolen, runs 20-70 ms on the GPU.
    n = stealing_node(x_gpu=0.050)
    x = n.submit("X", "hard", 10.0)
    n.clock.run_until(5.0)
    rep = next(r for r in n.reports if r.key == x.key)
    assert (rep.slot, rep.start, rep.finish) == ("gpu", pytest.approx(0.020), pytest.approx(0.070))


def test_the_first_task_in_the_other_queues_order_is_stolen() -> None:
    # At 20 ms the GPU steals `first` (earlier deadline, first in the CPU queue's order), not `later`. At 70
    # ms
    # it frees again: `later` is then stolen too (70 + 50 = 120 ms < its CPU finish 70 + 30 + 60 = 160 ms).
    n = stealing_node(x_gpu=0.050)
    later = n.submit("X", "hard", 10.0)
    first = n.submit("X", "hard", 5.0)
    n.clock.run_until(5.0)
    starts = {r.key: (r.slot, r.start) for r in n.reports}
    assert starts[first.key] == ("gpu", pytest.approx(0.020))
    assert starts[later.key] == ("gpu", pytest.approx(0.070))


def test_the_estimated_finish_counts_only_the_tasks_ahead_in_queue_order() -> None:
    # A soft task L (500 ms) is queued on the CPU before hard X arrives, but hard comes first in the queue's
    # order (8.2), so L is not ahead of X. At 20 ms X's CPU finish = 20 + 80 + 60 = 160 ms; the GPU would take
    # 20 + 200 = 220 ms: no steal. Counting L as ahead (660 ms) would wrongly steal it.
    n = stealing_node(x_gpu=0.200)
    n.submit("L", "soft", 10.0, l_max=10.0)
    x = n.submit("X", "hard", 10.0)
    n.clock.run_until(5.0)
    assert slot_of(n, x) == "cpu"


def test_no_steal_when_it_would_not_finish_earlier_here() -> None:
    # X takes 500 ms on the GPU: 20 + 500 = 520 ms > its CPU finish 160 ms. It stays and runs on the CPU.
    n = stealing_node(x_gpu=0.500)
    x = n.submit("X", "hard", 10.0)
    n.clock.run_until(5.0)
    assert slot_of(n, x) == "cpu"


def test_no_steal_of_a_task_that_cannot_run_here() -> None:
    n = stealing_node(x_gpu=0.050)
    c = n.submit("C", "hard", 10.0)  # C has no GPU row
    n.clock.run_until(5.0)
    assert slot_of(n, c) == "cpu"


def test_no_steal_of_a_task_that_fails_the_skip_rule_here() -> None:
    # The GPU frees at 150 ms. "c" (mean 100 ms) really runs 300 ms, so at 150 ms X's estimated CPU finish is
    # 150 + 0 + 60 = 210 ms and the GPU (150 + 50 = 200 ms) is earlier, but X's deadline is 165 ms and
    # 150 + 45 (GPU low) > 165: not stolen. At 300 ms the CPU skips it (300 + 54 > 165).
    rows = [
        row("g", "gpu", 0.200, run=0.150),
        row("c", "cpu", 0.100, run=0.300),
        row("X", "gpu", 0.050),
        row("X", "cpu", 0.060),
    ]
    n = tiny(rows, c_gpu=1, c_cpu=1)
    n.submit("g", "hard", 10.0)
    n.submit("c", "hard", 10.0)
    x = n.submit("X", "hard", 0.165)  # CPU F = 100 + 60 = 160 <= 165 ms
    n.clock.run_until(5.0)
    rep = next(r for r in n.reports if r.key == x.key)
    assert (rep.outcome, rep.slot) == ("skipped", "cpu")


# --- S6: idle -----------------------------------------------------------------------------------------------


def test_a_slot_with_nothing_to_take_idles() -> None:
    n = tiny([row("A", "cpu", 0.010)], c_cpu=2)
    n.submit("A", "hard", 10.0)
    n.clock.run_until(1.0)
    assert n.ex.busy("cpu") == 0 and n.ex.queue_length("cpu") == 0
