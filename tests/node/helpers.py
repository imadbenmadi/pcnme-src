"""Small executor nodes with explicit rows, shared by the scheduler, drop and acceptance tests."""

from __future__ import annotations

from dataclasses import dataclass, field

from pcnme.config import HardwareClass, SlotType, TaskClass
from pcnme.node.executor import Accepted, AttemptKey, Executor, Report, TaskRequest
from pcnme.node.timetable import Deterministic, RunRow, RunTimeDistribution, RunTimeTable
from pcnme.sim.clock import EventClock
from pcnme.sim.rng import RandomStreams

A_MAX = 0.200  # config scheduler.soft_promotion_A_max
SPLIT = {"training": range(2000, 3000)}


def row(
    task: str,
    slot: SlotType,
    mean: float,
    run: float | None = None,
    low: float | None = None,
    distribution: RunTimeDistribution | None = None,
) -> RunRow:
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
        distribution=distribution or Deterministic(mean if run is None else run),
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
        t_keep=10.0,
        on_report=reports.append,
        trace=trace.append,
    )
    return Tiny(clock, ex, reports, trace)


def report(n: Tiny, key: AttemptKey) -> Report:
    """The report of one attempt."""
    return next(r for r in n.reports if r.key == key)
