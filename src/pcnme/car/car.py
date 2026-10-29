"""The car app: creates tasks, keeps its latency clock, keeps the first result, confirms (final-design
6A, 9.5, 14.1-14.3, 18).

A car creates tasks at its arrival rate λ_car (18; Poisson, QUESTIONS Q12), from the run's fleet size, each
of a type drawn by the count mix, with the scenario's deadlines and the config's sizes. Its clock starts at
creation (6A.1); the upload starts after the encode time (timing.encode, 14.3, Q14) and carries
B_car = D - time since creation (6A.2), so the encode time is counted once. It keeps the first result per
task id until D + L_max and confirms every copy, naming the copy it kept (9.5).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pcnme.config import Config, DeadlineMode, Scenario, TaskClass
from pcnme.ids import TaskId, TaskIds
from pcnme.sim.clock import EventClock
from pcnme.sim.quantize import to_ns
from pcnme.sim.rng import RandomStreams


@dataclass(frozen=True, slots=True)
class Task:
    """One task as the car creates it (6A.1)."""

    task_id: TaskId
    car_id: str
    task_type: str
    cls: TaskClass
    deadline: float  # D, s
    l_max: float  # = D (2)
    input_bits: float
    result_bits: float
    created: float  # on the car's clock


@dataclass(frozen=True, slots=True)
class Result:
    """A copy of a result as it reaches the car: which attempt of which anchor produced it."""

    task_id: TaskId
    anchor_id: str
    attempt: int


@dataclass(frozen=True, slots=True)
class Confirm:
    """The car's confirm (9.5.2): the copy it kept and its latency L = arrival - creation (5.4)."""

    task_id: TaskId
    anchor_id: str
    attempt: int
    latency: float


class Car:
    """One car in fast mode."""

    def __init__(
        self,
        car_id: str,
        config: Config,
        scenario: Scenario,
        streams: RandomStreams,
        clock: EventClock,
        position: tuple[float, float],
        on_task: Callable[[Task], None],
        fleet: int,
        coverage: float = 1.0,
    ) -> None:
        """`fleet` is the run's city-wide number of cars: λ_car = the per-zone rate / cars per zone (18)."""
        self.car_id = car_id
        self.position = position
        self.clock = clock
        self._mode: DeadlineMode = scenario.deadlines
        self._types = list(config.task_types.values())
        self._shares = np.array([t.share for t in self._types])
        # Q29: the per-zone rate counts covered creations; frames made in a gap reach no zone
        self._rate = config.arrival_rate_per_car(scenario.name, fleet) / coverage
        self._encode = config.timing.encode
        self._arrivals = streams.generator("arrivals", car_id)  # gaps and types, in order (sequential, Q4)
        self._ids = TaskIds(car_id)
        self._on_task = on_task
        self._tasks: dict[TaskId, Task] = {}
        self._kept: dict[TaskId, Confirm] = {}

    def start(self, until: float) -> None:
        """Create tasks from now until `until` (s)."""
        self._schedule_next(until)

    def _schedule_next(self, until: float) -> None:
        gap = to_ns(float(self._arrivals.exponential(1 / self._rate)))  # Poisson (Q12), whole ns
        t = self.clock.now + gap
        if t <= until:
            self.clock.schedule(t, lambda: self._create(until))

    def _create(self, until: float) -> None:
        kind = self._types[int(self._arrivals.choice(len(self._types), p=self._shares))]
        task = Task(
            task_id=self._ids.next(),
            car_id=self.car_id,
            task_type=kind.name,
            cls=kind.cls,
            deadline=kind.deadline[self._mode],
            l_max=kind.l_max[self._mode],
            input_bits=kind.input_bits,
            result_bits=kind.result_bits,
            created=self.clock.now,
        )
        self._tasks[task.task_id] = task
        self.clock.schedule(task.created + task.deadline + task.l_max, lambda: self._forget(task.task_id))
        self._schedule_next(until)
        self.clock.schedule(task.created + self._encode, lambda: self._on_task(task))  # upload (14.3, Q14)

    def budget(self, task: Task, now: float) -> float:
        """B_car = D - time since creation, sent with the upload (6A.2). The upload starts after the encode
        time, so `now - created` already holds it (Q14)."""
        return task.deadline - (now - task.created)

    def receive(self, result: Result, now: float) -> Confirm | None:
        """A result copy reaches the car. The first one per task id is kept; every copy is confirmed with the
        kept copy's attempt (9.5.1-2). Unknown or expired task ids (after D + L_max) are ignored."""
        task = self._tasks.get(result.task_id)
        if task is None:
            return None
        if result.task_id not in self._kept:
            self._kept[result.task_id] = Confirm(
                result.task_id, result.anchor_id, result.attempt, now - task.created
            )
        return self._kept[result.task_id]

    def kept(self, task_id: TaskId) -> tuple[str, int] | None:
        c = self._kept.get(task_id)
        return None if c is None else (c.anchor_id, c.attempt)

    def _forget(self, task_id: TaskId) -> None:
        """9.5.1: the car keeps the id until deadline + L_max."""
        self._tasks.pop(task_id, None)
        self._kept.pop(task_id, None)
