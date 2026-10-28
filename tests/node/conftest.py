"""Shared helpers for executor tests: an F-GPU node from the config with fixed (deterministic) run times."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from pcnme.config import Config, load_config
from pcnme.node.executor import AttemptKey, Executor, Report, TaskRequest
from pcnme.node.timetable import RunTimeTable, deterministic
from pcnme.sim.clock import EventClock
from pcnme.sim.rng import RandomStreams


@pytest.fixture(scope="session")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


@dataclass
class Node:
    """An executor on its own clock, with every report collected."""

    clock: EventClock
    executor: Executor
    cfg: Config
    reports: list[Report] = field(default_factory=list)
    trace: list[tuple[str, float, AttemptKey, str]] = field(default_factory=list)
    _seq: int = 0

    def request(
        self, task_type: str, budget: float, mode: str = "loose", car: str = "c01", attempt: int = 1
    ) -> TaskRequest:
        """A request for a new task of `task_type` (class and L_max from the config)."""
        self._seq += 1
        t = self.cfg.task_types[task_type]
        return TaskRequest(
            key=AttemptKey(task_id=(car, 0, self._seq), anchor_id="z1", attempt=attempt),
            task_type=task_type,
            cls=t.cls,
            l_max=t.l_max[mode],  # type: ignore[index]
            budget=budget,
        )

    def submit(self, task_type: str, budget: float, **kw: object) -> object:
        return self.executor.submit(self.request(task_type, budget, **kw))  # type: ignore[arg-type]

    def run(self, until: float) -> None:
        self.clock.run_until(until)


@pytest.fixture
def make_node(cfg: Config) -> Callable[..., Node]:
    """make_node(hw="F-GPU", shape=deterministic): an executor of that hardware class from the config."""

    def make(hw: str = "F-GPU", shape: Callable[..., object] = deterministic) -> Node:
        clock = EventClock()
        reports: list[Report] = []
        trace: list[tuple[str, float, AttemptKey, str]] = []
        executor = Executor.from_config(
            clock,
            RandomStreams.from_config(cfg, 2000, "training"),
            cfg,
            hw,
            RunTimeTable.from_config(cfg, shape),  # type: ignore[arg-type]
            on_report=reports.append,
            trace=trace.append,
        )
        return Node(clock=clock, executor=executor, cfg=cfg, reports=reports, trace=trace)

    return make
