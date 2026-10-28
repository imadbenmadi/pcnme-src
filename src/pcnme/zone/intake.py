"""Intake at the anchor: the anchor deadline and the anchor table (final-design 6A, 6E, 9.7).

A task exists only once its whole upload has arrived (t_rx, 6A.2, 9.7.1). The anchor sets its deadline on its
own clock, taking off the upload time (6A.3):

    d_A = t_rx + B_car - Û_up,    Û_up = W_radio + S / R_car + δ̂_radio

W_radio is known exactly because the zone owns its uplink; δ̂_radio is the mean radio delay.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pcnme.car.car import Task
from pcnme.ids import TaskId


def anchor_deadline(
    t_rx: float, b_car: float, w_radio: float, size_bits: float, rate: float, mean_radio_delay: float
) -> float:
    """d_A (6A.3)."""
    upload = w_radio + size_bits / rate + mean_radio_delay
    return t_rx + b_car - upload


@dataclass(slots=True)
class AnchorRow:
    """One task this zone owns (3, 6A.4)."""

    task: Task
    anchor_id: str
    t_rx: float
    d_a: float
    attempts: int = 0
    notes: dict[str, object] = field(default_factory=dict)


class AnchorTable:
    """The tasks a zone owns, from intake to close (6A.4, 6E.15). Every task closes exactly once."""

    def __init__(self, zone_id: str) -> None:
        self.zone_id = zone_id
        self._open: dict[TaskId, AnchorRow] = {}
        self.closed: dict[TaskId, tuple[str, float]] = {}

    def open(self, task: Task, t_rx: float, d_a: float) -> AnchorRow:
        if task.task_id in self._open or task.task_id in self.closed:
            raise ValueError(f"task {task.task_id} already has a row in {self.zone_id}")
        row = AnchorRow(task=task, anchor_id=self.zone_id, t_rx=t_rx, d_a=d_a)
        self._open[task.task_id] = row
        return row

    def is_open(self, task_id: TaskId) -> bool:
        return task_id in self._open

    def row(self, task_id: TaskId) -> AnchorRow:
        return self._open[task_id]

    def next_attempt(self, task_id: TaskId) -> int:
        """The next attempt number: 1 for the first placement, then retries (5.5)."""
        row = self._open[task_id]
        row.attempts += 1
        return row.attempts

    def close(self, task_id: TaskId, outcome: str, now: float) -> AnchorRow:
        if task_id in self.closed:
            raise ValueError(f"task {task_id} already closed as {self.closed[task_id][0]}")
        row = self._open.pop(task_id)
        self.closed[task_id] = (outcome, now)
        return row

    @property
    def open_count(self) -> int:
        return len(self._open)
