"""Identifiers: task ids, attempt keys, snapshot versions (final-design 5.5).

No central id database: every id is built from parts that are already unique, so no message or shared store is
needed and fast mode stays reproducible (no random UUIDs). In fast mode every epoch is 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from pcnme.sim.rng import KeyPart


class TaskId(NamedTuple):
    """(car_id, epoch, seq), made by the car when it creates the task (5.5). A tuple, so it orders as
    (car_id, epoch, seq) (the last tie rule of 8.2) and is a valid random-stream key."""

    car_id: str
    epoch: int
    seq: int

    def __str__(self) -> str:
        return f"{self.car_id}.{self.epoch}.{self.seq}"  # c03-047.2.1187 (5.5)


class TaskIds:
    """A car's task numbering: seq counts up within an epoch; the epoch rises at every start (5.5)."""

    def __init__(self, car_id: str, epoch: int = 0) -> None:
        self._car_id = car_id
        self._epoch = epoch
        self._seq = 0

    def next(self) -> TaskId:
        self._seq += 1
        return TaskId(self._car_id, self._epoch, self._seq)


@dataclass(frozen=True, slots=True, order=True)
class AttemptKey:
    """(task_id, anchor_id, attempt): names one attempt everywhere (5.5). The anchor id keeps two anchors of
    one task apart (9.7); attempts count from 1 (first placement, then retries)."""

    task_id: tuple[KeyPart, ...]  # a TaskId (car_id, epoch, seq)
    anchor_id: str
    attempt: int

    def __post_init__(self) -> None:
        if self.attempt < 1:
            raise ValueError(f"attempts count from 1, got attempt {self.attempt}")


class SnapshotVersion(NamedTuple):
    """(epoch, counter) on every state snapshot; newest wins, epoch compared first (5.5)."""

    epoch: int
    counter: int


class SnapshotCounter:
    """A zone's snapshot versions: the counter grows on every snapshot it sends (9.2 R1)."""

    def __init__(self, epoch: int = 0) -> None:
        self._epoch = epoch
        self._counter = 0

    def next(self) -> SnapshotVersion:
        self._counter += 1
        return SnapshotVersion(self._epoch, self._counter)
