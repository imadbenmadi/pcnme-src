"""Testbed-only payloads (final-design 5.3 POST /tasks, car -> its serving zone): the car's upload carries
B_car and R_car with the task (15.3: the radio pipe runs in software at the zone's intake)."""

from __future__ import annotations

from dataclasses import dataclass

from pcnme.car.car import Task
from pcnme.ids import TaskId


@dataclass(frozen=True, slots=True)
class Upload:
    task: Task
    b_car: float  # B_car, s: the car's budget when the upload starts (6A.2); never a timestamp (5.4)
    rate: float  # R_car, bit/s: the Shannon rate at the car's simulated position (4.2)


@dataclass(frozen=True, slots=True)
class UploadReply:
    task_id: TaskId
    anchor_id: str
