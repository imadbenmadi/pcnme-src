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


@dataclass(frozen=True, slots=True)
class Attach:
    """POST /cars, car -> the zone it now uses (9.6): its distance for the radio session, where results go."""

    car_id: str
    distance_m: float
    url: str  # the car container's address for POST /results (radio_net)


@dataclass(frozen=True, slots=True)
class Gps:
    """POST /gps, car -> its serving zone every GPS period (5.2, 10); car_time is the car's own clock."""

    car_id: str
    x: float
    y: float
    vx: float
    vy: float
    car_time: float


@dataclass(frozen=True, slots=True)
class GpsBatch:
    """POST /gps: the fixes of one car container's cars in one zone for one GPS period. Transport only (the
    zone handles each fix as its own message, 5.2); it keeps the zones' HTTP load to the testbed's means."""

    fixes: tuple[Gps, ...]
