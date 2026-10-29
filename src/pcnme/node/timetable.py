"""The time table a node runs from, and run-time draws (final-design 14.4, 12.2, 18; CHANGES C3).

One row per (task type, hardware class, slot type): the mean run Ê, Ê^low = p5 and Ê^high = p95, the longest
step (d_abort, 8.6) and the mean dynamic energy per run. A missing row means the task cannot run there.

A run time is drawn by task, not by event order (C3): u in (0, 1) from the keyed run_times stream by
(task_id, attempt), mapped through the inverse CDF of the row the task actually runs on, so a task keeps its
quantile on any node or slot type. The shape is a separate, swappable piece; the default is a lognormal fitted
to the row's mean and p95, both kept exact (CHANGES C5, QUESTIONS Q7).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from scipy.special import ndtri

from pcnme.config import SLOT_TYPES, Config, ConfigError, SlotType, TimeRow
from pcnme.sim.quantize import to_ns
from pcnme.sim.rng import KeyPart, RandomStreams

# The table's quantile columns (14.4): Ê^low and Ê^high are its p5 and p95.
_TABLE_QUANTILES = (0.05, 0.95)


class RunTimeDistribution(Protocol):
    """The shape of a row's run times: the inverse CDF of u in the open interval (0, 1), in seconds."""

    def inverse_cdf(self, u: float) -> float: ...


DistributionFactory = Callable[[TimeRow], RunTimeDistribution]


def _check_u(u: float) -> None:
    if not 0.0 < u < 1.0:  # also refuses NaN
        raise ValueError(f"u must lie in the open interval (0, 1), got {u!r}")


@dataclass(frozen=True, slots=True)
class Deterministic:
    """Every run takes exactly `value` seconds (for tests and the M/D/c checks)."""

    value: float

    def inverse_cdf(self, u: float) -> float:
        _check_u(u)
        return self.value


@dataclass(frozen=True, slots=True)
class Exponential:
    """Exponential run times with the given mean (for tests and the M/M/c checks, 8.7)."""

    mean: float

    def inverse_cdf(self, u: float) -> float:
        _check_u(u)
        return -self.mean * math.log1p(-u)


_Z95 = float(ndtri(0.95))  # 1.6449


@dataclass(frozen=True, slots=True)
class Lognormal:
    """ln(run) ~ N(mu, sigma^2), fitted to a row's mean and p95 (CHANGES C5)."""

    mu: float
    sigma: float

    @classmethod
    def fit(cls, mean: float, p95: float) -> Lognormal:
        """mean = exp(mu + sigma^2 / 2) and p95 = exp(mu + z95 sigma): the smaller root of
        sigma^2 / 2 - z95 sigma + ln(p95 / mean) = 0, so sigma = z95 - sqrt(z95^2 - 2 ln(p95 / mean))."""
        if not 0 < mean <= p95:
            raise ValueError(f"a lognormal fit needs 0 < mean <= p95, got mean {mean!r}, p95 {p95!r}")
        disc = _Z95**2 - 2 * math.log(p95 / mean)
        if disc < 0:
            raise ValueError(f"no lognormal has p95 / mean = {p95 / mean:.3f} > exp(z95^2 / 2) = 3.87")
        sigma = _Z95 - math.sqrt(disc)
        return cls(mu=math.log(mean) - sigma**2 / 2, sigma=sigma)

    @property
    def p5(self) -> float:
        """The fitted p5 (for the calibration report, C5)."""
        return math.exp(self.mu - _Z95 * self.sigma)

    def inverse_cdf(self, u: float) -> float:
        _check_u(u)
        return math.exp(self.mu + self.sigma * float(ndtri(u)))


def lognormal(row: TimeRow) -> RunTimeDistribution:
    """The default shape (C5): fitted to the row's mean and p95."""
    return Lognormal.fit(row.mean, row.p95)


def deterministic(row: TimeRow) -> RunTimeDistribution:
    return Deterministic(row.mean)


def exponential(row: TimeRow) -> RunTimeDistribution:
    return Exponential(row.mean)


@dataclass(frozen=True, slots=True)
class RunRow:
    """One time-table row as a node uses it, in seconds and joules."""

    task: str
    hw: str
    slot: SlotType
    mean: float  # Ê
    low: float  # Ê^low = p5 (14.4)
    high: float  # Ê^high = p95
    longest_step: float  # d_abort (8.6)
    energy_per_run: float  # mean dynamic energy per run (14.4)
    distribution: RunTimeDistribution
    p5_measured: bool = False  # the row's p5 came from the calibration run, not the 0.75 factor

    @property
    def power(self) -> float:
        """p = dynamic energy per run / run time (12.2)."""
        return self.energy_per_run / self.mean


@dataclass(frozen=True, slots=True)
class P5Line:
    task: str
    hw: str
    slot: SlotType
    fitted_p5: float
    measured_p5: float | None


class RunTimeTable:
    """Every (task type, hardware class, slot type) row; a missing row means cannot run there."""

    def __init__(self, rows: Mapping[tuple[str, str, SlotType], RunRow]) -> None:
        self._rows = MappingProxyType(dict(rows))

    @classmethod
    def from_config(cls, config: Config, distribution: DistributionFactory = lognormal) -> RunTimeTable:
        """The config's rows with the given shape. Energy per run is the slot's dynamic power x the mean run
        until the calibration run measures it (18, Energy: placeholders)."""
        quantiles = (config.timing.quantile_low, config.timing.quantile_high)
        if quantiles != _TABLE_QUANTILES:
            raise ConfigError(
                "timing.quantiles",
                f"must be {_TABLE_QUANTILES}: the time table holds p5 and p95 only (14.4), got {quantiles}",
            )
        rows: dict[tuple[str, str, SlotType], RunRow] = {}
        for (task, hw, slot), r in config.time_table.rows.items():
            rows[(task, hw, slot)] = RunRow(
                task=task,
                hw=hw,
                slot=slot,
                mean=r.mean,
                low=r.p5,
                high=r.p95,
                longest_step=r.longest_step,
                energy_per_run=config.hardware[hw].p_dyn_w[slot] * r.mean,
                distribution=distribution(r),
                p5_measured=r.p5_measured,
            )
        return cls(rows)

    def row(self, task: str, hw: str, slot: SlotType) -> RunRow | None:
        return self._rows.get((task, hw, slot))

    def slot_types(self, task: str, hw: str) -> tuple[SlotType, ...]:
        """The slot types of `hw` that can run `task`, GPU first."""
        return tuple(q for q in SLOT_TYPES if (task, hw, q) in self._rows)

    def p5_report(self) -> list[P5Line]:
        """Each lognormal row's fitted p5 next to its measured p5 (None for placeholder rows) (C5)."""
        lines = []
        for (task, hw, slot), row in self._rows.items():
            if isinstance(row.distribution, Lognormal):
                measured = row.low if row.p5_measured else None
                lines.append(P5Line(task, hw, slot, row.distribution.p5, measured))
        return lines

    @staticmethod
    def draw(streams: RandomStreams, row: RunRow, task_id: tuple[KeyPart, ...], attempt: int) -> float:
        """The run time of this attempt on this row: u keyed by (task_id, attempt), through the row's inverse
        CDF (CHANGES C3), in whole nanoseconds so every OS gives the same run (sim/quantize.py)."""
        return to_ns(row.distribution.inverse_cdf(streams.uniform("run_times", *task_id, attempt)))
