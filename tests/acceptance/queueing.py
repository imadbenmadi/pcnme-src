"""M/M/c runs of the real executor, for the M1 acceptance tests (final-design 8.7, 16.5)."""

from __future__ import annotations

import math

import numpy as np

from pcnme.config import HardwareClass
from pcnme.node.executor import AttemptKey, Executor, Report, TaskRequest
from pcnme.node.timetable import Deterministic, Exponential, RunRow, RunTimeTable
from pcnme.sim.clock import EventClock
from pcnme.sim.rng import RandomStreams

MEAN_RUN = 0.010  # s: mu = 100 tasks/s per slot
SPLIT = {"training": range(2000, 3000)}
NEVER = 1e12  # a budget no run reaches: no skip, no abort, never late


def erlang_c_wait(c: int, lam: float, mu: float) -> float:
    """Mean wait in queue of M/M/c (Erlang C): C(c, a) / (c mu - lam), a = lam / mu."""
    a = lam / mu
    rho = a / c
    top = a**c / math.factorial(c) / (1 - rho)
    p_wait = top / (sum(a**k / math.factorial(k) for k in range(c)) + top)
    return p_wait / (c * mu - lam)


def mmc_waits(c: int, rho: float, n: int, seed: int, exponential: bool = True) -> np.ndarray:
    """n tasks through one slot type with c slots: one hard class, equal budgets (EDF = FIFO, 8.7), Poisson
    arrivals at lam = rho c mu from the arrivals stream, run times from the keyed run_times stream. Returns
    each task's wait (start - arrival), in arrival order."""
    hw = HardwareClass(name="M", c_gpu=0, c_cpu=c, queue={"cpu": 10**9}, p_dyn_w={"cpu": 1.0}, p_idle_w=0.0)
    shape = Exponential(MEAN_RUN) if exponential else Deterministic(MEAN_RUN)
    row = RunRow("M", "M", "cpu", MEAN_RUN, MEAN_RUN, MEAN_RUN, MEAN_RUN, MEAN_RUN, shape)
    clock = EventClock()
    starts: dict[AttemptKey, float] = {}

    def on_report(r: Report) -> None:
        assert r.outcome == "done" and r.start is not None
        starts[r.key] = r.start

    streams = RandomStreams(seed, "training", SPLIT)
    ex = Executor(
        clock,
        streams,
        hw,
        RunTimeTable({("M", "M", "cpu"): row}),
        reserve_hard=0,
        pue=1.0,
        a_max=NEVER,
        promotion_estimate="mean",  # one class: promotion never applies
        t_keep=1.0,
        on_report=on_report,
    )
    lam = rho * c / MEAN_RUN
    gaps = streams.generator("arrivals", "source").exponential(1 / lam, n)
    arrivals = np.cumsum(gaps)
    keys = [AttemptKey(("src", 0, i), "z1", 1) for i in range(n)]
    for i, t in enumerate(arrivals):
        clock.run_until(float(t))
        ex.submit(TaskRequest(keys[i], "M", "hard", 0.0, NEVER))
    clock.run_until(float(arrivals[-1]) + 1e6)
    return np.array([starts[k] for k in keys]) - arrivals
