"""M1 acceptance tests (PLAN M1; final-design 8.7, 16.5).

- M/M/1 at rho 0.5 and 0.8 and M/M/c with c = 2 and 4 against Erlang C, within 3 %, on the real executor
  (one slot type, one class, FIFO order, Poisson arrivals, exponential run times). Tolerance checked as a
  fair margin on three seeds: at 200 000 tasks rho 0.5 stays within 0.8 %, but rho 0.8 reached -2.3 %
  (heavy-tailed waits), so rho 0.8 uses 1 000 000 tasks (within 1.4 % on three seeds).
- One unit test per scheduler rule of 8.4 (tests/node/test_scheduler.py) and per row of the 8.6 table
  (tests/node/test_drops.py, with the arrival rows in test_executor.py): listed in RULE_TESTS below and
  checked to exist.
- Ŵ: 0 with a free slot, the formula otherwise, c_q = 1 exact in the mean (tests/node/test_state.py).
- Determinism: the same seed gives the same event-log hash; a different seed a different one.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import numpy as np
import pytest

from pcnme.config import Config, load_config
from pcnme.node.executor import AttemptKey, Executor, Report, TaskRequest, TraceEntry
from pcnme.node.timetable import RunTimeTable, exponential
from pcnme.sim.clock import EventClock
from pcnme.sim.rng import RandomStreams
from tests.acceptance.queueing import MEAN_RUN, erlang_c_wait, mmc_waits

WARM_UP = 10_000  # tasks dropped from the start of each run (the queue starts empty)


# --- M/M/1 and M/M/c (8.7) ----------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    ("c", "rho", "n"),
    [
        (1, 0.5, 200_000),
        (1, 0.8, 1_000_000),
        (2, 0.5, 200_000),
        (2, 0.8, 1_000_000),
        (4, 0.5, 200_000),
        (4, 0.8, 1_000_000),
    ],
)
def test_mean_wait_matches_erlang_c(c: int, rho: float, n: int) -> None:
    waits = mmc_waits(c, rho, n, seed=2000)[WARM_UP:]
    lam, mu = rho * c / MEAN_RUN, 1 / MEAN_RUN
    expected = erlang_c_wait(c, lam, mu)
    if c == 1:
        assert expected == pytest.approx(lam / (mu * (mu - lam)), rel=1e-12)  # M/M/1: λ / (μ (μ − λ))
    assert waits.mean() == pytest.approx(expected, rel=0.03)


def test_erlang_c_formula_by_hand() -> None:
    # c = 2, a = λ/μ = 1.6 (ρ = 0.8): C = (1.6² / 2 / 0.2) / (1 + 1.6 + 6.4) = 6.4 / 9 = 0.7111;
    # Wq = C / (2 μ − λ) = 0.7111 / (200 − 160) = 17.78 ms.
    assert erlang_c_wait(2, 160.0, 100.0) == pytest.approx((6.4 / 9) / 40, rel=1e-12)


def test_a_short_mmc_run_is_in_the_right_place() -> None:
    # A fast smoke check of the same pipeline for the default (non-slow) suite: c = 2, ρ = 0.5, 20 000 tasks,
    # within 10 %.
    waits = mmc_waits(2, 0.5, 20_000, seed=2000)[2_000:]
    assert waits.mean() == pytest.approx(erlang_c_wait(2, 100.0, 100.0), rel=0.10)


# --- One unit test per rule of 8.4 and per row of 8.6 -------------------------------------------------------

RULE_TESTS = {
    "8.4 step 1 skip hopeless heads": [
        "tests/node/test_scheduler.py::test_a_hard_head_is_skipped_when_it_can_no_longer_finish",
        "tests/node/test_scheduler.py::test_a_soft_head_is_skipped_only_past_deadline_plus_l_max",
    ],
    "8.4 step 2 soft promotion": [
        "tests/node/test_scheduler.py::test_a_soft_task_that_waited_a_max_runs_before_hard_tasks_that_can_spare_it",
        "tests/node/test_scheduler.py::test_no_promotion_when_a_queued_hard_task_would_miss",
    ],
    "8.4 step 3 first hard": [
        "tests/node/test_executor.py::test_a_free_slot_takes_hard_before_soft_then_earliest_deadline"
    ],
    "8.4 step 4 first soft": ["tests/node/test_scheduler.py::test_no_promotion_before_a_max"],
    "8.4 step 5 work stealing": [
        "tests/node/test_scheduler.py::test_an_empty_queue_steals_a_task_that_finishes_earlier_here"
    ],
    "8.4 step 6 idle": ["tests/node/test_scheduler.py::test_a_slot_with_nothing_to_take_idles"],
    "8.6 on arrival": [
        "tests/node/test_executor.py::test_a_hard_task_is_refused_full_when_every_queue_is_full",
        "tests/node/test_executor.py::test_a_soft_task_is_refused_late_only_past_deadline_plus_l_max",
    ],
    "8.6 head of the queue": ["tests/node/test_scheduler.py::test_skipping_continues_until_a_head_can_run"],
    "8.6 while running": [
        "tests/node/test_drops.py::test_a_hard_task_running_past_d_k_is_aborted_with_its_tail",
        "tests/node/test_drops.py::test_a_soft_task_runs_late_until_d_k_plus_l_max",
    ],
    "8.6 anchor cancel": [
        "tests/node/test_drops.py::test_cancel_removes_a_queued_task",
        "tests/node/test_drops.py::test_cancel_aborts_a_running_task_with_its_tail",
    ],
}


@pytest.mark.parametrize("rule", sorted(RULE_TESTS))
def test_every_rule_of_8_4_and_8_6_has_a_test(rule: str, repo_root: Path) -> None:
    for node_id in RULE_TESTS[rule]:
        path, name = node_id.split("::")
        module = importlib.import_module(path.removesuffix(".py").replace("/", "."))
        assert callable(getattr(module, name, None)), f"{rule}: {node_id} is missing"
        assert (repo_root / path).is_file()


# --- Determinism: the same seed gives the same event-log hash -----------------------------------------------


def event_log_hash(cfg: Config, seed: int, seconds: float = 5.0) -> str:
    """A realistic run on the config's F-GPU node: the task mix at S2's per-zone rate (74 tasks/s), loose
    deadlines, exponential run times (a test shape; Q7 is open). The hash covers every trace entry and every
    report, floats by their exact repr."""
    clock = EventClock()
    streams = RandomStreams.from_config(cfg, seed, "training")
    trace: list[TraceEntry] = []
    reports: list[Report] = []
    ex = Executor.from_config(
        clock,
        streams,
        cfg,
        "F-GPU",
        RunTimeTable.from_config(cfg, exponential),
        promotion_estimate="mean",
        on_report=reports.append,
        trace=trace.append,
    )
    types = list(cfg.task_types.values())
    shares = np.array([t.share for t in types])
    arrivals = streams.generator("arrivals", "z1")
    rate = cfg.derived.arrival_rate_per_zone["S2"]
    t, seq = 0.0, 0
    while True:
        t += float(arrivals.exponential(1 / rate))
        if t > seconds:
            break
        clock.run_until(t)
        task = types[int(arrivals.choice(len(types), p=shares))]
        seq += 1
        d = task.deadline["loose"]
        ex.submit(
            TaskRequest(AttemptKey(("c01", 0, seq), "z1", 1), task.name, task.cls, task.l_max["loose"], d)
        )
    clock.run_until(seconds + 60.0)
    digest = hashlib.sha256()
    for entry in trace:
        digest.update(repr(entry).encode())
    for r in reports:
        digest.update(repr(r).encode())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


def test_the_same_seed_gives_the_same_event_log_hash(cfg: Config) -> None:
    assert event_log_hash(cfg, 2000) == event_log_hash(cfg, 2000)


def test_a_different_seed_gives_a_different_event_log_hash(cfg: Config) -> None:
    assert event_log_hash(cfg, 2000) != event_log_hash(cfg, 2001)


def test_the_event_log_hash_is_pinned(cfg: Config) -> None:
    # Regression pin, computed on Windows; CI checks it on Linux too, so it also proves the run is identical
    # across platforms. Change it only on purpose, in a PR that says why.
    assert event_log_hash(cfg, 2000) == GOLDEN_HASH


GOLDEN_HASH = "059bd63a4d7390c3429bdf447c97641d56c6bc26c574b0df0415ba6b9304ca48"  # 28 Oct, Windows
