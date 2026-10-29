"""Cars and tasks (PLAN M2.2; final-design 6A, 9.5, 14.1-14.3, 18; QUESTIONS Q12-Q14)."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from pcnme.car.car import Car, Confirm, Result, Task
from pcnme.car.placement import standin_uniform_disk
from pcnme.config import Config, config_from_mapping, load_config, load_yaml
from pcnme.ids import TaskId
from pcnme.sim.clock import EventClock
from pcnme.sim.rng import RandomStreams


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


def make_car(
    cfg: Config, scenario: str = "S2", car_id: str = "c01", seed: int = 2000, fleet: int | None = None
) -> tuple[Car, list[Task]]:
    clock = EventClock()
    tasks: list[Task] = []
    streams = RandomStreams.from_config(cfg, seed, "training")
    car = Car(
        car_id,
        cfg,
        cfg.scenarios[scenario],
        streams,
        clock,
        position=(300.0, 300.0),
        on_task=tasks.append,
        fleet=cfg.cars.count[0] if fleet is None else fleet,
    )
    return car, tasks


def first_task(cfg: Config) -> tuple[Car, Task]:
    """A car stopped just after it created its first task (within 1 ms)."""
    car, tasks = make_car(cfg)
    car.start(until=1e6)
    while not tasks:
        car.clock.run_until(car.clock.now + 0.001)
    return car, tasks[0]


def run_car(cfg: Config, scenario: str, seconds: float, seed: int = 2000, car_id: str = "c01") -> list[Task]:
    car, tasks = make_car(cfg, scenario, car_id, seed)
    car.start(until=seconds)
    car.clock.run_until(seconds)
    return tasks


# --- C1-C4: tasks -------------------------------------------------------------------------------------------


def test_a_task_carries_its_type_class_deadline_sizes_and_creation_time(cfg: Config) -> None:
    task = run_car(cfg, "S2", 5.0)[0]
    t = cfg.task_types[task.task_type]
    assert task.task_id == TaskId("c01", 0, 1)
    assert (task.cls, task.deadline, task.l_max) == (t.cls, t.deadline["loose"], t.l_max["loose"])  # S2 loose
    assert (task.input_bits, task.result_bits) == (t.input_bits, t.result_bits)
    assert 0.0 < task.created <= 5.0


def test_tight_scenarios_give_tight_deadlines_and_t2s_is_1_1_e_strong(cfg: Config) -> None:
    tasks = run_car(cfg, "S4", 200.0)
    t2 = [t for t in tasks if t.task_type == "T2"]
    assert t2 and all(t.deadline == pytest.approx(0.792) for t in t2)
    assert all(t.deadline == 0.050 for t in tasks if t.task_type == "T1")


def test_the_mix_follows_the_shares_by_count(cfg: Config) -> None:
    tasks = [t for s in range(10) for t in run_car(cfg, "S3", 400.0, car_id=f"c{s:02d}")]
    counts = Counter(t.task_type for t in tasks)
    n = len(tasks)
    expected = {name: tt.share for name, tt in cfg.task_types.items()}
    observed = [counts[k] for k in expected]
    assert stats.chisquare(observed, [expected[k] * n for k in expected]).pvalue > 1e-3


def test_arrivals_are_poisson_at_lambda_car_standin(cfg: Config) -> None:
    # QUESTIONS Q12 stand-in: exponential gaps at lambda_car (S2: 74.26 tasks/s per zone / 50 cars = 1.485
    # /s).
    tasks = run_car(cfg, "S2", 4000.0)
    lam = cfg.arrival_rate_per_car("S2", cfg.cars.count[0])
    gaps = np.diff([0.0] + [t.created for t in tasks])
    assert len(tasks) / 4000.0 == pytest.approx(lam, rel=0.03)
    assert stats.kstest(gaps, "expon", args=(0, 1 / lam)).pvalue > 1e-3


@pytest.mark.parametrize("fleet", [200, 1000])
def test_the_rate_comes_from_the_runs_fleet_so_a_zone_gets_the_same_load(cfg: Config, fleet: int) -> None:
    # 18 (Workload): lambda_car = per-zone rate / cars per zone. A zone's share of a 1000-car city (250 cars)
    # makes as many tasks per second as its share of a 200-car city (50 cars): 74.26 /s in S2.
    per_zone = cfg.derived.arrival_rate_per_zone["S2"]
    clock = EventClock()
    streams = RandomStreams.from_config(cfg, 2000, "training")
    tasks: list[Task] = []
    seconds = 100.0
    for i in range(fleet // len(cfg.layout.active_zones)):
        car = Car(
            f"c{i:03d}", cfg, cfg.scenarios["S2"], streams, clock, (0.0, 0.0), tasks.append, fleet=fleet
        )
        car.start(until=seconds)
    clock.run_until(seconds)
    assert len(tasks) / seconds == pytest.approx(per_zone, rel=0.03)  # 7 426 tasks: sd 1.2 %


def test_arrival_gaps_are_whole_nanoseconds(cfg: Config) -> None:
    created = [t.created for t in run_car(cfg, "S2", 50.0)]
    assert all(round(c * 1e9) == pytest.approx(c * 1e9, abs=1e-3) for c in created)


def test_the_upload_starts_after_the_encode_time_and_counts_it_once(default_config_path: Path) -> None:
    # 14.3, QUESTIONS Q14: L holds the encode time. The upload starts at creation + encode, and B_car = D -
    # time since creation then already holds it: with encode 4 ms, B_car = D - 4 ms (not D - 8 ms).
    data = load_yaml(default_config_path)
    data["timing"]["encode"] = 0.004
    cfg4 = config_from_mapping(data)
    clock = EventClock()
    uploads: list[tuple[float, float]] = []
    car: Car

    def on_task(task: Task) -> None:
        uploads.append((clock.now - task.created, car.budget(task, clock.now) - task.deadline))

    streams = RandomStreams.from_config(cfg4, 2000, "training")
    car = Car("c01", cfg4, cfg4.scenarios["S2"], streams, clock, (0.0, 0.0), on_task, fleet=200)
    car.start(until=5.0)
    clock.run_until(6.0)
    assert uploads
    assert all(wait == pytest.approx(0.004, abs=1e-12) for wait, _ in uploads)
    assert all(b == pytest.approx(-0.004, abs=1e-12) for _, b in uploads)


def test_task_ids_count_up_per_car(cfg: Config) -> None:
    tasks = run_car(cfg, "S2", 10.0)
    assert [t.task_id.seq for t in tasks] == list(range(1, len(tasks) + 1))


def test_two_cars_and_the_same_seed_give_their_own_and_repeatable_streams(cfg: Config) -> None:
    a = [(t.created, t.task_type) for t in run_car(cfg, "S2", 30.0, car_id="c01")]
    b = [(t.created, t.task_type) for t in run_car(cfg, "S2", 30.0, car_id="c02")]
    assert a != b
    assert a == [(t.created, t.task_type) for t in run_car(cfg, "S2", 30.0, car_id="c01")]


# --- C5, C6: the car's clock and its budget (5.4, 6A) -------------------------------------------------------


def test_the_budget_at_upload_is_d_minus_the_time_since_creation(cfg: Config) -> None:
    car, task = first_task(cfg)
    assert car.budget(task, now=task.created) == pytest.approx(task.deadline)  # encode time 0 (Q14 stand-in)
    assert car.budget(task, now=task.created + 0.030) == pytest.approx(task.deadline - 0.030)


# --- C7: the first result wins; every copy is confirmed (9.5) -----------------------------------------------


def test_the_car_keeps_the_first_result_and_confirms_every_copy(cfg: Config) -> None:
    car, task = first_task(cfg)
    t0 = task.created
    first = car.receive(Result(task.task_id, "z1", 1), now=t0 + 0.040)
    second = car.receive(Result(task.task_id, "z2", 1), now=t0 + 0.060)
    assert first == Confirm(task.task_id, "z1", 1, pytest.approx(0.040))  # L = arrival - creation
    assert second == first  # the confirm names the copy it kept, not the new one
    assert car.kept(task.task_id) == ("z1", 1)


def test_results_for_unknown_or_expired_tasks_are_ignored(cfg: Config) -> None:
    car, task = first_task(cfg)
    assert car.receive(Result(TaskId("c99", 0, 1), "z1", 1), now=car.clock.now) is None
    assert car.receive(Result(task.task_id, "z1", 1), now=car.clock.now) is not None  # still kept
    car.clock.run_until(task.created + task.deadline + task.l_max / 2)  # past D, before D + L_max: kept
    assert car.receive(Result(task.task_id, "z1", 1), now=car.clock.now) is not None
    car.clock.run_until(task.created + task.deadline + task.l_max + 1.0)  # the id is kept until D + L_max
    assert car.receive(Result(task.task_id, "z1", 1), now=car.clock.now) is None


# --- C8: stand-ins ------------------------------------------------------------------------------------------


def test_the_placement_standin_is_uniform_over_the_zone_disk(cfg: Config) -> None:
    # QUESTIONS Q13 stand-in: P(distance <= r) = (r / R_z)^2 for a uniform disk.
    streams = RandomStreams.from_config(cfg, 2000, "training")
    centre, radius = (300.0, 300.0), cfg.layout.zone_radius_m
    points = [standin_uniform_disk(streams, f"c{i}", centre, radius) for i in range(4000)]
    d = np.array([math.dist(p, centre) for p in points])
    assert d.max() <= radius
    assert stats.kstest(d / radius, lambda x: np.clip(x, 0, 1) ** 2).pvalue > 1e-3
    again = RandomStreams.from_config(cfg, 2000, "training")
    assert standin_uniform_disk(again, "c7", centre, radius) == points[7]  # same seed, same car: same point
