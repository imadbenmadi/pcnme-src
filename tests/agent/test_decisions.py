"""The 48 inputs at a zone's decisions (PLAN M4.1; final-design 11.1, 11.3.1, 6B.6-8, 10.1, 16.3).

Two zones with every delay and run at its mean (tests/zone/harness.py), so the numbers are by hand; the city
with scripted cars for the GPS. The inputs come only from the anchor's own memory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pcnme.agent.inputs import INPUT_NAMES
from pcnme.config import Config, load_config
from pcnme.experiments.city import City
from pcnme.ids import AttemptKey, TaskId
from pcnme.node.executor import TaskRequest
from pcnme.zone.decision import OPTIONS
from tests.experiments.test_city import Line, city_of
from tests.zone.harness import Fixed, Seq, city, city_task, fixed_delay_config

pq = pytest.importorskip("pyarrow.parquet")


def two(placement: object) -> object:
    seq = {"z1": placement, "z2": Fixed(None)}
    c = city(fixed_delay_config(), lambda z: seq[z], zone_ids=("z1", "z2"))
    c.clock.run_until(0.1)
    return c


def named(inputs: tuple[float, ...]) -> dict[str, float]:
    return dict(zip(INPUT_NAMES, inputs, strict=True))


def test_every_decision_is_kept_with_its_inputs_masks_and_action() -> None:
    # 9.1: the reply of attempt 1 (to n1 = z2) is lost: the one re-decision (here) is attempt 2.
    c = two(Seq("n1", "here"))
    z1 = c.zones["z1"]  # type: ignore[attr-defined]
    z1.lose_reply = lambda key: key.attempt == 1
    task = city_task(c, "T1")  # type: ignore[arg-type]
    z1.upload(task)
    c.clock.run_until(3.0)  # type: ignore[attr-defined]
    first, second = z1.decisions
    assert (first.kind, first.attempt, OPTIONS[first.action]) == ("first", 1, "n1")
    assert (second.kind, second.attempt, OPTIONS[second.action]) == ("redecision", 2, "here")
    assert first.task_id == second.task_id == task.task_id
    assert len(first.inputs) == len(second.inputs) == 48
    assert first.masked[1] is None and first.masked[2:5] == ("empty",) * 3
    assert first.masked[5] == "cannot_meet"  # T1 loose: about 137 ms to the cloud and back > 100 ms
    assert first.valid == (True, True, False, False, False, False)
    # the re-decision sees less time left (its smaller slack is its only sign, 11.3.5)
    assert named(second.inputs)["slack"] < named(first.inputs)["slack"]
    assert named(first.inputs)["n1_present"] == named(second.inputs)["n1_present"] == 1.0


def test_a_boulder_and_an_all_masked_task_make_no_decision() -> None:
    # 7.1.3 and 7.3: no decision is made, so nothing is kept.
    c = city(fixed_delay_config(), lambda z: Fixed(None), zone_ids=("z1",), mode="tight", scenario="S4")
    z1 = c.zones["z1"]
    c.clock.run_until(0.1)
    z1.upload(city_task(c, "T2", seq=1, mode="tight"))  # a boulder in tight mode (18)
    z1.upload(city_task(c, "T1", seq=2, mode="tight"))  # Fixed(None): as if every option were masked
    c.clock.run_until(3.0)
    assert {r.outcome for r in c.closed} >= {"drop_masked"} and z1.decisions == []


def test_the_inputs_come_from_the_state_store_not_the_neighbours_live_queue() -> None:
    # 11.1: only the zone's own memory. At 0.19 s z1 holds z2's snapshot of 0.1 s (empty). Load z2's GPU
    # directly with 6 T1s: until z2's next state message (sent at 0.2 s, at z1 7 ms later) arrives, z1's
    # input for z2 still shows the empty snapshot. At 0.2 s two run (6 ms left) and two wait: Ŵ = 14 ms.
    c = two(Seq("n1"))
    z1, z2 = c.zones["z1"], c.zones["z2"]  # type: ignore[attr-defined]
    c.clock.run_until(0.19)  # type: ignore[attr-defined]
    for i in range(6):
        key = AttemptKey(TaskId("c99", 0, i), "z2", 1)
        z2.executor.submit(TaskRequest(key, "T1", "hard", 0.1, 0.09))
    assert z2.executor.snapshot()["gpu"].n_hard > 0
    task = city_task(c, "T1")  # type: ignore[arg-type]
    ctx = z1._context(task, 0.19, 0.28)
    x = named(ctx.inputs)
    assert x["n1_wait"] == 0.0
    assert x["n1_age"] == pytest.approx(z1.store.age("z2") / 0.3)
    c.clock.run_until(0.2071)  # type: ignore[attr-defined]
    later = named(z1._context(task, 0.2071, 0.28).inputs)
    assert later["n1_wait"] == pytest.approx(0.014 / 0.1)  # z2's 0.2 s snapshot arrived


def test_the_serving_zones_own_gps_feeds_the_first_decision() -> None:
    # 2 (car registry: the cars it serves or anchors), 10.1-10.2: the car drives east at 40 m/s along
    # y = 300 from x = 480. A T2 made at 2.0 s is in at about 2.1 s: z1 has just become its anchor (its
    # car/{id}/gps subscription has nothing yet), but as the serving zone it holds the car's GPS of the last
    # 100 ms tick. T_exit = (650 - x) / 40 s; next zone z2.
    from pcnme.car.car import Task

    c = city_of({"c000": Line((480.0, 300.0), (1100.0, 300.0), 40.0)})
    c.clock.run_until(2.0)
    t = c.cfg.task_types["T2"]
    task = Task(
        TaskId("c000", 0, 1),
        "c000",
        "T2",
        t.cls,
        t.deadline["loose"],
        t.l_max["loose"],
        t.input_bits,
        t.result_bits,
        c.clock.now,
    )  # type: ignore[index]
    c.apps[0].car._tasks[task.task_id] = task
    c._task(c.apps[0], task)
    c.clock.run_until(2.3)
    z1 = c.zones["z1"]
    (d,) = z1.decisions
    tick = round(d.time * 1e6) // 100_000 / 10  # the last GPS tick at or before the decision
    x_gps = 480.0 + 40.0 * tick
    x = named(d.inputs)
    assert x["exit"] == pytest.approx((650.0 - x_gps) / 40.0 / 3.0)
    k = z1.neighbours.positions.index("z2") + 1
    assert x[f"next_n{k}"] == 1.0 and x["next_stays"] == 0.0


def test_the_registry_follows_the_serving_and_the_anchor_roles() -> None:
    # 9.6: a served car stays in the registry when its last anchored task closes; it leaves when the zone
    # neither serves nor anchors it.
    c = city_of({"c000": Line((300.0, 300.0), (300.0, 300.0), 10.0)})
    c.clock.run_until(0.15)
    z1 = c.zones["z1"]
    assert "c000" in z1.registry  # served, no task yet
    z1.detach("c000")
    assert "c000" not in z1.registry


@pytest.fixture(scope="module")
def run_dir(default_config_path: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    cfg: Config = load_config(default_config_path)
    return City(cfg, "S2", "greedy", 2000, n_cars=40).run(2.0).write(tmp_path_factory.mktemp("results"))


def test_the_run_logs_every_decision_with_its_48_inputs(run_dir: Path) -> None:
    # 16.3, PLAN M4.1: decisions.parquet, one row per decision; x_<input> in the order of 11.1.
    t = pq.read_table(run_dir / "decisions.parquet")
    xs = [n for n in t.column_names if n.startswith("x_")]
    assert xs == [f"x_{n}" for n in INPUT_NAMES]
    assert {f"masked_{o}" for o in OPTIONS} <= set(t.column_names)
    rows = t.to_pylist()
    assert rows and {r["kind"] for r in rows} <= {"first", "redecision", "retry"}
    assert all(r[f"masked_{r['action']}"] == "" for r in rows)  # the action taken was open
    tasks = pq.read_table(run_dir / "tasks.parquet").to_pylist()
    pebbles = [r for r in tasks if not r["boulder"] and r["outcome"] != "drop_masked"]
    assert len({(r["anchor"], r["task_id"]) for r in rows}) == len(pebbles)


def test_a_parked_car_stays() -> None:
    # 10.2.1, C8: speed 0: T_exit infinite, exit = 1, and the car stays in its zone (Q34): next_stays.
    from pcnme.car.car import Task

    c = city_of({"c000": Line((300.0, 300.0), (300.0, 300.0), 10.0)})
    c.clock.run_until(1.0)
    t = c.cfg.task_types["T1"]
    task = Task(
        TaskId("c000", 0, 1),
        "c000",
        "T1",
        t.cls,
        t.deadline["loose"],
        t.l_max["loose"],
        t.input_bits,
        t.result_bits,
        c.clock.now,
    )  # type: ignore[index]
    c.apps[0].car._tasks[task.task_id] = task
    c._task(c.apps[0], task)
    c.clock.run_until(1.2)
    (d,) = c.zones["z1"].decisions
    x = named(d.inputs)
    assert x["exit"] == 1.0 and x["next_stays"] == 1.0
