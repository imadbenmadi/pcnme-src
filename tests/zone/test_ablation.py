"""The 16.4 switches of the environment (PLAN M7; final-design 16.4, 9.2 R3; QUESTIONS Q47): no neighbours,
no predictor, no result forwarding, no TOF-Broker, R3 event-driven state, and the broker's message count.
Every switch is off by default (the design itself)."""

from __future__ import annotations

from typing import Any

import pytest

from pcnme.config import ConfigError, load_config
from pcnme.experiments.city import City
from pcnme.sim.clock import EventClock
from pcnme.zone.zone import start_state_loop
from tests.zone.harness import fixed_delay_config

OPTS = ("here", "n1", "n2", "n3", "n4", "cloud")


def city(scenario: str = "S2", seconds: float = 2.0, n_cars: int = 40, seed: int = 2000, **over: Any) -> City:
    return City(fixed_delay_config(**over), scenario, "greedy", seed, n_cars).run(seconds)


def test_every_switch_defaults_to_the_design() -> None:
    cfg = load_config()
    a = cfg.ablation
    assert (a.neighbours, a.predictor, a.result_forwarding, a.tof) == (True, True, True, True)
    r3 = cfg.stateplane.r3
    assert (r3.enabled, r3.delta, r3.min_interval, r3.heartbeat) == (False, 2, 0.010, 0.100)
    with pytest.raises(ConfigError, match="heartbeat"):
        fixed_delay_config(stateplane__r3__heartbeat=0.015)


def test_no_neighbours_leaves_here_and_the_cloud() -> None:
    c = city(ablation__neighbours=False)
    rows = c.log.rows
    c._decision_rows()
    decisions = rows["decisions"]
    assert decisions and all(d[f"masked_{o}"] == "empty" for d in decisions for o in OPTS[1:5])
    assert {d["action"] for d in decisions} <= {"here", "cloud"}
    for zone in c.zones.values():
        assert all(i.option in ("here", "cloud") for r in zone.records.values() for i in r.attempts.values())
    base = city()
    base._decision_rows()
    assert {d["action"] for d in base.log.rows["decisions"]} - {"here", "cloud"}  # the design uses them


def test_no_predictor_gives_exit_inf_and_stays() -> None:
    c = city(ablation__predictor=False)
    c._decision_rows()
    d = c.log.rows["decisions"]
    assert d and all(r["x_exit"] == 1.0 and r["x_next_stays"] == 1.0 for r in d)
    assert all(r[f"x_next_{o}"] == 0.0 for r in d for o in OPTS[1:5])
    base = city()
    base._decision_rows()
    assert any(r["x_next_stays"] == 0.0 for r in base.log.rows["decisions"])  # the predictor moves cars


def forwarded(**over: Any) -> tuple[object, int, str | None]:
    """The scripted car of tests/experiments/test_city: a T4 anchored and run at z1 at 11.09 s, the car handed
    to z2 at 11.1 s. Returns its outcome, the copies the car got, and the zone that delivered the first."""
    from pcnme.car.car import Task
    from pcnme.ids import TaskId
    from tests.experiments.test_city import Line, city_of

    c = city_of({"c000": Line((500.0, 300.0), (700.0, 300.0), 10.0)}, **over)
    app = c.apps[0]
    c.clock.run_until(11.09)
    t = c.cfg.task_types["T4"]
    task = Task(
        TaskId("c000", 0, 99), "c000", "T4", t.cls, t.deadline["loose"], t.l_max["loose"], t.input_bits,
        t.result_bits, c.clock.now,
    )  # type: ignore[index]  # fmt: skip
    app.car._tasks[task.task_id] = task
    c._task(app, task)
    c.clock.run_until(16.0)
    assert app.serving == "z2"
    (rec,) = (r for r in c.records if r.task.task_id == task.task_id)
    return rec.outcome, rec.deliveries, rec.delivered_by


def test_no_result_forwarding_delivers_only_through_the_anchor() -> None:
    assert forwarded() == ("on_time", 1, "z2")  # the design: z2, the car's new zone, delivers it
    outcome, deliveries, by = forwarded(ablation__result_forwarding=False)
    assert deliveries == 0 and by is None and outcome is not None and outcome.startswith("drop")
    c = city(ablation__result_forwarding=False)  # most cars stay in their anchor's zone: still delivered
    done = [r for r in c.records if r.outcome in ("on_time", "late")]
    assert len(done) > 0.9 * len(c.records) and all(r.delivered_by == r.anchor for r in done)


def test_no_tof_makes_no_boulder() -> None:
    base = city("S4")  # tight deadlines: T2 is a boulder at alpha 0.8
    assert any(r.route == "cloud" for r in base.records)
    c = city("S4", ablation__tof=False)
    assert not any(r.route == "cloud" for r in c.records)
    assert c.cfg.derived.b_wan == base.cfg.derived.b_wan  # the network stays the same


# --- R3, event-driven state (9.2) -------------------------------------------------------------------------


def publish_times(queues: list[tuple[float, int]], end: float, **r3: Any) -> list[float]:
    """The publish times of a state loop whose queue length steps to q at time t, from (t, q) pairs."""
    cfg = fixed_delay_config(**{f"stateplane__r3__{k}": v for k, v in {"enabled": True, **r3}.items()})
    clock = EventClock()
    out: list[float] = []

    def queue() -> tuple[int, ...]:
        q = 0
        for t, v in queues:
            if t <= clock.now + 1e-12:
                q = v
        return (q,)

    start_state_loop(clock, cfg.stateplane, lambda: out.append(round(clock.now, 6)), queue)
    clock.run_until(end)
    return out


def test_r3_publishes_on_a_change_of_delta_and_on_the_heartbeat() -> None:
    # delta 2: at 0 (start), 0.1 and 0.2 (heartbeat), 0.235 -> 0.24 (first tick after the queue reached 2),
    # 0.34 (heartbeat after the last publish); a change of 1 at 0.4 waits for the heartbeat 0.44
    times = publish_times([(0.235, 2), (0.4, 3)], 0.5, delta=2)
    assert times == [0.0, 0.1, 0.2, 0.24, 0.34, 0.44]
    assert publish_times([(0.05, 1)], 0.15, delta=1) == [0.0, 0.05, 0.15]


def test_without_r3_the_state_period_alone() -> None:
    cfg = fixed_delay_config(stateplane__state_period=0.05)
    clock = EventClock()
    out: list[float] = []
    start_state_loop(clock, cfg.stateplane, lambda: out.append(round(clock.now, 6)), lambda: (0,))
    clock.run_until(0.2)
    assert out == [0.0, 0.05, 0.1, 0.15, 0.2]


def test_the_broker_counts_messages_and_r3_changes_them() -> None:
    base = city(seconds=1.0)
    r3 = city(seconds=1.0, stateplane__r3__enabled=True)
    assert base.bus.published > 0 and r3.bus.published != base.bus.published
    period = city(seconds=1.0, stateplane__state_period=0.01, stateplane__stale_after=0.03)
    assert period.bus.published > base.bus.published  # 10 ms state: about ten times the state messages
