"""A one-zone system for tests: the config's zone, the cloud, one car at a chosen distance."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pcnme.bus.bus import Bus
from pcnme.car.car import Car, Task
from pcnme.config import Config, config_from_mapping, load_yaml
from pcnme.net.pipe import FifoPipe
from pcnme.node.timetable import RunTimeTable, deterministic
from pcnme.sim.clock import EventClock
from pcnme.sim.rng import RandomStreams
from pcnme.zone.zone import Cloud, DecisionContext, Option, TaskRecord, Zone

CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def fixed_delay_config(**overrides: Any) -> Config:
    """The default config with every random delay at its mean: radio 3 ms, backhaul 3 ms, WAN 62.5 ms."""
    data: dict[str, Any] = copy.deepcopy(load_yaml(CONFIG))
    data["radio"]["delay"] = [0.003, 0.003]
    data["radio"]["downlink_delay"] = [0.003, 0.003]
    data["links"]["backhaul_delay"] = [0.003, 0.003]
    data["links"]["wan_delay"] = [0.0625, 0.0625]
    for dotted, value in overrides.items():
        *head, last = dotted.split("__")
        node = data
        for part in head:
            node = node[part]
        node[last] = value
    return config_from_mapping(data)


class Fixed:
    """A placement that always picks one option (and counts its calls)."""

    def __init__(self, option: Option | None) -> None:
        self.option = option
        self.calls = 0

    def choose(self, ctx: DecisionContext) -> Option | None:
        self.calls += 1
        return self.option


@dataclass
class System:
    cfg: Config
    clock: EventClock
    zone: Zone
    car: Car
    closed: list[TaskRecord]
    bus: Bus


def one_zone(
    cfg: Config,
    placement: Any,
    mode: str = "loose",
    distance: float = 300.0,
    shape: Callable[..., Any] = deterministic,
    seed: int = 2000,
    hw_name: str = "F-GPU",
) -> System:
    clock = EventClock()
    streams = RandomStreams.from_config(cfg, seed, "training")
    table = RunTimeTable.from_config(cfg, shape)
    wan = FifoPipe(cfg.derived.b_wan)
    bus = Bus(clock, streams, cfg.links.broker_extra_delay)
    cloud = Cloud(clock, streams, cfg, table, bus)
    closed: list[TaskRecord] = []
    zone = Zone(
        "z1",
        clock,
        streams,
        cfg,
        table,
        wan,
        cloud,
        placement,
        mode=mode,
        hw_name=hw_name,
        bus=bus,
        on_close=closed.append,
    )  # type: ignore[arg-type]
    scenario = cfg.scenarios["S2" if mode == "loose" else "S4"]
    car = Car(
        "c01",
        cfg,
        scenario,
        streams,
        clock,
        position=(0.0, 0.0),
        on_task=zone.upload,
        fleet=cfg.cars.count[0],
    )
    zone.add_car(car, distance)
    return System(cfg, clock, zone, car, closed, bus)


def task_now(s: System, kind: str, seq: int = 1, mode: str = "loose") -> Task:
    """A task of `kind` created now on the system's car (no random arrival)."""
    from pcnme.ids import TaskId

    t = s.cfg.task_types[kind]
    task = Task(
        TaskId("c01", 0, seq),
        "c01",
        kind,
        t.cls,
        t.deadline[mode],
        t.l_max[mode],
        t.input_bits,  # type: ignore[index]
        t.result_bits,
        s.clock.now,
    )
    s.car._tasks[task.task_id] = task  # the car knows its own task (as Car._create does)
    s.clock.schedule(task.created + task.deadline + task.l_max, lambda: s.car._forget(task.task_id))
    return task


@dataclass
class City:
    cfg: Config
    clock: EventClock
    bus: Bus
    cloud: Cloud
    zones: dict[str, Zone]
    closed: list[TaskRecord]
    car: Car


def city(
    cfg: Config,
    placement: Callable[[str], Any],
    zone_ids: tuple[str, ...] = ("z1", "z2", "z3", "z4"),
    mode: str = "loose",
    seed: int = 2000,
    scenario: str = "S2",
    shape: Callable[..., Any] = deterministic,
) -> City:
    """Zones on one bus and one WAN with the cloud; neighbours from the layout (7.5); no cars."""
    from pcnme.zone.layout import Site

    clock = EventClock()
    streams = RandomStreams.from_config(cfg, seed, "training")
    bus = Bus(clock, streams, cfg.links.broker_extra_delay)
    table = RunTimeTable.from_config(cfg, shape)
    wan = FifoPipe(cfg.derived.b_wan)
    cloud = Cloud(clock, streams, cfg, table, bus)
    r = cfg.layout.zone_radius_m
    sites = {z: Site(z, *cfg.layout.zone_positions_m[z], r) for z in zone_ids}
    fog = cfg.scenarios[scenario].fog
    closed: list[TaskRecord] = []
    ledger: dict[tuple[str, object], TaskRecord] = {}
    zones = {
        z: Zone(
            z,
            clock,
            streams,
            cfg,
            table,
            wan,
            cloud,
            placement(z),
            mode,
            hw_name=fog[z],
            bus=bus,  # type: ignore[arg-type]
            start_sites=tuple(s for o, s in sites.items() if o != z),
            fog_classes=fog,
            ledger=ledger,
            on_close=closed.append,
        )
        for z in zone_ids
    }
    for zone in zones.values():
        zone.peers = zones  # direct calls (5.3)
    car = Car(
        "c01", cfg, cfg.scenarios[scenario], streams, clock, (0.0, 0.0), zones[zone_ids[0]].upload, fleet=200
    )
    zones[zone_ids[0]].add_car(car, 300.0)  # one car in the first zone, 300 m from its access point
    return City(cfg, clock, bus, cloud, zones, closed, car)


class Seq:
    """A placement that answers from a list, one per call (then None), and counts its calls."""

    def __init__(self, *options: Option | None) -> None:
        self.options = list(options)
        self.calls = 0

    def choose(self, ctx: DecisionContext) -> Option | None:
        self.calls += 1
        return self.options.pop(0) if self.options else None


def city_task(c: City, kind: str, seq: int = 1, mode: str = "loose") -> Task:
    """A task created now on the city's car (in the first zone)."""
    from pcnme.ids import TaskId

    t = c.cfg.task_types[kind]
    task = Task(
        TaskId("c01", 0, seq),
        "c01",
        kind,
        t.cls,
        t.deadline[mode],
        t.l_max[mode],
        t.input_bits,
        t.result_bits,
        c.clock.now,
    )  # type: ignore[index]
    c.car._tasks[task.task_id] = task
    c.clock.schedule(task.created + task.deadline + task.l_max, lambda: c.car._forget(task.task_id))
    return task
