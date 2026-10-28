"""Run one zone (with the cloud) for a while and summarise it (PLAN M2 acceptance).

One zone z1 with its cars, the cloud and the shared WAN pipe, in fast mode. The cars stay where the Q13
stand-in puts them (no mobility before M3.8). Used by the M2 acceptance tests and the sanity table.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pcnme.car.car import Car
from pcnme.car.placement import standin_uniform_disk
from pcnme.config import Config
from pcnme.energy import EnergyModel
from pcnme.methods.simple import Greedy, Random
from pcnme.net.pipe import FifoPipe
from pcnme.node.timetable import RunTimeTable, lognormal
from pcnme.sim.clock import EventClock
from pcnme.sim.rng import RandomStreams
from pcnme.zone.estimates import PathEstimates, ZeroWanWait
from pcnme.zone.zone import Cloud, Placement, TaskRecord, Zone

DRAIN = 30.0  # s after the last arrival, so every task can close


@dataclass
class Run:
    cfg: Config
    scenario: str
    seconds: float
    records: list[TaskRecord] = field(default_factory=list)


def run_one_zone(
    cfg: Config,
    scenario: str,
    method: str | Callable[[Config, RandomStreams], Placement],
    seconds: float,
    seed: int,
    n_cars: int | None = None,
    shape: Callable[..., Any] = lognormal,
) -> Run:
    """`n_cars` defaults to the zone's share of the smallest city fleet: cars.count[0] / 4 active zones."""
    clock = EventClock()
    streams = RandomStreams.from_config(cfg, seed, "training")
    table = RunTimeTable.from_config(cfg, shape)
    wan = FifoPipe(cfg.derived.b_wan)
    cloud = Cloud(clock, streams, cfg, table)
    est = PathEstimates(cfg, wan_wait=ZeroWanWait())
    if method == "greedy":
        placement: Placement = Greedy(cfg)
    elif method == "random":
        placement = Random(cfg, streams, "z1")
    else:
        placement = method(cfg, streams)  # type: ignore[operator]
    s = cfg.scenarios[scenario]
    run = Run(cfg, scenario, seconds)
    zone = Zone(
        "z1",
        clock,
        streams,
        cfg,
        table,
        wan,
        cloud,
        est,
        placement,
        mode=s.deadlines,
        hw_name=s.fog["z1"],
        on_close=run.records.append,
    )
    centre = cfg.layout.zone_positions_m["z1"]
    count = n_cars if n_cars is not None else cfg.cars.count[0] // len(cfg.layout.active_zones)
    for i in range(count):
        car_id = f"c{i:03d}"
        pos = standin_uniform_disk(streams, car_id, centre, cfg.layout.zone_radius_m)  # STAND-IN (Q13)
        car = Car(car_id, cfg, s, streams, clock, position=pos, on_task=zone.upload)
        zone.add_car(car, math.dist(pos, centre))
        car.start(until=seconds)
    clock.run_until(seconds + DRAIN)
    run.records.sort(key=lambda r: r.task.task_id)
    return run


def check_invariants(run: Run) -> list[str]:
    """Every task closes exactly once; L = the sum of its parts; no hard task counts as on time after D;
    energy >= 0. Returns the violations (empty when all hold)."""
    bad: list[str] = []
    ids = [r.task.task_id for r in run.records]
    if len(ids) != len(set(ids)):
        bad.append("a task closed twice")
    em = EnergyModel(run.cfg)
    for r in run.records:
        if r.outcome is None:
            bad.append(f"{r.task.task_id} never closed")
        if r.confirm is not None:
            parts = sum(r.parts.values())
            if r.confirm.latency < parts - 1e-9:
                bad.append(f"{r.task.task_id}: L {r.confirm.latency} < parts {parts}")
            if r.task.cls == "hard" and r.outcome == "on_time" and r.confirm.latency > r.task.deadline:
                bad.append(f"{r.task.task_id}: hard on time after D")
        e = em.of_record(r)
        if min(e.forwarding, e.execution, e.ret) < 0:
            bad.append(f"{r.task.task_id}: negative energy")
    return bad


def summarise(run: Run) -> dict[str, dict[str, float]]:
    """Per task type: arrivals, here / cloud / dropped, refusals by reason, on-time rate, mean latency of
    delivered results, mean ê. Plus the measured ρ_fog and TI_WAN of the zone."""
    em = EnergyModel(run.cfg)
    rows: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    lat: dict[str, list[float]] = defaultdict(list)
    ehat: dict[str, list[float]] = defaultdict(list)
    for r in run.records:
        row = rows[r.task.task_type]
        row["arrivals"] += 1
        row["here"] += r.option == "here"
        row["cloud"] += r.option == "cloud"
        dropped = (r.outcome or "").startswith("drop")
        row["dropped"] += dropped
        if r.reply in ("full", "late"):
            row[f"refused_{r.reply}"] += 1
        row["on_time"] += r.outcome == "on_time"
        row["late"] += r.outcome == "late"
        if r.confirm is not None:
            lat[r.task.task_type].append(r.confirm.latency)
        if r.option is not None:
            ehat[r.task.task_type].append(em.e_hat(r.task.task_type, em.of_record(r).total))
    for t, row in rows.items():
        row["on_time_rate"] = row["on_time"] / row["arrivals"]
        row["mean_latency_ms"] = 1e3 * sum(lat[t]) / len(lat[t]) if lat[t] else float("nan")
        row["mean_e_hat"] = sum(ehat[t]) / len(ehat[t]) if ehat[t] else float("nan")
    cfg = run.cfg
    n = len(run.records)
    mix_mean_gpu = cfg.derived.mean_gpu_run
    c_gpu = cfg.hardware["F-GPU"].c_gpu
    eligible = sum(
        r.task.input_bits
        for r in run.records
        if r.task.cls == "soft" or cfg.is_boulder(r.task.task_type, cfg.scenarios[run.scenario].deadlines)
    )
    sent = sum(r.task.input_bits for r in run.records if r.option == "cloud")
    rows["_zone"] = {
        "rho_fog": n / run.seconds * mix_mean_gpu / c_gpu,
        "rho_fog_target": cfg.scenarios[run.scenario].rho_fog,
        "ti_wan_eligible": eligible / run.seconds / cfg.derived.b_wan,
        "ti_wan_target_one_zone": cfg.derived.ti_wan[run.scenario] / len(cfg.layout.active_zones),
        "ti_wan_sent": sent / run.seconds / cfg.derived.b_wan,
        "radio_ti": sum(r.task.input_bits / r.rate for r in run.records) / run.seconds,
        "hard_upload_late": sum(
            1
            for r in run.records
            if r.task.cls == "hard" and r.d_a <= r.t_rx  # no time left on arrival
        ),
        "outcomes": dict(Counter(r.outcome for r in run.records)),  # type: ignore[dict-item]
    }
    return {k: dict(v) for k, v in rows.items()}
