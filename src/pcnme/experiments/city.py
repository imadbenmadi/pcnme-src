"""The city in fast mode: zones, the cloud, the bus, the WAN and cars that move (final-design 3, 9.7, 10,
16.2; PLAN M3.8).

Every GPS period (100 ms) each car, in a fixed order: moves on the grid, chooses its access point (in range,
the current one kept unless another is handover_hysteresis_m closer, C13), and sends its GPS to the serving
zone. A car in no zone's coverage is in a gap (C12): it talks to nobody. A task the car creates goes to its
serving zone at the rate of its current distance; if the car changes zone or enters a gap before the
upload's last bit, the upload is cut (9.7): the car keeps the frame and uploads it to the zone it reaches
next with the same task id and its new B_car. It gives the frame up when the budget is gone (9.7.3): hard,
below the fastest possible upload + run + return; soft, past D + L_max (a car-side drop, logged).

The hotspot (18, QUESTIONS Q25): a share h of the cars stays on the roads between z1's intersections, the
rest roam the whole grid; h is set so that z1 holds hotspot_share of the cars in coverage on average.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pcnme.bus.bus import Bus
from pcnme.car.car import Car, Task
from pcnme.config import Config
from pcnme.energy import EnergyModel
from pcnme.ids import TaskId
from pcnme.methods.simple import Greedy, Random
from pcnme.metrics.runlog import RunLog
from pcnme.mobility.grid import RoadGrid, Walker
from pcnme.net.pipe import FifoPipe
from pcnme.net.radio import MIN_DISTANCE, shannon_rate
from pcnme.node.timetable import RunTimeTable, lognormal
from pcnme.sim.clock import EventClock
from pcnme.sim.rng import RandomStreams
from pcnme.zone.estimates import PathEstimates
from pcnme.zone.layout import Site
from pcnme.zone.statestore import StateStore
from pcnme.zone.zone import Cloud, Placement, TaskRecord, Zone

DRAIN = 30.0  # s after the last arrival, so every task can close


class Mover(Protocol):
    def position(self, t: float | None = None) -> tuple[float, float]: ...
    def velocity(self, t: float | None = None) -> tuple[float, float]: ...


def choose_zone(
    pos: tuple[float, float],
    serving: str | None,
    centres: Mapping[str, tuple[float, float]],
    radius: float,
    hysteresis: float,
) -> str | None:
    """The access point the car uses now (C13, 18): the nearest in range; the current one is kept unless
    another is at least `hysteresis` closer. None: a coverage gap (C12)."""
    dist = {z: math.dist(pos, c) for z, c in centres.items()}
    in_range = sorted((d, z) for z, d in dist.items() if d <= radius)
    if not in_range:
        return None
    best_d, best = in_range[0]
    if (
        serving is not None
        and dist[serving] <= radius
        and best != serving
        and best_d + hysteresis > dist[serving]
    ):
        return serving
    if serving is not None and dist[serving] <= radius and best == serving:
        return serving
    return best


def hotspot_fraction(
    grid: RoadGrid, centres: Mapping[str, tuple[float, float]], radius: float, share: float
) -> float:
    """Q25: h with (h + (1 - h) p1) / (h + (1 - h) p_cov) = share, where p1 and p_cov are the shares of road
    length inside z1's circle and inside any circle (a roaming car is anywhere on the roads alike)."""
    z1 = centres["z1"]
    inside = covered = total = 0
    step = 1.0  # m: a fine sampling of every road
    for k in range(grid.nx + 1):
        for along in range(int(grid.nx * grid.block / step) + 1):
            for p in ((k * grid.block, along * step), (along * step, k * grid.block)):
                total += 1
                covered += any(math.dist(p, c) <= radius for c in centres.values())
                inside += math.dist(p, z1) <= radius
    p1, pc = inside / total, covered / total
    h = (share * pc - p1) / (1 - p1 - share * (1 - pc))
    return min(1.0, max(0.0, h))


@dataclass
class CarApp:
    """One car app (15.4): the car, how it moves, its serving zone, and the frames it holds (9.7)."""

    car: Car
    mover: Mover
    hotspot: bool
    serving: str | None = None
    held: dict[TaskId, Task] = field(default_factory=dict)
    uploading: dict[TaskId, str] = field(default_factory=dict)  # task id -> the zone receiving its upload


@dataclass(frozen=True, slots=True)
class CarDrop:
    """A frame the car gave up before any zone held it (9.7.3)."""

    task: Task
    at: float


class City:
    def __init__(
        self,
        cfg: Config,
        scenario: str,
        method: str | Callable[[Config, RandomStreams, str], Placement],
        seed: int,
        n_cars: int | None = None,
        shape: Callable[..., Any] = lognormal,
        mover: Callable[[str, bool], Mover] | None = None,
    ) -> None:
        self.cfg = cfg
        self.scenario = cfg.scenarios[scenario]
        self.log = RunLog(scenario, method if isinstance(method, str) else "custom", seed)
        self.clock = clock = EventClock()
        self.streams = streams = RandomStreams.from_config(cfg, seed, "training")
        self.bus = bus = Bus(clock, streams, cfg.links.broker_extra_delay)
        table = RunTimeTable.from_config(cfg, shape)
        self.wan = FifoPipe(cfg.derived.b_wan)
        self.cloud = Cloud(clock, streams, cfg, table, bus)
        lay = cfg.layout
        self.active = tuple(z for z in lay.zones if z not in lay.joining_zones)
        self.centres = {z: lay.zone_positions_m[z] for z in self.active}
        sites = {z: Site(z, *self.centres[z], lay.zone_radius_m) for z in self.active}
        self.records: list[TaskRecord] = []
        self.car_drops: list[CarDrop] = []
        self.created: list[TaskId] = []
        ledger: dict[tuple[str, object], TaskRecord] = {}

        def placement(z: str) -> Placement:
            if method == "greedy":
                return Greedy(cfg)
            if method == "random":
                return Random(cfg, streams, z)
            return method(cfg, streams, z)  # type: ignore[operator]

        self.zones = {
            z: Zone(
                z,
                clock,
                streams,
                cfg,
                table,
                self.wan,
                self.cloud,
                placement(z),
                self.scenario.deadlines,
                hw_name=self.scenario.fog[z],
                bus=bus,
                start_sites=tuple(s for o, s in sites.items() if o != z),
                fog_classes=self.scenario.fog,
                ledger=ledger,
                on_close=self.records.append,
            )
            for z in self.active
        }
        for zone in self.zones.values():
            zone.peers = self.zones
        self.grid = RoadGrid.from_area(lay.area_m, cfg.cars.grid.block_m)
        hot_nodes = self.grid.within(lay.zone_positions_m["z1"], lay.zone_radius_m)
        fleet = n_cars if n_cars is not None else cfg.cars.count[0]
        self.h = hotspot_fraction(self.grid, self.centres, lay.zone_radius_m, cfg.cars.hotspot_share)
        n_hot = round(self.h * fleet)
        speeds = cfg.cars.speed_low_mps if self.scenario.mobility == "low" else cfg.cars.speed_high_mps
        est = PathEstimates(cfg, StateStore("car", clock, 1.0, {}, 0.0))
        self._fastest = {
            t: tt.input_bits / shannon_rate(MIN_DISTANCE, cfg.radio)
            + min(r.mean for q in ("gpu", "cpu") if (r := cfg.time_table.row(t, cfg.tof.strongest_class, q)))
            + est.t_ret_fog()
            for t, tt in cfg.task_types.items()
        }
        self.apps: list[CarApp] = []
        for i in range(fleet):
            car_id = f"c{i:03d}"
            hot = i < n_hot
            if mover is not None:
                m = mover(car_id, hot)
            else:
                rng = streams.generator("mobility", car_id)
                m = Walker(clock, self.grid, rng, speeds, cfg.cars.grid, hot_nodes if hot else None)
            app = CarApp(car=None, mover=m, hotspot=hot)  # type: ignore[arg-type]
            app.car = Car(
                car_id,
                cfg,
                self.scenario,
                streams,
                clock,
                m.position(),
                functools.partial(self._task, app),
                fleet=fleet,
            )
            self.apps.append(app)
        self._period = cfg.cars.gps_period
        clock.schedule(clock.now, self._tick)
        self._bits = {"wan": 0.0, **{f"backhaul-{z}": 0.0 for z in self.zones}}
        self._refused = {z: 0 for z in self.zones}
        clock.schedule(clock.now, self._sample)

    # --- the GPS period: move, attach, GPS ------------------------------------------------------------------

    def _tick(self) -> None:
        now = self.clock.now
        lay = self.cfg.layout
        for app in self.apps:
            pos = app.mover.position(now)
            app.car.position = pos
            z = choose_zone(pos, app.serving, self.centres, lay.zone_radius_m, lay.handover_hysteresis_m)
            if z != app.serving:
                self._switch(app, z)
            self._give_up(app, now)
            vx, vy = app.mover.velocity(now)
            speed = math.hypot(vx, vy)
            self.log.add(
                "positions",
                now,
                car_id=app.car.car_id,
                x_m=pos[0],
                y_m=pos[1],
                speed_mps=speed,
                heading_x=vx / speed if speed else 0.0,
                heading_y=vy / speed if speed else 0.0,
                zone=app.serving or "",
            )
            if app.serving is not None:
                self.zones[app.serving].gps(app.car.car_id, pos[0], pos[1], vx, vy, now)
        self.clock.schedule(now + self._period, self._tick)

    def _rate(self, app: CarApp, zone: str) -> float:
        return shannon_rate(math.dist(app.car.position, self.centres[zone]), self.cfg.radio)

    def _switch(self, app: CarApp, new: str | None) -> None:
        """A handover or a gap: uploads in flight to the old zone are cut (9.7) and held; the new zone gets
        the attach and the held frames."""
        old = app.serving
        kind = "attach" if old is None else ("gap" if new is None else "handover")
        self.log.add(
            "events",
            self.clock.now,
            car_id=app.car.car_id,
            kind=kind,
            zone_from=old or "",
            zone_to=new or "",
            task_id="",
        )
        if old is not None:
            for tid, zone in list(app.uploading.items()):
                if zone != old:
                    continue
                del app.uploading[tid]
                if self.zones[old].cut_upload(tid):
                    app.held[tid] = app.car._tasks[tid]  # cut: the car keeps the frame (9.7.2)
                    self.log.add(
                        "events",
                        self.clock.now,
                        car_id=app.car.car_id,
                        kind="upload_cut",
                        zone_from=old,
                        zone_to=new or "",
                        task_id=str(tid),
                    )
                # else: its last bit was sent, it arrives there and that zone is its anchor
        app.serving = new
        if new is not None:
            self.zones[new].attach(app.car, math.dist(app.car.position, self.centres[new]))
            for tid in list(app.held):
                self._upload(app, app.held.pop(tid))

    # --- tasks: upload now, or hold the frame (9.7) ---------------------------------------------------------

    def _task(self, app: CarApp, task: Task) -> None:
        self.created.append(task.task_id)
        if app.serving is None:
            app.held[task.task_id] = task  # in a gap: keep the frame
        else:
            self._upload(app, task)

    def _upload(self, app: CarApp, task: Task) -> None:
        if self._hopeless(app, task, self.clock.now):
            self._car_drop(app, task, self.clock.now)
            return
        assert app.serving is not None
        app.uploading[task.task_id] = app.serving
        self.zones[app.serving].upload(task, self._rate(app, app.serving))

    def _hopeless(self, app: CarApp, task: Task, now: float) -> bool:
        """9.7.3: hard: the budget is below the fastest possible upload + run + return; soft: past D +
        L_max."""
        if task.cls == "hard":
            return app.car.budget(task, now) < self._fastest[task.task_type]
        return now > task.created + task.deadline + task.l_max

    def _give_up(self, app: CarApp, now: float) -> None:
        for tid, task in list(app.held.items()):
            if self._hopeless(app, task, now):
                del app.held[tid]
                self._car_drop(app, task, now)

    def _car_drop(self, app: CarApp, task: Task, now: float) -> None:
        self.car_drops.append(CarDrop(task, now))
        self.log.add(
            "events",
            now,
            car_id=app.car.car_id,
            kind="car_drop",
            zone_from=app.serving or "",
            zone_to="",
            task_id=str(task.task_id),
        )

    # --- logs (16.3, METRICS.md, Q28) ----------------------------------------------------------------------

    def _sample(self) -> None:
        """Every state period: per zone and slot type (queues, busy slots, refusals, ŵ_WAN) and per pipe
        (wait, bits pushed, TI over the period)."""
        now, period = self.clock.now, self.cfg.stateplane.state_period
        for z, zone in self.zones.items():
            refused = zone.executor.refused - self._refused[z]
            self._refused[z] = zone.executor.refused
            for q, s in zone.executor.snapshot().items():
                self.log.add(
                    "zones",
                    now,
                    zone=z,
                    slot_type=q,
                    rho=s.u / s.c,
                    queue_hard=s.n_hard,
                    queue_soft=s.n_soft,
                    busy=s.u,
                    refusals=refused,
                    wan_wait_est_s=zone.store.estimate(),
                )
        pipes = {"wan": self.wan, **{f"backhaul-{z}": zone.backhaul for z, zone in self.zones.items()}}
        for name, pipe in pipes.items():
            bits = pipe.bits_sent - self._bits[name]
            self._bits[name] = pipe.bits_sent
            assert pipe.bandwidth is not None
            self.log.add(
                "pipes",
                now,
                pipe=name,
                wait_s=pipe.wait_at(now),
                bits=bits,
                ti=bits / pipe.bandwidth / period,
            )
        self.clock.schedule(now + period, self._sample)

    def _task_rows(self) -> None:
        em = EnergyModel(self.cfg)
        for r in self.records:
            p = r.parts
            self.log.add(
                "tasks",
                r.task.created,
                anchor=r.anchor,
                type=r.task.task_type,
                cls=r.task.cls,
                outcome=r.outcome,
                boulder=r.route == "cloud",
                latency_s=r.confirm.latency if r.confirm else None,
                deadline_s=r.task.deadline,
                attempts=len(r.attempts),
                energy_j=em.of_record(r).total + em.earlier(r),
                energy_wasted_j=em.earlier(r),
                upload_s=p.get("upload"),
                forward_s=p.get("input"),
                wait_s=p.get("queue"),
                run_s=p.get("run"),
                return_s=p.get("return"),
                zone_changed=int(r.delivered_by is not None and r.delivered_by != r.anchor),
            )
            for a, i in sorted(r.attempts.items()):
                last = a == max(r.attempts)
                if a == r.result_attempt:
                    result = "delivered"
                elif not last:
                    e = next((x for x in r.earlier if x.attempt == a), None)
                    result = (
                        f"refused_{e.reply}"
                        if e and e.reply in ("full", "late")
                        else (e.reply if e else "lost")
                    )
                else:
                    result = r.reply if r.reply in ("lost", "no_reply") else (r.outcome or "")
                rep_ = r.report if r.report is not None and r.report.key.attempt == a else None
                self.log.add(
                    "attempts",
                    i.t_send,
                    anchor=r.anchor,
                    executor=i.executor,
                    slot_type=rep_.slot if rep_ else "",
                    result=result,
                    option=i.option,
                    attempt=a,
                    task_id=str(r.task.task_id),
                    t_send_s=i.t_send,
                    t_at_executor_s=i.t_at_executor,
                    t_start_s=rep_.start if rep_ else None,
                    t_finish_s=rep_.finish if rep_ else None,
                    t_result_at_car_s=r.result_at_car if a == r.result_attempt else None,
                )

    def write(self, out_dir: Path) -> Path:
        """results/<run_id>/ (METRICS.md): the Parquet files, run.json, points.lp."""
        self._task_rows()
        info = {
            "scenario": self.scenario.name,
            "seed": self.log.seed,
            "seconds": self.seconds,
            "fleet": len(self.apps),
            "hotspot_share_h": self.h,
        }
        return self.log.write(out_dir, info)

    # --- running --------------------------------------------------------------------------------------------

    def run(self, seconds: float) -> City:
        for app in self.apps:
            app.car.start(until=seconds)
        self.seconds = seconds
        self.clock.run_until(seconds + DRAIN)
        for app in self.apps:  # frames still held at the end were given up (the budget is long gone)
            for task in app.held.values():
                self.car_drops.append(CarDrop(task, self.clock.now))
            app.held.clear()
        return self
