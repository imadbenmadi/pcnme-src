"""The car container (PLAN M9.1 step 6; final-design 15.3, 15.4): a share of the fleet, simulated with the
fast-mode code (Car, mobility, zone choice with hysteresis), talking HTTP over radio_net to the zones.

Per GPS period (10, C13): move, choose the zone; on a change, cut the open uploads to the old zone (DELETE
/uploads, 9.7) and attach to the new one (POST /cars); then POST /gps. A task uploads to the serving zone
(POST /tasks with B_car and R_car, 15.3) or is held in a gap; a held or cut frame goes to the next zone with
the same task id, until it is hopeless (9.7.3). Results arrive on POST /results and are confirmed in the
reply (9.5). `send` is the network: tests pass a fake, main() passes HTTP.

In a container: python -m pcnme.testbed.car, with PCNME_CONFIG, PCNME_SCENARIO, PCNME_SEED, PCNME_CARS
("c000,c001" or "0-199/10:3": every 10th id from 3), PCNME_FLEET, PCNME_ZONE_URLS ("z1=http://...,..."),
PCNME_SELF_URL, PCNME_SECONDS.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pcnme.car.car import Car, Confirm, Result, Task
from pcnme.config import Config, load_config
from pcnme.experiments.city import CarApp, Mover, choose_zone
from pcnme.ids import TaskId
from pcnme.mobility.grid import RoadGrid, Walker
from pcnme.net.radio import MIN_DISTANCE, shannon_rate
from pcnme.sim.rng import RandomStreams
from pcnme.testbed.messages import Attach, Gps, Upload, UploadReply
from pcnme.zone.estimates import PathEstimates
from pcnme.zone.statestore import StateStore

OnReply = Callable[[int, Any], None]
Send = Callable[[str, str, str, Any, OnReply], None]  # zone, method, path, body, on_reply(status, payload)


class CarFleet:
    def __init__(
        self,
        cfg: Config,
        scenario: str,
        seed: int,
        clock: Any,
        car_ids: Sequence[str],
        send: Send,
        self_url: str,
        *,
        fleet_size: int | None = None,
        mover: Callable[[str, bool], Mover] | None = None,
    ) -> None:
        self.cfg = cfg
        self.clock = clock
        self._send = send
        self._url = self_url
        sc = cfg.scenarios[scenario]
        usage = "evaluation" if seed >= 3000 else "training"
        streams = RandomStreams.from_config(cfg, seed, usage)
        lay = cfg.layout
        self.centres = {z: lay.zone_positions_m[z] for z in lay.zones if z not in lay.joining_zones}
        fleet = fleet_size if fleet_size is not None else cfg.cars.count[0]
        n_hot = round(cfg.cars.hotspot_h[sc.mobility] * fleet)
        grid = RoadGrid.from_area(lay.area_m, cfg.cars.grid.block_m)
        hot_nodes = grid.within(lay.zone_positions_m["z1"], lay.zone_radius_m)
        speeds = cfg.cars.speed_low_mps if sc.mobility == "low" else cfg.cars.speed_high_mps
        est = PathEstimates(cfg, StateStore("car", clock, 1.0, {}, 0.0))
        self._fastest = {  # 9.7.3, as the fast-mode city
            t: tt.input_bits / shannon_rate(MIN_DISTANCE, cfg.radio)
            + min(r.mean for q in ("gpu", "cpu") if (r := cfg.time_table.row(t, cfg.tof.strongest_class, q)))
            + est.t_ret_fog()
            for t, tt in cfg.task_types.items()
        }
        self.apps: list[CarApp] = []
        self._by_id: dict[str, CarApp] = {}
        for car_id in car_ids:
            hot = int(car_id[1:]) < n_hot
            if mover is not None:
                m = mover(car_id, hot)
            else:
                rng = streams.generator("mobility", car_id)
                m = Walker(clock, grid, rng, speeds, cfg.cars.grid, hot_nodes if hot else None)
            app = CarApp(car=None, mover=m, hotspot=hot)  # type: ignore[arg-type]
            app.car = Car(
                car_id,
                cfg,
                sc,
                streams,
                clock,
                m.position(),
                self._on_task(app),
                fleet=fleet,
                coverage=cfg.cars.coverage[sc.mobility],
            )
            self.apps.append(app)
            self._by_id[car_id] = app
        self._period = cfg.cars.gps_period
        clock.schedule(clock.now, self._tick)

    def start(self, seconds: float) -> None:
        """The cars make tasks for `seconds` from now (as City.start)."""
        for app in self.apps:
            app.car.start(until=self.clock.now + seconds)

    def _on_task(self, app: CarApp) -> Callable[[Task], None]:
        return lambda task: self.task(app, task)

    # --- the GPS period ------------------------------------------------------------------------------------

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
            if app.serving is not None:
                vx, vy = app.mover.velocity(now)
                gps = Gps(app.car.car_id, pos[0], pos[1], vx, vy, now)
                self._send(app.serving, "POST", "/gps", gps, _ignore)
        self.clock.schedule(now + self._period, self._tick)

    def _switch(self, app: CarApp, new: str | None) -> None:
        old, app.serving = app.serving, new
        if old is not None:
            for tid, zone in list(app.uploading.items()):
                if zone == old:
                    del app.uploading[tid]
                    self._send(old, "DELETE", "/uploads", tid, self._on_cut(app, tid))
        if new is not None:
            dist = math.dist(app.car.position, self.centres[new])
            self._send(new, "POST", "/cars", Attach(app.car.car_id, dist, self._url), _ignore)
            for tid in list(app.held):
                self._upload(app, app.held.pop(tid))

    def _on_cut(self, app: CarApp, tid: TaskId) -> OnReply:
        def cut(status: int, _reply: Any) -> None:
            """200: cut before the last bit (9.7.2): the car keeps the frame for the next zone. Otherwise the
            last bit was on the air and that zone is the anchor."""
            task = app.car._tasks.get(tid)
            if status != 200 or task is None:
                return
            if app.serving is not None:
                self._upload(app, task)
            else:
                app.held[tid] = task

        return cut

    # --- tasks ---------------------------------------------------------------------------------------------

    def task(self, app: CarApp, task: Task) -> None:
        if app.serving is None:
            app.held[task.task_id] = task  # in a gap: keep the frame (9.7)
            app.uncovered.add(task.task_id)
        else:
            self._upload(app, task)

    def _upload(self, app: CarApp, task: Task) -> None:
        now = self.clock.now
        if self._hopeless(app, task, now):
            return  # a car-side drop (9.7.3)
        zone = app.serving
        assert zone is not None
        app.uploading[task.task_id] = zone
        rate = shannon_rate(math.dist(app.car.position, self.centres[zone]), self.cfg.radio)
        up = Upload(task, b_car=app.car.budget(task, now), rate=rate)

        def replied(status: int, reply: Any) -> None:
            if status == 200 and isinstance(reply, UploadReply):
                app.uploading.pop(task.task_id, None)  # t_rx: the task exists at its anchor

        self._send(zone, "POST", "/tasks", up, replied)

    def _hopeless(self, app: CarApp, task: Task, now: float) -> bool:
        if task.cls == "hard":
            return app.car.budget(task, now) < self._fastest[task.task_type]
        return now > task.created + task.deadline + task.l_max

    def _give_up(self, app: CarApp, now: float) -> None:
        for tid, task in list(app.held.items()):
            if self._hopeless(app, task, now):
                del app.held[tid]

    def result(self, result: Result) -> Confirm | None:
        """POST /results: a result copy reaches the car (9.5.1-2)."""
        app = self._by_id.get(str(result.task_id[0]))
        return None if app is None else app.car.receive(result, self.clock.now)


def _ignore(_status: int, _reply: Any) -> None:
    pass


def http_send(clock: Any, urls: dict[str, str], timeout: float = 5.0) -> Send:
    """The network for main(): each request on a worker thread, the reply handed to the loop thread."""
    import concurrent.futures

    import httpx

    from pcnme.testbed.codec import decode, encode

    client = httpx.Client(timeout=timeout)
    pool = concurrent.futures.ThreadPoolExecutor(32, thread_name_prefix="car-http")

    def send(zone: str, method: str, path: str, body: Any, on_reply: OnReply) -> None:
        def work() -> None:
            try:
                r = client.request(method, urls[zone] + path, content=encode(body))
            except Exception:
                return  # lost: the car's own rules notice (9.7.3)
            reply = decode(r.content) if r.status_code == 200 and r.content else None
            clock.call_soon(lambda: on_reply(r.status_code, reply))

        pool.submit(work)

    return send


def car_ids(spec: str) -> list[str]:
    """ "c000,c001" or "0-199/10:3" (every 10th id from 3 in 0..199)."""
    if "/" in spec:
        span, step = spec.split("/")
        lo, hi = (int(x) for x in span.split("-"))
        k, start = (int(x) for x in step.split(":"))
        return [f"c{i:03d}" for i in range(lo + start, hi + 1, k)]
    return [c for c in spec.split(",") if c]


def main() -> None:  # pragma: no cover - runs in the container (the compose test)
    from pcnme.testbed.clock import RealClock
    from pcnme.testbed.radio import software_radio
    from pcnme.testbed.serve import results_app, serve

    cfg, _ = software_radio(load_config(Path(os.environ.get("PCNME_CONFIG", "config/default.yaml"))))
    clock = RealClock()
    urls = dict(kv.split("=", 1) for kv in os.environ["PCNME_ZONE_URLS"].split(","))
    fleet = CarFleet(
        cfg,
        os.environ.get("PCNME_SCENARIO", "S2"),
        int(os.environ.get("PCNME_SEED", "3000")),
        clock,
        car_ids(os.environ.get("PCNME_CARS", "c000")),
        http_send(clock, urls),
        os.environ["PCNME_SELF_URL"],
        fleet_size=int(os.environ["PCNME_FLEET"]) if "PCNME_FLEET" in os.environ else None,
    )
    seconds = float(os.environ.get("PCNME_SECONDS", "3600"))
    serve(results_app(clock, fleet))
    fleet.start(seconds)
    clock.run_for(seconds + 30.0)  # then drain (as the fast-mode city)


if __name__ == "__main__":  # pragma: no cover
    main()
