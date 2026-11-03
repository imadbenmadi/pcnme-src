"""The zone container (PLAN M9.1 step 6; final-design 15.4): the fast-mode Zone on RealClock, MQTT (EMQX) for
the bus, HTTP for the calls (QUESTIONS Q53: netem gives the delays), the radio pipe in software at the intake
(15.3), and the car endpoints: POST /tasks (uploads, testbed/http.py), DELETE /uploads, POST /cars (attach),
POST /gps; GET /closed lists the closed tasks for the runner.

python -m pcnme.testbed.zone, with PCNME_CONFIG, PCNME_ZONE_ID, PCNME_SCENARIO, PCNME_SEED, PCNME_METHOD
(greedy or random), MQTT_HOST, PCNME_URLS ("z1=http://10.20.0.11:8000,...,cloud=http://10.30.0.10:8000"),
PCNME_SECONDS.
"""

# No `from __future__ import annotations`: FastAPI must resolve `request: Request` (imported locally).

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pcnme.config import Config, load_config
from pcnme.methods.simple import Greedy, Random
from pcnme.net.pipe import FifoPipe
from pcnme.node.timetable import RunTimeTable, lognormal
from pcnme.sim.rng import RandomStreams
from pcnme.testbed.clock import RealClock
from pcnme.testbed.http import HttpCalls, make_app
from pcnme.testbed.messages import Attach, Gps
from pcnme.testbed.radio import RemoteCar, software_radio
from pcnme.testbed.serve import PORT, on_loop
from pcnme.zone.layout import Site
from pcnme.zone.zone import TaskRecord, Zone


def build_zone(
    cfg: Config,
    zone_id: str,
    scenario: str,
    seed: int,
    clock: RealClock,
    bus: Any,
    urls: Mapping[str, str],
    method: str = "greedy",
) -> tuple[Zone, list[TaskRecord]]:
    """One zone as the city builds it (experiments/city.py), on a real network."""
    cfg, _ = software_radio(cfg)  # the base radio delays are netem's (15.3)
    usage = "evaluation" if seed >= 3000 else "training"
    streams = RandomStreams.from_config(cfg, seed, usage)
    table = RunTimeTable.from_config(cfg, lognormal)
    sc = cfg.scenarios[scenario]
    lay = cfg.layout
    active = [z for z in lay.zones if z not in lay.joining_zones]
    sites = [Site(z, *lay.zone_positions_m[z], lay.zone_radius_m) for z in active if z != zone_id]
    placement = Greedy(cfg) if method == "greedy" else Random(cfg, streams, zone_id)
    closed: list[TaskRecord] = []
    zone = Zone(
        zone_id,
        clock,  # type: ignore[arg-type]
        streams,
        cfg,
        table,
        FifoPipe(cfg.derived.b_wan),  # bookkeeping only: the WAN is the real wan-gw (Q53)
        None,  # type: ignore[arg-type]  # the cloud is another container, reached by HTTP
        placement,
        sc.deadlines,
        hw_name=sc.fog[zone_id],
        bus=bus,
        start_sites=tuple(sites),
        fog_classes=sc.fog,
        on_close=closed.append,
    )
    # The other active zones (deployment configuration, 7.5.2): the predictor reads their ids (10.2); the
    # calls to them go over HTTP, never through these entries.
    zone.peers = {z: None for z in active if z != zone_id}  # type: ignore[misc]
    own = f"http://127.0.0.1:{PORT}"  # the watchdog probes its own executor too (9.4), as in fast mode
    zone.calls = HttpCalls(clock, {**urls, zone_id: own})
    return zone, closed


def zone_app(clock: RealClock, zone: Zone, closed: list[TaskRecord]) -> Any:
    from fastapi import HTTPException, Request, Response

    from pcnme.testbed.codec import decode, encode

    app = make_app(clock, zone)

    def ok() -> Response:
        return Response(b"null", media_type="application/json")

    @app.post("/cars")
    async def cars(request: Request) -> Response:
        a = decode(await request.body())
        if not isinstance(a, Attach):
            raise HTTPException(422, "expected an Attach")
        car = RemoteCar(a.car_id, a.url, clock, zone)
        await on_loop(clock, lambda: zone.attach(car, a.distance_m))  # type: ignore[arg-type]
        return ok()

    @app.post("/gps")
    async def gps(request: Request) -> Response:
        g = decode(await request.body())
        if not isinstance(g, Gps):
            raise HTTPException(422, "expected a Gps")
        await on_loop(clock, lambda: zone.gps(g.car_id, g.x, g.y, g.vx, g.vy, g.car_time))
        return ok()

    @app.get("/status")
    async def status() -> Response:
        """Counts for the runner and for debugging (never on the decision path)."""

        def counts() -> dict[str, int]:
            bus = zone._bus
            return {
                "records": len(zone.records),
                "closed": len(closed),
                "cars": len(zone._cars),
                "published": getattr(bus, "published", 0),
                "delivered": getattr(bus, "delivered", 0),
            }

        return Response(encode(await on_loop(clock, counts)), media_type="application/json")

    @app.get("/closed")
    async def closed_tasks() -> Response:
        rows = await on_loop(clock, lambda: [(str(r.task.task_id), r.outcome) for r in closed])
        return Response(encode(rows), media_type="application/json")

    return app


def urls_from_env(spec: str) -> dict[str, str]:
    return dict(kv.split("=", 1) for kv in spec.split(",") if kv)


def main() -> None:  # pragma: no cover - runs in the container (the compose test)
    from pcnme.testbed.mqtt import MqttBus
    from pcnme.testbed.serve import serve

    cfg = load_config(Path(os.environ.get("PCNME_CONFIG", "config/default.yaml")))
    clock = RealClock()
    bus = MqttBus(clock, os.environ.get("MQTT_HOST", "broker"))
    zone, closed = build_zone(
        cfg,
        os.environ["PCNME_ZONE_ID"],
        os.environ.get("PCNME_SCENARIO", "S2"),
        int(os.environ.get("PCNME_SEED", "3000")),
        clock,
        bus,
        urls_from_env(os.environ["PCNME_URLS"]),
        os.environ.get("PCNME_METHOD", "greedy"),
    )
    serve(zone_app(clock, zone, closed))
    clock.run_for(float(os.environ.get("PCNME_SECONDS", "3600")))


if __name__ == "__main__":  # pragma: no cover
    main()
