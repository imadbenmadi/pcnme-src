"""The zone container (PLAN M9.1 step 6; final-design 15.4): the fast-mode Zone on RealClock, MQTT (EMQX) for
the bus, HTTP for the calls (QUESTIONS Q53: netem gives the delays), the radio pipe in software at the intake
(15.3), and the car endpoints: POST /tasks (uploads, testbed/http.py), DELETE /uploads, POST /cars (attach),
POST /gps; GET /closed lists the closed tasks for the runner.

python -m pcnme.testbed.zone, with PCNME_CONFIG, PCNME_ZONE_ID, PCNME_SCENARIO, PCNME_SEED, PCNME_METHOD
(greedy, random, or a learning method deployed from PCNME_MODELS/<scenario>-<method>/trained.pt), MQTT_HOST,
PCNME_URLS ("z1=http://10.20.0.11:8000,...,cloud=http://10.30.0.10:8000"), PCNME_SECONDS.
"""

# No `from __future__ import annotations`: FastAPI must resolve `request: Request` (imported locally).

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pcnme.config import Config, load_config
from pcnme.experiments.city import record_rows
from pcnme.methods.simple import Greedy, Random
from pcnme.metrics.runlog import RunLog
from pcnme.net.pipe import FifoPipe
from pcnme.node.timetable import RunTimeTable, lognormal
from pcnme.sim.rng import RandomStreams
from pcnme.testbed.clock import RealClock
from pcnme.testbed.http import HttpCalls, make_app
from pcnme.testbed.messages import Attach, Gps, GpsBatch
from pcnme.testbed.netloop import NetLoop
from pcnme.testbed.radio import RemoteCar, software_radio
from pcnme.testbed.serve import PORT, on_loop
from pcnme.zone.decision import Placement
from pcnme.zone.layout import Site
from pcnme.zone.zone import TaskRecord, Zone


def zone_placement(
    cfg: Config, streams: RandomStreams, zone_id: str, method: str, scenario: str, seed: int, models: Path
) -> Placement:
    """What decides in this zone: Greedy, Random, or the trained method deployed as in fast mode (16.1, Q42:
    it keeps learning online; Q54: CPU torch in the zone image)."""
    if method == "greedy":
        return Greedy(cfg)
    if method == "random":
        return Random(cfg, streams, zone_id)
    import torch

    from pcnme.experiments.runner import deployed
    from pcnme.pipeline import checkpoint

    f = models / f"{scenario}-{method}" / "trained.pt"
    if not f.exists():
        raise FileNotFoundError(f"no trained {method} for {scenario} at {f}")
    torch.set_num_threads(1)  # one zone, one CPU (15.5)
    return deployed(cfg, checkpoint.load(f), seed)(cfg, streams, zone_id)


def build_zone(
    cfg: Config,
    zone_id: str,
    scenario: str,
    seed: int,
    clock: RealClock,
    bus: Any,
    urls: Mapping[str, str],
    method: str = "greedy",
    *,
    net: NetLoop,
    models: Path = Path("/models"),
) -> tuple[Zone, list[TaskRecord]]:
    """One zone as the city builds it (experiments/city.py), on a real network."""
    from pcnme.experiments.runner import method_config

    cfg, _ = software_radio(method_config(cfg, method))  # the base radio delays are netem's (15.3)
    usage = "evaluation" if seed >= 3000 else "training"
    streams = RandomStreams.from_config(cfg, seed, usage)
    table = RunTimeTable.from_config(cfg, lognormal)
    sc = cfg.scenarios[scenario]
    lay = cfg.layout
    active = [z for z in lay.zones if z not in lay.joining_zones]
    sites = [Site(z, *lay.zone_positions_m[z], lay.zone_radius_m) for z in active if z != zone_id]
    placement = zone_placement(cfg, streams, zone_id, method, scenario, seed, models)
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
    sizes = {name: round(t.input_bits / 8) for name, t in cfg.task_types.items()}
    zone.calls = HttpCalls(clock, {**urls, zone_id: own}, net, input_bytes=sizes)
    return zone, closed


def zone_app(clock: RealClock, zone: Zone, closed: list[TaskRecord], net: NetLoop) -> Any:
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
        car = RemoteCar(a.car_id, a.url, zone, net)
        await on_loop(clock, lambda: zone.attach(car, a.distance_m))  # type: ignore[arg-type]
        return ok()

    @app.post("/gps")
    async def gps(request: Request) -> Response:
        g = decode(await request.body())
        fixes = (g,) if isinstance(g, Gps) else g.fixes if isinstance(g, GpsBatch) else None
        if fixes is None:
            raise HTTPException(422, "expected a Gps or a GpsBatch")

        def each() -> list[str]:
            for f in fixes:  # each fix as its own message (5.2)
                zone.gps(f.car_id, f.x, f.y, f.vx, f.vy, f.car_time)
            # QUESTIONS Q61: the cars this zone does not serve (T_leave fired, 9.6): they attach again
            return [f.car_id for f in fixes if f.car_id not in zone._cars]

        unknown = await on_loop(clock, each)
        return Response(encode(unknown), media_type="application/json")

    @app.get("/status")
    async def status() -> Response:
        """Counts for the runner and for debugging (never on the decision path)."""

        def counts() -> dict[str, object]:
            bus = zone._bus
            slots = {q: (s.n_hard, s.n_soft, s.u) for q, s in zone.executor.snapshot().items()}
            ages = {z: zone.store.age(z) for z in (*zone.peers, "cloud")}
            return {
                "records": len(zone.records),
                "closed": len(closed),
                "cars": len(zone._cars),
                "published": getattr(bus, "published", 0),
                "delivered": getattr(bus, "delivered", 0),
                "slots": slots,  # per slot type: queued hard, queued soft, busy
                "ages": ages,  # each peer's newest snapshot age, s (stale after stateplane.stale_after)
                "wan_estimate": zone.store.estimate(),
            }

        return Response(encode(await on_loop(clock, counts)), media_type="application/json")

    @app.get("/closed")
    async def closed_tasks() -> Response:
        rows = await on_loop(clock, lambda: [(str(r.task.task_id), r.outcome) for r in closed])
        return Response(encode(rows), media_type="application/json")

    return app


def write_rows(
    cfg: Config, scenario: str, method: str, seed: int, zone_id: str, closed: list[TaskRecord], results: Path
) -> Path:
    """The tasks and attempts rows of the tasks this zone anchored (METRICS.md), as JSON for the runner
    (the zone image has no Parquet writer): results/<run_id>/zones/<zone>.json."""
    log = RunLog(scenario, method, seed, "testbed")
    record_rows(log, cfg, closed)
    path = results / log.run_id / "zones" / f"{zone_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log.rows, default=str), encoding="utf-8")
    return path


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
        net=(net := NetLoop(clock)),
        models=Path(os.environ.get("PCNME_MODELS", "/models")),
    )
    serve(zone_app(clock, zone, closed, net))
    seconds = float(os.environ.get("PCNME_SECONDS", "3600")) + float(os.environ.get("PCNME_DRAIN", "30"))
    if os.environ.get("PCNME_PROFILE"):  # where the loop thread spends its time (testbed tuning)
        import cProfile

        prof = cProfile.Profile()
        prof.runcall(clock.run_for, seconds)
        prof.dump_stats(str(Path(os.environ.get("PCNME_RESULTS", "/tmp")) / f"profile-{zone.zone_id}.prof"))
    else:
        clock.run_for(seconds)
    if "PCNME_RESULTS" in os.environ:  # the runner collects these (M9.3)
        scenario, seed = os.environ.get("PCNME_SCENARIO", "S2"), int(os.environ.get("PCNME_SEED", "3000"))
        method = os.environ.get("PCNME_METHOD", "greedy")
        write_rows(zone._cfg, scenario, method, seed, zone.zone_id, closed, Path(os.environ["PCNME_RESULTS"]))
        counts = Path(os.environ["PCNME_RESULTS"]) / f"counts-{zone.zone_id}.json"
        diag = {**zone.counts, **clock.lag, **clock.slow}
        counts.write_text(json.dumps(diag, indent=1), encoding="utf-8")  # diagnosis (PLAN Q1)


if __name__ == "__main__":  # pragma: no cover
    main()
