"""The cloud container (PLAN M9.1 step 6; final-design 15.4): the fast-mode Cloud (its executor with many
slots) on RealClock, MQTT for its reports and state, and POST/GET/DELETE /tasks over HTTP (testbed/http.py).

python -m pcnme.testbed.cloud, with PCNME_CONFIG, PCNME_SEED, MQTT_HOST, PCNME_SECONDS.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pcnme.config import Config, load_config
from pcnme.node.timetable import RunTimeTable, lognormal
from pcnme.sim.rng import RandomStreams
from pcnme.testbed.clock import RealClock
from pcnme.zone.zone import Cloud


def build_cloud(cfg: Config, seed: int, clock: RealClock, bus: Any) -> Cloud:
    usage = "evaluation" if seed >= 3000 else "training"
    streams = RandomStreams.from_config(cfg, seed, usage)
    return Cloud(clock, streams, cfg, RunTimeTable.from_config(cfg, lognormal), bus)  # type: ignore[arg-type]


def main() -> None:  # pragma: no cover - runs in the container (the compose test)
    from pcnme.testbed.http import make_app
    from pcnme.testbed.mqtt import MqttBus
    from pcnme.testbed.serve import serve

    cfg = load_config(Path(os.environ.get("PCNME_CONFIG", "config/default.yaml")))
    clock = RealClock()
    bus = MqttBus(clock, os.environ["MQTT_HOST"])
    cloud = build_cloud(cfg, int(os.environ.get("PCNME_SEED", "3000")), clock, bus)
    serve(make_app(clock, cloud))
    clock.run_for(float(os.environ.get("PCNME_SECONDS", "3600")))


if __name__ == "__main__":  # pragma: no cover
    main()
