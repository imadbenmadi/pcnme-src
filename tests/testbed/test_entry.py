"""The testbed entry points (PLAN M9.1 step 6; final-design 15.4): a zone on RealClock with real-network
calls, the car endpoints at its intake (POST /cars, POST /gps, GET /closed), and the cloud."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from pcnme.testbed.clock import RealClock
from pcnme.testbed.codec import decode, encode
from pcnme.testbed.messages import Attach, Gps
from pcnme.testbed.radio import RemoteCar
from tests.zone.harness import fixed_delay_config

pytest.importorskip("fastapi")


class FakeBus:
    """MqttBus's interface with no broker: every client is alive and publishes go nowhere."""

    def connect(self, client_id: str, delay: Any, on_message: Any, will: Any = None) -> Any:
        class C:
            alive = True

            def subscribe(self, topic: str) -> int:
                return 1

            def unsubscribe(self, topic: str) -> int:
                return 0

            def count(self, topic: str) -> int:
                return 0

            def publish(self, *a: Any, **k: Any) -> None:
                pass

        return C()

    def is_alive(self, client_id: str) -> bool:
        return True


def test_a_testbed_zone_takes_cars_gps_and_lists_closed_tasks() -> None:
    from fastapi.testclient import TestClient

    from pcnme.testbed.zone import build_zone, zone_app

    cfg = fixed_delay_config()
    clock = RealClock()
    zone, closed = build_zone(cfg, "z1", "S2", 3000, clock, FakeBus(), {"cloud": "http://cloud:8000"})
    assert zone.calls.real_network and cfg.radio.uplink_delay != (0.0, 0.0)
    assert zone.calls._urls["z1"] == "http://127.0.0.1:8000"  # its own executor (watchdog probes, 9.4)
    assert set(zone.peers) == {"z2", "z3", "z4"}  # the other active zones, for the predictor (10.2)
    assert zone._cfg.radio.uplink_delay == (0.0, 0.0)  # netem gives the radio delays (15.3)
    threading.Thread(target=clock.run_for, args=(1.0,), name="zone-loop", daemon=True).start()
    http = TestClient(zone_app(clock, zone, closed))
    assert http.post("/cars", content=encode(Attach("c000", 120.0, "http://car-1:8000"))).status_code == 200
    car, rate = zone._cars["c000"]
    assert isinstance(car, RemoteCar) and rate > 0
    x, y = cfg.layout.zone_positions_m["z1"]
    assert http.post("/gps", content=encode(Gps("c000", x, y, 1.0, 0.0, 0.5))).status_code == 200
    assert "c000" in zone.registry
    assert decode(http.get("/closed").content) == []
    st = decode(http.get("/status").content)
    assert isinstance(st, dict) and st["cars"] == 1 and st["closed"] == 0
    clock.stop()
