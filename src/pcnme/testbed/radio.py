"""The radio on the testbed (PLAN M9.1 step 5; final-design 15.3).

tc cannot give every virtual car its own rate on one interface, so the uplink pipe runs in software at the
zone's intake with the fast-mode code (Zone.upload with the car's R_car and B_car, net/radio.py), and netem
adds only the radio's base delays. software_radio() therefore zeroes the modelled radio delays in the zone's
config and returns them for netem, so they are not counted twice.

RemoteCar stands in for a car in the zone's registry: a result copy is POSTed to the car container
(/results); the car's Confirm comes back in the HTTP reply and is handed to the zone loop
(Zone.car_confirmed). No reply (the car left, the radio dropped it): no confirm, the anchor's T_ack timer
notices (9.5.4), as in fast mode.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

from pcnme.car.car import Confirm, Result, Task
from pcnme.config import Config
from pcnme.testbed.codec import decode, encode
from pcnme.testbed.messages import Upload, UploadReply
from pcnme.testbed.netloop import NetLoop

__all__ = ["RemoteCar", "Upload", "UploadReply", "software_radio"]


def software_radio(cfg: Config) -> tuple[Config, dict[str, tuple[float, float]]]:
    """The zone's config with zero modelled radio delays, and those delays for netem on radio_net (15.3)."""
    r = cfg.radio
    netem = {"uplink": r.uplink_delay, "downlink": r.downlink_delay}
    radio = dataclasses.replace(r, uplink_delay=(0.0, 0.0), downlink_delay=(0.0, 0.0))
    return dataclasses.replace(cfg, radio=radio), netem


class ConfirmSink(Protocol):
    def car_confirmed(self, car_id: str, result: Result, confirm: Confirm) -> None: ...


class RemoteCar:
    """A car in another container, as the zone's registry sees it (Zone._cars)."""

    def __init__(self, car_id: str, url: str, zone: ConfirmSink, net: NetLoop) -> None:
        self.car_id = car_id
        self._url = url
        self._zone = zone
        self._net = net  # the zone process's one network thread (testbed/netloop.py)

    def budget(self, task: Task, now: float) -> float:
        raise RuntimeError("testbed: B_car comes with the upload (Upload.b_car), never from the zone's clock")

    def receive(self, result: Result, now: float) -> Confirm | None:
        def done(status: int, content: bytes) -> None:
            counts = getattr(self._zone, "counts", None)
            if status != 200:
                if counts is not None:
                    counts[f"POST /results answered {status}"] += 1
                return  # no answer: no confirm; the anchor's T_ack timer notices (9.5.4)
            confirm = decode(content)
            if isinstance(confirm, Confirm):
                self._zone.car_confirmed(self.car_id, result, confirm)
            elif counts is not None:
                counts["POST /results: 200 without a confirm"] += 1

        self._net.request("POST", self._url, "/results", encode(result), done)
        return None
