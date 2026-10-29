"""Every hop a message or transfer crosses, its fixed name, and its random delay (CHANGES C3; QUESTIONS Q5,
Q17).

A delay is drawn by task, not by event order: u keyed by (task_id, attempt, hop) on the hop's stream, then
d = low + u (high - low) on the hop's uniform interval from the config (4.2, 18). Radio hops use the radio
stream (Q5); backhaul and WAN hops the link_delays stream. MQTT messages draw their delays in the bus
(bus/bus.py), keyed the same way. The upload comes before any attempt, so it uses
attempt 0 (Q17).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from pcnme.sim.quantize import to_ns
from pcnme.sim.rng import KeyPart, RandomStreams

RADIO_UP = "radio_up"  # car -> access point: the upload (and later the car's confirm)
RADIO_DOWN = "radio_down"  # access point -> car: a result
BH_OUT = "bh_out"  # a zone's outgoing backhaul: to a neighbour, or the first part of a cloud transfer
WAN_IN = "wan_in"  # metro-edge router -> cloud: the WAN part of a cloud transfer
ACK_UP = "ack_up"  # car -> serving zone: the car's confirm (radio)

STREAM: Mapping[str, str] = MappingProxyType(
    {
        RADIO_UP: "radio",
        RADIO_DOWN: "radio",
        BH_OUT: "link_delays",
        WAN_IN: "link_delays",
        ACK_UP: "radio",
    }
)

UPLOAD_ATTEMPT = 0  # QUESTIONS Q17: attempts count from 1, the upload comes before them


def delay(
    streams: RandomStreams,
    hop: str,
    interval: tuple[float, float],
    task_id: tuple[KeyPart, ...],
    attempt: int,
) -> float:
    """The delay of one crossing of `hop`, uniform on `interval` (s), keyed by (task_id, attempt, hop), in
    whole nanoseconds like every drawn time (sim/quantize.py)."""
    low, high = interval
    u = streams.uniform(STREAM[hop], *task_id, attempt, hop)
    return to_ns(low + u * (high - low))
