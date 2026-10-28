"""The radio uplink of a zone (final-design 4.2, 18; CHANGES C1).

A car's rate is the Shannon rate at its distance with free-space path loss (the VFC model, noise in correct
units, C1): R = B log2(1 + P G_t G_r (lambda / (4 pi d))^2 / (N0 B)), lambda = c / f. One uplink per zone,
shared by all its cars, one car at a time: a FIFO pipe whose push time is S / R_car.
"""

from __future__ import annotations

import math

from pcnme.config import Radio
from pcnme.net.pipe import FifoPipe, Transfer

SPEED_OF_LIGHT = 299_792_458.0  # m/s
MIN_DISTANCE = 1.0  # m: engineering choice, free-space loss has no limit as d -> 0


def shannon_rate(distance: float, radio: Radio) -> float:
    """bit/s at `distance` metres from the access point (4.2, 18)."""
    d = max(distance, MIN_DISTANCE)
    wavelength = SPEED_OF_LIGHT / radio.carrier_hz
    received = radio.tx_power_w * radio.gain_tx * radio.gain_rx * (wavelength / (4 * math.pi * d)) ** 2
    noise = radio.noise_w_per_hz * radio.bandwidth_hz  # 10^-20.4 W/Hz x B (C1)
    return radio.bandwidth_hz * math.log2(1 + received / noise)


class RadioUplink:
    """One zone's uplink: a FIFO pipe with no fixed bandwidth; each upload pushes at its car's rate (4.2)."""

    def __init__(self) -> None:
        self._pipe = FifoPipe(None)

    def wait_at(self, now: float) -> float:
        """W_radio: known exactly, because the zone owns its uplink (6A.3)."""
        return self._pipe.wait_at(now)

    def send(self, now: float, size_bits: float, rate: float, delay: float) -> Transfer:
        return self._pipe.send(now=now, size_bits=size_bits, delay=delay, rate=rate)
