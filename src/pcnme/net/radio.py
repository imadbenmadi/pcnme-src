"""The radio uplink of a zone (final-design 4.2, 18; CHANGES C1).

A car's rate is the Shannon rate at its distance with free-space path loss (the VFC model, noise in correct
units, C1): R = B log2(1 + P G_t G_r (lambda / (4 pi d))^2 / (N0 B)), lambda = c / f. One uplink per zone,
shared by all its cars: the cars uploading at once share its airtime equally, so while n uploads are in
progress each advances at its own R_car / n (fluid processor sharing, CHANGES C6).
"""

from __future__ import annotations

import functools
import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass

from pcnme.config import Radio
from pcnme.sim.clock import EventClock, EventHandle
from pcnme.sim.quantize import to_whole_bps

SPEED_OF_LIGHT = 299_792_458.0  # m/s
MIN_DISTANCE = 1.0  # m: engineering choice, free-space loss has no limit as d -> 0


def shannon_rate(distance: float, radio: Radio) -> float:
    """bit/s at `distance` metres from the access point (4.2, 18), in whole bit/s: log2, pi and the car's
    position (cos, sin) may differ in the last bit between platforms (sim/quantize.py)."""
    d = max(distance, MIN_DISTANCE)
    wavelength = SPEED_OF_LIGHT / radio.carrier_hz
    received = radio.tx_power_w * radio.gain_tx * radio.gain_rx * (wavelength / (4 * math.pi * d)) ** 2
    noise = radio.noise_w_per_hz * radio.bandwidth_hz  # 10^-20.4 W/Hz x B (C1)
    return to_whole_bps(radio.bandwidth_hz * math.log2(1 + received / noise))


@dataclass(frozen=True, slots=True)
class RadioTransfer:
    """One upload through the zone's uplink, in seconds."""

    start: float  # the car starts pushing
    push: float  # S / R_car: its push time alone on the air
    push_end: float  # its last bit is on the air
    delay: float  # the radio delay d
    delivery: float  # push_end + d: the whole upload has arrived (t_rx)

    @property
    def wait(self) -> float:
        """W_radio: the push's extra time from sharing the air (6A.3); 0 for a car alone."""
        return self.push_end - self.start - self.push

    @property
    def duration(self) -> float:
        """The upload time U_up = W_radio + S / R_car + d."""
        return self.delivery - self.start


class RadioUplink:
    """One zone's uplink with equal airtime among the uploads in progress (4.2, CHANGES C6).

    Virtual time V advances at 1 / n while n uploads are in progress; an upload that starts at virtual time
    V_s ends when V reaches V_s + S / R_car, its push time alone. Uploads end in the order of these finish
    tags (ties by order of arrival). The zone runs its uplink, so each upload's W_radio is known exactly when
    it arrives (6A.3). The air is busy exactly when a one-car-at-a-time pipe would be: only the order of
    the ends changes.
    """

    def __init__(self, clock: EventClock) -> None:
        self._clock = clock
        self._v = 0.0  # virtual time
        self._t = 0.0  # when V was last advanced
        self._active: list[tuple[float, int, _Upload]] = []  # heap of (finish tag, arrival seq, upload)
        self._seq = 0
        self._next: EventHandle | None = None

    @property
    def in_progress(self) -> int:
        return len(self._active)

    def send(
        self, size_bits: float, rate: float, delay: float, on_delivered: Callable[[RadioTransfer], None]
    ) -> int:
        """Start an upload now; `on_delivered` runs at its delivery time with the transfer. Returns its id,
        for `cancel`."""
        if not (math.isfinite(size_bits) and size_bits > 0 and math.isfinite(rate) and rate > 0):
            raise ValueError(f"size and rate must be > 0, got {size_bits!r} bits at {rate!r} bit/s")
        if not (math.isfinite(delay) and delay >= 0):
            raise ValueError(f"the delay must be a finite number >= 0, got {delay!r}")
        now = self._clock.now
        self._advance(now)
        push = size_bits / rate
        self._seq += 1
        heapq.heappush(self._active, (self._v + push, self._seq, _Upload(now, push, delay, on_delivered)))
        self._reschedule(now)
        return self._seq

    def cancel(self, upload_id: int) -> bool:
        """9.7: an upload cut by a handover or a gap before its last bit leaves nothing behind; the others
        share the air from now on. False if its last bit was already on the air (it will arrive)."""
        now = self._clock.now
        if not any(seq == upload_id for _, seq, _ in self._active):
            return False
        self._advance(now)
        self._active = [e for e in self._active if e[1] != upload_id]
        heapq.heapify(self._active)
        self._reschedule(now)
        return True

    def _advance(self, now: float) -> None:
        if self._active:
            self._v += (now - self._t) / len(self._active)
        self._t = now

    def _reschedule(self, now: float) -> None:
        if self._next is not None:
            self._clock.cancel(self._next)
            self._next = None
        if self._active:
            tag = self._active[0][0]
            end = max(now, now + (tag - self._v) * len(self._active))  # an end at `now` may round below it
            self._next = self._clock.schedule(end, self._end_first)

    def _end_first(self) -> None:
        now = self._clock.now
        self._next = None
        tag, _, _ = self._active[0]
        self._advance(now)
        self._v = tag  # exact: the next end is this tag (no drift from the division above)
        while self._active and self._active[0][0] == tag:  # equal tags end together
            _, _, up = heapq.heappop(self._active)
            tr = RadioTransfer(up.start, up.push, now, up.delay, now + up.delay)
            self._clock.schedule(tr.delivery, functools.partial(up.on_delivered, tr))
        self._reschedule(now)


@dataclass(frozen=True, slots=True)
class _Upload:
    start: float
    push: float
    delay: float
    on_delivered: Callable[[RadioTransfer], None]
