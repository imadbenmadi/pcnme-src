"""A cloud transfer: the zone's own backhaul pipe, then the WAN pipe shared by all zones (final-design 4.1).

    T = W_bh + d_bh + W_WAN + S / B_WAN + d_WAN

Packets flow through both pipes at once, so the transfer reaches the WAN pipe d_bh after it starts on the
backhaul pipe, and the WAN pipe, the narrowest, sets the push time: S / B_bh is not part of T. The zone's own
backhaul pipe is still busy for S / B_bh, as every pipe is for what it pushes (4.1).

The WAN is entered by a clock event at the entry time, so transfers from different zones join the shared
pipe in the order they reach it, not the order they were sent.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from pcnme.net.pipe import FifoPipe
from pcnme.sim.clock import EventClock


@dataclass(frozen=True, slots=True)
class SeriesTransfer:
    """One cloud transfer, in seconds."""

    start: float  # handed to the zone's backhaul pipe
    w_bh: float  # W_bh: wait in the zone's own backhaul pipe (known exactly, 4.3)
    d_bh: float
    wan_entry: float  # start + W_bh + d_bh
    w_wan: float  # W_WAN: wait in the shared WAN pipe
    push_wan: float  # S / B_WAN
    d_wan: float
    delivery: float  # wan_entry + W_WAN + S / B_WAN + d_WAN

    @property
    def duration(self) -> float:
        """T = W_bh + d_bh + W_WAN + S / B_WAN + d_WAN (4.1)."""
        return self.delivery - self.start


class CloudPath:
    """One zone's path to the cloud: its backhaul pipe, then the shared WAN pipe."""

    def __init__(self, clock: EventClock, backhaul: FifoPipe, wan: FifoPipe) -> None:
        if wan.bandwidth is None or backhaul.bandwidth is None:
            raise ValueError("both pipes of a cloud path need a fixed bandwidth")
        if wan.bandwidth > backhaul.bandwidth:
            # final-design 4.1 and 4.2: the WAN is the narrowest pipe and sets the push time.
            raise ValueError(
                f"the WAN ({wan.bandwidth:g} bit/s) must be the narrow pipe, "
                f"not wider than the backhaul ({backhaul.bandwidth:g} bit/s)"
            )
        self._clock = clock
        self._backhaul = backhaul
        self._wan = wan

    def send(
        self,
        size_bits: float,
        d_bh: float,
        d_wan: float,
        on_delivered: Callable[[SeriesTransfer], None],
    ) -> None:
        """Send `size_bits` to the cloud now. The delays are drawn by the caller (keyed by task, attempt and
        hop: CHANGES C3); `on_delivered` runs at the delivery time."""
        for name, d in (("backhaul", d_bh), ("WAN", d_wan)):
            if not (math.isfinite(d) and d >= 0):  # checked before any pipe changes
                raise ValueError(f"the {name} delay must be a finite number >= 0, got {d!r}")
        start = self._clock.now
        # The zone's own pipe: its wait, and busy for its own push S / B_bh. The backhaul delay is applied
        # before the WAN, not by this pipe, so it is 0 here.
        bh = self._backhaul.send(now=start, size_bits=size_bits, delay=0.0)
        entry = start + bh.wait + d_bh  # reaches the WAN pipe d_bh after starting on the backhaul (4.1)

        def enter_wan() -> None:
            wan = self._wan.send(now=entry, size_bits=size_bits, delay=d_wan)
            transfer = SeriesTransfer(
                start=start,
                w_bh=bh.wait,
                d_bh=d_bh,
                wan_entry=entry,
                w_wan=wan.wait,
                push_wan=wan.push,
                d_wan=d_wan,
                delivery=wan.delivery,
            )
            self._clock.schedule(wan.delivery, lambda: on_delivered(transfer))

        self._clock.schedule(entry, enter_wan)
