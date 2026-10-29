"""The anchor's path estimates (final-design 4.3; QUESTIONS Q16).

Bandwidths and mean delays are configuration; only pipe waits change (4.3):

    T̂_in,k = ŵ_k + S / B_k + δ̄_k        here: 0;  cloud: own backhaul wait + ŵ_WAN, B_WAN, δ̄_bh + δ̄_WAN
    T̂_ret,k = the sum of the mean delays on the result path, down to the car (Q16):
        fog executor: backhaul to the broker + broker processing + backhaul to the zone + radio downlink
        cloud: WAN + broker processing + backhaul to the zone + radio downlink

ŵ_WAN comes from a WanWait source; M2 uses ZeroWanWait ("before any sample, ŵ_WAN = 0", 4.3), which M3.3
replaces with the state plane's moving average.
"""

from __future__ import annotations

from typing import Protocol

from pcnme.config import Config


class WanWait(Protocol):
    """Where ŵ_WAN comes from (4.3)."""

    def estimate(self) -> float: ...


class ZeroWanWait:
    """M3 PLACEHOLDER (M3.3): ŵ_WAN = 0, as before any sample (4.3)."""

    def estimate(self) -> float:
        return 0.0


def _mean(interval: tuple[float, float]) -> float:
    return (interval[0] + interval[1]) / 2


class PathEstimates:
    def __init__(self, config: Config, wan_wait: WanWait) -> None:
        self._cfg = config
        self._wan_wait = wan_wait
        self._bh = _mean(config.links.backhaul_delay)
        self._wan = _mean(config.links.wan_delay)
        self._broker = config.links.broker_extra_delay
        self._down = _mean(config.radio.downlink_delay)

    def t_in_here(self) -> float:
        return 0.0

    def t_in_cloud(self, size_bits: float, backhaul_wait: float) -> float:
        """4.3: own backhaul wait (exact) + ŵ_WAN + S / B_WAN + δ̄_bh + δ̄_WAN."""
        wait = backhaul_wait + self._wan_wait.estimate()
        return wait + size_bits / self._cfg.derived.b_wan + self._bh + self._wan

    def t_in_neighbour(self, size_bits: float, backhaul_wait: float) -> float:
        """4.3: own backhaul wait (exact) + S / B_bh + δ̄_bh."""
        return backhaul_wait + size_bits / self._cfg.links.backhaul_bw + self._bh

    def t_ret_fog(self) -> float:
        return self._bh + self._broker + self._bh + self._down

    def t_ret_cloud(self) -> float:
        return self._wan + self._broker + self._bh + self._down

    def e_cloud(self, task_type: str) -> float:
        """Ê on the cloud's fastest slot type for this task (the cloud never queues, 7.1.3)."""
        rows = self._cfg.time_table
        means = [r.mean for q in ("gpu", "cpu") if (r := rows.row(task_type, "cloud", q)) is not None]
        return min(means)
