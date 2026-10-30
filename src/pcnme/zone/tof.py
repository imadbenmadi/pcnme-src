"""The TOF-Broker, the fixed first filter at the zone entry (final-design 7.1).

1. Boulder iff Ê_strong >= α · D (the mean run on the strongest fog class's fastest slot type).
2. A boulder goes to the cloud, never to a fog queue.
3. A boulder is dropped at intake when the cloud fails "cannot meet the deadline": T̂_in,cloud + Ê_cloud +
   T̂_ret,cloud > time left (hard), > time left + L_max (soft); the cloud never queues. No agent decision is
   made, so no transition is stored. A pebble goes to the agent (the placement method).
"""

from __future__ import annotations

from typing import Literal

from pcnme.config import Config, DeadlineMode, TaskClass
from pcnme.zone.estimates import PathEstimates

Route = Literal["agent", "cloud", "drop"]


class TofBroker:
    def __init__(self, config: Config, estimates: PathEstimates) -> None:
        self._cfg = config
        self._est = estimates

    def is_boulder(self, task_type: str, mode: DeadlineMode) -> bool:
        if not self._cfg.ablation.tof:
            return False  # 16.4: no TOF-Broker (α = ∞)
        return self._cfg.is_boulder(task_type, mode)  # 7.1 step 1 (Ê_strong >= α D)

    def classify(
        self,
        task_type: str,
        mode: DeadlineMode,
        cls: TaskClass,
        size_bits: float,
        time_left: float,
        l_max: float,
        backhaul_wait: float,
    ) -> Route:
        if not self.is_boulder(task_type, mode):
            return "agent"
        cloud = self._est.t_in_cloud(size_bits, backhaul_wait) + self._est.e_cloud(task_type)
        cloud += self._est.t_ret_cloud()
        limit = time_left if cls == "hard" else time_left + l_max  # 7.1 step 3
        return "drop" if cloud > limit else "cloud"
