"""The training budget in decisions and the baselines' ε schedule (final-design 16.1, 18 RL row; PLAN M7).

16.1 fairness: every learning method trains on the same number of decisions. One Budget is shared by the
agents of all zones; each decision an agent makes counts once; once it is used up the agents stop learning
(they still act). ε for the learning baselines: linear from rl.epsilon_baselines.start (1) to .end (0.01)
over the first .fraction (20 %) of the budget, then .end (18; Tang & Wong's schedule).
"""

from __future__ import annotations

from dataclasses import dataclass

from pcnme.config import EpsilonSchedule


@dataclass
class Budget:
    total: int  # decisions, over every zone
    used: int = 0

    @property
    def exhausted(self) -> bool:
        return self.used >= self.total

    def spend(self) -> None:
        """One decision; the count stops at the total, so every method ends on the same number."""
        self.used = min(self.used + 1, self.total)


def epsilon(schedule: EpsilonSchedule, used: int, total: int) -> float:
    """ε after `used` of `total` budget decisions."""
    ramp = schedule.fraction * total
    if ramp <= 0 or used >= ramp:
        return schedule.end
    return schedule.start + (schedule.end - schedule.start) * used / ramp
