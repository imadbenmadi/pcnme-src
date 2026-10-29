"""Transitions (final-design 11.3; PLAN M4.3). THIS TOUCHES THE AGENT.

One step = one placement decision at an anchor (11.3.1). s = the 48 inputs at the decision, a = the option
taken, s' = the same anchor's inputs at its next decision, whichever task that is for (QUESTIONS Q36), Δt =
the time between the two decisions (11.3.2). The reward comes later than s' (11.3.3): the transition waits in
a pending buffer keyed by (task id, attempt) and moves to the replay buffer when its reward is final, at the
task's close + G, and its s' is known. Boulders and all-masked drops make no decision, so no transition
(11.3.5). There is no terminal state (11.3.4): at the end of an episode every transition whose reward is final
is moved as truncated, not terminal; the anchor's last decision has no s' (Q36); a transition whose reward is
not final yet is discarded and counted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pcnme.agent.inputs import Decision
from pcnme.agent.reward import AttemptReward

Key = tuple[object, int]  # (task id, attempt) at this anchor (5.5)


@dataclass(frozen=True, slots=True)
class Transition:
    """(s, a, r, s', Δt) at one anchor. valid / valid_next: the open actions at s and s' (11.3.7: the
    target's argmax uses only the actions valid in s'). s_next is None only for a truncated transition whose
    anchor made no later decision."""

    anchor: str
    task_id: object
    attempt: int
    time: float
    s: tuple[float, ...]
    a: int
    valid: tuple[bool, ...]
    r: float
    s_next: tuple[float, ...] | None
    valid_next: tuple[bool, ...] | None
    dt: float | None
    next_key: Key | None
    truncated: bool = False


@dataclass(slots=True)
class _Pending:
    decision: Decision
    reward: float | None = None
    next: Decision | None = None


class TransitionBuffer:
    """One anchor's pending buffer and its replay output (11.3.2-11.3.5)."""

    def __init__(self, anchor: str) -> None:
        self.anchor = anchor
        self.pending: dict[Key, _Pending] = {}
        self.replay: list[Transition] = []
        self.discarded = 0  # episode end: reward not final yet
        self._last: Key | None = None

    def on_decision(self, d: Decision) -> None:
        """A decision at this anchor: it is s' of the previous one (11.3.2), and its own transition waits."""
        if d.anchor != self.anchor:
            raise ValueError(f"decision of {d.anchor} given to {self.anchor}'s buffer")
        key: Key = (d.task_id, d.attempt)
        if key in self.pending:
            raise ValueError(f"a second decision for {key} at {self.anchor}")
        prev = self._last
        self.pending[key] = _Pending(d)
        self._last = key
        if prev is not None and prev in self.pending:
            self.pending[prev].next = d
            self._move(prev)

    def on_rewards(self, task_id: object, rewards: Mapping[int, AttemptReward]) -> None:
        """A task's rewards are final (close + G, 11.3.3). Attempts no decision made have no transition."""
        for attempt, rw in rewards.items():
            key: Key = (task_id, attempt)
            if rw.decided and key in self.pending:
                self.pending[key].reward = rw.reward
                self._move(key)

    def end_episode(self) -> None:
        """11.3.4: the episode ends by a time limit: every transition with a final reward goes to the replay
        as truncated (it still bootstraps from s' when it has one); the rest are discarded."""
        for key in list(self.pending):
            p = self.pending.pop(key)
            if p.reward is None:
                self.discarded += 1
                continue
            self.replay.append(self._transition(p, truncated=True))
        self._last = None

    def _move(self, key: Key) -> None:
        p = self.pending[key]
        if p.reward is not None and p.next is not None:
            del self.pending[key]
            self.replay.append(self._transition(p, truncated=False))

    def _transition(self, p: _Pending, truncated: bool) -> Transition:
        d, n = p.decision, p.next
        assert p.reward is not None
        return Transition(
            anchor=self.anchor,
            task_id=d.task_id,
            attempt=d.attempt,
            time=d.time,
            s=d.inputs,
            a=d.action,
            valid=d.valid,
            r=p.reward,
            s_next=None if n is None else n.inputs,
            valid_next=None if n is None else n.valid,
            dt=None if n is None else n.time - d.time,
            next_key=None if n is None else (n.task_id, n.attempt),
            truncated=truncated,
        )
