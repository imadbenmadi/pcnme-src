"""The executor of one node: slots, queues, slot-type choice and runs (final-design 8.1-8.5, 6C, 12.2).

Every node, fog or cloud, runs this same executor. A node has c_GPU GPU slots and c_CPU CPU slots; one task
holds one slot for its whole run (8.1). There is one queue per slot type, ordered hard before soft, earliest
deadline first on the executor's own d_k, ties by arrival time then task id (8.2). A task joins the queue of
the slot type that would finish it earliest, or is refused "full" or "late" at once (8.3).
"""

from __future__ import annotations

import bisect
import heapq
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from pcnme.config import SLOT_TYPES, Config, HardwareClass, SlotType, TaskClass
from pcnme.node.timetable import RunRow, RunTimeTable
from pcnme.sim.clock import EventClock
from pcnme.sim.rng import KeyPart, RandomStreams


@dataclass(frozen=True, slots=True, order=True)
class AttemptKey:
    """(task_id, anchor_id, attempt): names one attempt everywhere (final-design 5.5)."""

    task_id: tuple[KeyPart, ...]  # (car_id, epoch, seq)
    anchor_id: str
    attempt: int


@dataclass(frozen=True, slots=True)
class TaskRequest:
    """What an executor receives with a task (6C step 9): its budget B_k, a duration, not a clock time."""

    key: AttemptKey
    task_type: str
    cls: TaskClass
    l_max: float  # how late a soft task may still be delivered (= D, final-design 2)
    budget: float  # B_k; the executor sets d_k = arrival + B_k


@dataclass(frozen=True, slots=True)
class Accepted:
    slot: SlotType
    wait: float  # Ŵ of the chosen slot type, as a duration (8.3 step 4)


@dataclass(frozen=True, slots=True)
class Refused:
    reason: Literal["full", "late"]


Reply = Accepted | Refused
Outcome = Literal["done", "skipped"]
Estimate = Literal["mean", "high"]


@dataclass(frozen=True, slots=True)
class Report:
    """The end of an attempt on this executor, reported at once (8.6)."""

    key: AttemptKey
    task_type: str
    outcome: Outcome
    slot: SlotType  # the slot type it ran on, or whose queue it left
    start: float | None  # None if it never started
    finish: float  # when the slot finished with it, or when it was skipped
    busy: float  # how long it held the slot (T_run, 12.2)
    energy: float  # execution energy: PUE x p x T_run (12.2)


TraceEntry = tuple[str, float, AttemptKey, str]  # (event, time, attempt key, detail): for the event log


def choose_slot(finishes: Mapping[SlotType, float]) -> SlotType:
    """The slot type with the earliest estimated finish (8.3 step 2). Ties go to the GPU: an engineering
    choice, the design does not say."""
    return min(finishes, key=lambda q: (finishes[q], SLOT_TYPES.index(q)))


@dataclass(slots=True)
class _Queued:
    req: TaskRequest
    d_k: float
    arrival: float
    slot: SlotType
    mean: float  # Ê on this queue's slot type

    @property
    def latest(self) -> float:
        """The last useful finish: d_k for a hard task, d_k + L_max for a soft one (8.3, 8.4, 8.6)."""
        return self.d_k if self.req.cls == "hard" else self.d_k + self.req.l_max

    @property
    def sort_key(self) -> tuple[object, ...]:
        # 8.2: hard before soft; EDF on d_k; ties by arrival time, then task id. The anchor id and attempt
        # make the order total (two attempts of one task at one executor).
        k = self.req.key
        return (0 if self.req.cls == "hard" else 1, self.d_k, self.arrival, k.task_id, k.anchor_id, k.attempt)


@dataclass(slots=True)
class _Running:
    task: _Queued
    row: RunRow
    start: float
    end: float


class Executor:
    """One node's executor on a simulated clock."""

    def __init__(
        self,
        clock: EventClock,
        streams: RandomStreams,
        hw: HardwareClass,
        table: RunTimeTable,
        *,
        reserve_hard: int,
        pue: float,
        a_max: float,
        promotion_estimate: Estimate,
        on_report: Callable[[Report], None],
        trace: Callable[[TraceEntry], None] | None = None,
    ) -> None:
        if promotion_estimate not in ("mean", "high"):
            raise ValueError(
                f"promotion_estimate must be 'mean' or 'high' (QUESTIONS Q10), got {promotion_estimate!r}"
            )
        self._clock = clock
        self._streams = streams
        self._hw = hw
        self._table = table
        self._reserve = reserve_hard
        self._pue = pue
        self._a_max = a_max
        self._promotion = promotion_estimate
        self._on_report = on_report
        self._trace = trace
        self._queues: dict[SlotType, list[_Queued]] = {q: [] for q in hw.slot_types}
        self._keys: dict[SlotType, list[tuple[object, ...]]] = {q: [] for q in hw.slot_types}
        self._running: dict[SlotType, list[_Running]] = {q: [] for q in hw.slot_types}

    @classmethod
    def from_config(
        cls,
        clock: EventClock,
        streams: RandomStreams,
        config: Config,
        hw_name: str,
        table: RunTimeTable,
        *,
        promotion_estimate: Estimate,
        on_report: Callable[[Report], None],
        trace: Callable[[TraceEntry], None] | None = None,
    ) -> Executor:
        hw = config.hardware[hw_name]
        return cls(
            clock,
            streams,
            hw,
            table,
            reserve_hard=config.scheduler.reserve_hard_r,
            pue=config.energy.pue(hw),
            a_max=config.scheduler.soft_promotion_a_max,
            promotion_estimate=promotion_estimate,
            on_report=on_report,
            trace=trace,
        )

    # --- reading the state ----------------------------------------------------------------------------------

    def queue_length(self, q: SlotType) -> int:
        return len(self._queues[q])

    def busy(self, q: SlotType) -> int:
        return len(self._running[q])

    def _remaining(self, q: SlotType, now: float) -> float:
        """R_q: the remaining part of the running tasks' mean runs, max(0, Ê - elapsed) each (8.5; Q9)."""
        return sum(max(0.0, r.start + r.row.mean - now) for r in self._running[q])

    def wait_estimate(self, q: SlotType, cls: TaskClass) -> float:
        """Ŵ_q for a new task of class `cls`: 8.5 on the executor's exact state (QUESTIONS Q8, Q9).

        0 while a slot is free; otherwise (R + B_hard) / c for hard and (R + B_hard + B_soft) / c for soft,
        with R = the remaining part of the running tasks' mean runs and B = the queued tasks' mean runs.
        """
        c = self._hw.slots(q)
        if self.busy(q) < c:
            return 0.0
        now = self._clock.now
        remaining = self._remaining(q, now)
        b_hard = sum(t.mean for t in self._queues[q] if t.req.cls == "hard")
        b_soft = sum(t.mean for t in self._queues[q] if t.req.cls == "soft")
        backlog = remaining + b_hard + (b_soft if cls == "soft" else 0.0)
        return backlog / c

    # --- arrival (8.3) --------------------------------------------------------------------------------------

    def submit(self, req: TaskRequest) -> Reply:
        """A task arrives now: accept it into the queue of its best slot type, or refuse it at once."""
        now = self._clock.now
        d_k = now + req.budget  # 6C step 9
        latest = d_k if req.cls == "hard" else d_k + req.l_max  # 8.3.3
        runnable = [q for q in self._table.slot_types(req.task_type, self._hw.name) if self._hw.slots(q) > 0]
        if not runnable:
            return self._refuse(req, "late")  # 8.3.3: no slot type here can run it
        usable = [q for q in runnable if self._has_place(q, req.cls)]
        if not usable:
            return self._refuse(req, "full")  # 8.3.3: no usable queue has a place for its class
        waits = {q: self.wait_estimate(q, req.cls) for q in usable}
        finishes = {q: waits[q] + self._row(req.task_type, q).mean for q in usable}  # F_q = Ŵ_q + Ê_q (8.3.1)
        q = choose_slot(finishes)
        if now + finishes[q] > latest:
            return self._refuse(req, "late")  # 8.3.3: hard misses d_k; soft is later than d_k + L_max
        task = _Queued(req=req, d_k=d_k, arrival=now, slot=q, mean=self._row(req.task_type, q).mean)
        self._enqueue(task)
        self._log("accept", req.key, q)
        self._dispatch()
        return Accepted(slot=q, wait=waits[q])

    def _has_place(self, q: SlotType, cls: TaskClass) -> bool:
        """8.2.3: capacity Q_q; the last r places are for hard tasks."""
        cap = self._hw.queue[q]
        return len(self._queues[q]) < (cap if cls == "hard" else cap - self._reserve)

    def _refuse(self, req: TaskRequest, reason: Literal["full", "late"]) -> Refused:
        self._log("refuse", req.key, reason)
        return Refused(reason=reason)

    def _enqueue(self, task: _Queued) -> None:
        key = task.sort_key
        i = bisect.bisect_right(self._keys[task.slot], key)
        self._keys[task.slot].insert(i, key)
        self._queues[task.slot].insert(i, task)

    def _dequeue(self, task: _Queued) -> None:
        i = self._queues[task.slot].index(task)
        del self._queues[task.slot][i]
        del self._keys[task.slot][i]

    # --- a free slot takes a task (8.4) ---------------------------------------------------------------------

    def _dispatch(self) -> None:
        for q in self._hw.slot_types:  # GPU slots first: deterministic
            while self.busy(q) < self._hw.slots(q):
                task = self._pick(q)
                if task is None:
                    break  # 8.4 step 6: the slot idles
                self._start(q, task)

    def _pick(self, q: SlotType) -> _Queued | None:
        """8.4: which task a free slot of type q takes, or None (the slot idles, step 6)."""
        now = self._clock.now
        self._skip_hopeless_heads(q, now)  # step 1
        queue = self._queues[q]
        hard = next((t for t in queue if t.req.cls == "hard"), None)
        soft = next((t for t in queue if t.req.cls == "soft"), None)
        if soft is not None and now - soft.arrival >= self._a_max and self._promotion_is_safe(q, soft, now):
            return soft  # step 2: soft promotion
        if hard is not None:
            return hard  # step 3
        if soft is not None:
            return soft  # step 4
        return self._steal(q, now)  # step 5 (None: step 6)

    def _hopeless(self, task: _Queued, q: SlotType, now: float) -> bool:
        """8.4 step 1: t + Ê_q^low > d_k (hard) or > d_k + L_max (soft)."""
        return now + self._row(task.req.task_type, q).low > task.latest

    def _skip_hopeless_heads(self, q: SlotType, now: float) -> None:
        """8.4 step 1: skip the head of each class while it is hopeless; each skip is reported at once."""
        for cls in ("hard", "soft"):
            while True:
                head = next((t for t in self._queues[q] if t.req.cls == cls), None)
                if head is None or not self._hopeless(head, q, now):
                    break
                self._skip(head, now)

    def _skip(self, task: _Queued, now: float) -> None:
        self._dequeue(task)
        report = Report(
            key=task.req.key,
            task_type=task.req.task_type,
            outcome="skipped",
            slot=task.slot,
            start=None,
            finish=now,
            busy=0.0,
            energy=0.0,  # 12.5: a skipped attempt spent forwarding only, which the anchor counts
        )
        self._log("skip", task.req.key, task.slot)
        self._on_report(report)

    def _promotion_is_safe(self, q: SlotType, soft: _Queued, now: float) -> bool:
        """8.4 step 2: with `soft` on this slot for Ê^high, list-schedule the queued hard tasks (EDF order) on
        the c_q slots; every one must finish by its d_k. Run times: the mean or the high (QUESTIONS Q10)."""
        hard = [t for t in self._queues[q] if t.req.cls == "hard"]
        if not hard:
            return True
        free_at = [now + self._row(soft.req.task_type, q).high]  # this slot
        for r in self._running[q]:
            run = r.row.mean if self._promotion == "mean" else r.row.high
            free_at.append(now + max(0.0, r.start + run - now))
        free_at += [now] * (self._hw.slots(q) - 1 - len(self._running[q]))  # other free slots
        heapq.heapify(free_at)
        for t in hard:
            row = self._row(t.req.task_type, q)
            end = heapq.heappop(free_at) + (row.mean if self._promotion == "mean" else row.high)
            if end > t.d_k:
                return False
            heapq.heappush(free_at, end)
        return True

    def _steal(self, q: SlotType, now: float) -> _Queued | None:
        """8.4 step 5: this queue is empty; take the first task in another queue's order that can run here,
        would finish earlier here than its current estimated finish (QUESTIONS Q11) and passes the skip rule
        here."""
        if self._queues[q]:
            return None
        for other in self._hw.slot_types:
            if other == q:
                continue
            for i, task in enumerate(self._queues[other]):
                row = self._table.row(task.req.task_type, self._hw.name, q)
                if row is None:
                    continue  # cannot run here
                if now + row.mean < self._current_finish(other, i, now) and not self._hopeless(task, q, now):
                    self._log("steal", task.req.key, f"{other}->{q}")
                    return task
        return None

    def _current_finish(self, q: SlotType, index: int, now: float) -> float:
        """The current estimated finish of the task at `index` in q's queue: now + (R + the mean runs of the
        tasks ahead of it) / c + its own Ê (8.5 on what is ahead of it; QUESTIONS Q11)."""
        queue = self._queues[q]
        c = self._hw.slots(q)
        if self.busy(q) < c:
            wait = 0.0
        else:
            remaining = self._remaining(q, now)
            wait = (remaining + sum(t.mean for t in queue[:index])) / c
        return now + wait + queue[index].mean

    def _start(self, q: SlotType, task: _Queued) -> None:
        self._dequeue(task)
        row = self._row(task.req.task_type, q)
        k = task.req.key
        run = self._table.draw(self._streams, row, k.task_id, k.attempt)  # keyed by task (CHANGES C3)
        now = self._clock.now
        running = _Running(task=task, row=row, start=now, end=now + run)
        self._running[q].append(running)
        self._log("start", k, q)
        self._clock.schedule(running.end, lambda: self._finish(q, running))

    def _finish(self, q: SlotType, running: _Running) -> None:
        self._running[q].remove(running)
        busy = running.end - running.start
        report = Report(
            key=running.task.req.key,
            task_type=running.task.req.task_type,
            outcome="done",
            slot=q,
            start=running.start,
            finish=running.end,
            busy=busy,
            energy=self._pue * running.row.power * busy,  # 12.2: PUE_k x p x T_run
        )
        self._log("done", report.key, q)
        self._on_report(report)
        self._dispatch()

    # --- helpers --------------------------------------------------------------------------------------------

    def _row(self, task_type: str, q: SlotType) -> RunRow:
        row = self._table.row(task_type, self._hw.name, q)
        if row is None:
            raise KeyError(f"{task_type} has no time-table row on {self._hw.name} {q}")
        return row

    def _log(self, event: str, key: AttemptKey, detail: str) -> None:
        if self._trace is not None:
            self._trace((event, self._clock.now, key, detail))
