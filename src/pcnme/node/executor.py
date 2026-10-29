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
from types import MappingProxyType
from typing import Literal

from pcnme.config import SLOT_TYPES, Config, HardwareClass, SlotType, TaskClass
from pcnme.ids import AttemptKey
from pcnme.node.state import SlotState, wait_estimate, water_fill
from pcnme.node.timetable import RunRow, RunTimeTable
from pcnme.sim.clock import EventClock, EventHandle
from pcnme.sim.rng import RandomStreams


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


@dataclass(frozen=True, slots=True)
class Ignored:
    """A repeated upload of an attempt this executor already has or kept (8.6): nothing is done. The reply
    itself is an engineering choice; the design says only that the repeat is ignored."""


Reply = Accepted | Refused | Ignored
Outcome = Literal["done", "skipped", "aborted", "cancelled"]
Status = Literal["queued", "running", "done", "skipped", "aborted", "cancelled", "not found"]
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
    end: float  # when the drawn run would end
    event: EventHandle | None = None  # its finish or abort event
    tail_end: float | None = None  # after an abort or cancel: when the slot is free again (8.6)


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
        t_keep: float,
        on_report: Callable[[Report], None],
        trace: Callable[[TraceEntry], None] | None = None,
        promotion_estimate: Estimate = "high",
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
        self._t_keep = t_keep
        self._on_report = on_report
        self._trace = trace
        self._queues: dict[SlotType, list[_Queued]] = {q: [] for q in hw.slot_types}
        self._keys: dict[SlotType, list[tuple[object, ...]]] = {q: [] for q in hw.slot_types}
        self._running: dict[SlotType, list[_Running]] = {q: [] for q in hw.slot_types}
        self._live: dict[AttemptKey, _Queued | _Running] = {}
        self._kept: dict[AttemptKey, tuple[Outcome, float]] = {}  # (outcome, time) for T_keep (8.6, 5.5)

    @classmethod
    def from_config(
        cls,
        clock: EventClock,
        streams: RandomStreams,
        config: Config,
        hw_name: str,
        table: RunTimeTable,
        *,
        on_report: Callable[[Report], None],
        trace: Callable[[TraceEntry], None] | None = None,
        promotion_estimate: Estimate = "high",
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
            t_keep=config.scheduler.t_keep,
            on_report=on_report,
            trace=trace,
        )

    # --- reading the state ----------------------------------------------------------------------------------

    def queue_length(self, q: SlotType) -> int:
        return len(self._queues[q])

    def busy(self, q: SlotType) -> int:
        return len(self._running[q])

    def _slot_remaining(self, q: SlotType, now: float) -> tuple[float, ...]:
        """r_i of each of the c_q slots, ascending: max(0, Ê - elapsed) for a running task (Q9), its exact
        tail after an abort, 0 for a free slot (8.5, CHANGES C4)."""
        busy = [self._left(r, now) for r in self._running[q]]
        return tuple(sorted(busy + [0.0] * (self._hw.slots(q) - len(busy))))

    @staticmethod
    def _left(r: _Running, now: float) -> float:
        """One slot's remaining time: exact in an abort tail, else max(0, Ê - elapsed) (8.5; Q9)."""
        if r.tail_end is not None:
            return r.tail_end - now
        return max(0.0, r.start + r.row.mean - now)

    def snapshot(self) -> Mapping[SlotType, SlotState]:
        """What this node publishes, per slot type with c_q > 0 (8.5)."""
        return MappingProxyType({q: self._slot_state(q) for q in self._hw.slot_types})

    def _slot_state(self, q: SlotType) -> SlotState:
        queue = self._queues[q]
        return SlotState(
            c=self._hw.slots(q),
            capacity=self._hw.queue[q],
            n_hard=sum(1 for t in queue if t.req.cls == "hard"),
            n_soft=sum(1 for t in queue if t.req.cls == "soft"),
            b_hard=sum(t.mean for t in queue if t.req.cls == "hard"),
            b_soft=sum(t.mean for t in queue if t.req.cls == "soft"),
            remaining=self._slot_remaining(q, self._clock.now),
            u=self.busy(q),
        )

    def wait_estimate(self, q: SlotType, cls: TaskClass) -> float:
        """Ŵ_q for a new task of class `cls`: the 8.5 estimate on this executor's exact state, with no pending
        sends (QUESTIONS Q8). One formula, shared with the anchor: `pcnme.node.state.wait_estimate`."""
        return wait_estimate(self._slot_state(q), cls)

    # --- arrival (8.3) --------------------------------------------------------------------------------------

    def submit(self, req: TaskRequest) -> Reply:
        """A task arrives now: accept it into the queue of its best slot type, or refuse it at once."""
        now = self._clock.now
        if req.key in self._live or req.key in self._kept:
            self._log("ignore", req.key, "repeat")
            return Ignored()  # 8.6: a repeated upload of the same attempt is ignored
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
        self._live[req.key] = task
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
        self._remove_queued(task, "skipped", now)

    def _remove_queued(self, task: _Queued, outcome: Outcome, now: float) -> None:
        """A queued task leaves without running: skipped (8.4 step 1) or cancelled (8.6). Reported at once;
        no execution energy (12.5: forwarding only, which the anchor counts)."""
        self._dequeue(task)
        del self._live[task.req.key]
        report = Report(
            key=task.req.key,
            task_type=task.req.task_type,
            outcome=outcome,
            slot=task.slot,
            start=None,
            finish=now,
            busy=0.0,
            energy=0.0,
        )
        self._log(outcome, task.req.key, task.slot)
        self._keep(task.req.key, outcome)
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
            left = r.tail_end - now if r.tail_end is not None else max(0.0, r.start + run - now)
            free_at.append(now + left)
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
        else:  # the C4 estimate over the tasks ahead of it
            wait = water_fill(self._slot_remaining(q, now), sum(t.mean for t in queue[:index]))
        return now + wait + queue[index].mean

    def _start(self, q: SlotType, task: _Queued) -> None:
        self._dequeue(task)
        row = self._row(task.req.task_type, q)
        k = task.req.key
        run = self._table.draw(self._streams, row, k.task_id, k.attempt)  # keyed by task (CHANGES C3)
        now = self._clock.now
        running = _Running(task=task, row=row, start=now, end=now + run)
        self._running[q].append(running)
        self._live[k] = running
        self._log("start", k, q)
        latest = task.latest
        if running.end > latest:
            # 8.6: aborted while running when t > d_k (hard) or d_k + L_max (soft); the run is drawn, so the
            # abort time is known now. Finishing exactly at the limit is in time.
            running.event = self._clock.schedule(latest, lambda: self._abort(q, running, "aborted"))
        else:
            running.event = self._clock.schedule(running.end, lambda: self._finish(q, running))

    def _finish(self, q: SlotType, running: _Running) -> None:
        self._running[q].remove(running)
        del self._live[running.task.req.key]
        self._report_run(q, running, "done", running.end)
        self._dispatch()

    def _abort(self, q: SlotType, running: _Running, outcome: Outcome) -> None:
        """Stop a running task for good (8.6): the worker stops at the end of its current DAG step, so the
        slot stays busy for min(remaining, d_abort), d_abort = the longest step of the task on this slot type.
        The drop is reported at once with the energy of the whole busy time (12.2, 12.5)."""
        now = self._clock.now
        if running.event is not None:
            self._clock.cancel(running.event)
        running.tail_end = now + min(running.end - now, running.row.longest_step)
        del self._live[running.task.req.key]
        self._report_run(q, running, outcome, running.tail_end)
        running.event = self._clock.schedule(running.tail_end, lambda: self._release(q, running))

    def _release(self, q: SlotType, running: _Running) -> None:
        """The abort tail is over: the slot is free (8.6 "no leftovers")."""
        self._running[q].remove(running)
        self._dispatch()

    def _report_run(self, q: SlotType, running: _Running, outcome: Outcome, slot_free: float) -> None:
        busy = slot_free - running.start
        report = Report(
            key=running.task.req.key,
            task_type=running.task.req.task_type,
            outcome=outcome,
            slot=q,
            start=running.start,
            finish=slot_free,
            busy=busy,
            energy=self._pue * running.row.power * busy,  # 12.2: PUE_k x p x T_run, the d_abort tail included
        )
        self._log(outcome, report.key, q)
        self._keep(report.key, outcome)
        self._on_report(report)

    # --- cancel, status and T_keep (8.6, 9.3, 9.4, 5.5) -----------------------------------------------------

    def cancel(self, key: AttemptKey) -> bool:
        """The anchor's cancel (9.4): removed if queued, aborted if running; unknown or finished: ignored.
        True if it removed or aborted something; sending it twice is harmless."""
        live = self._live.get(key)
        if live is None:
            return False
        if isinstance(live, _Queued):
            self._remove_queued(live, "cancelled", self._clock.now)
            return True
        q = next(q for q, runs in self._running.items() if live in runs)
        self._abort(q, live, "cancelled")
        return True

    def status(self, key: AttemptKey) -> Status:
        """What the executor knows of an attempt, for the status probe (9.3): live, kept, or not found."""
        live = self._live.get(key)
        if live is not None:
            return "queued" if isinstance(live, _Queued) else "running"
        kept = self._kept.get(key)
        return kept[0] if kept is not None else "not found"

    def probe(self, key: AttemptKey) -> tuple[Status, float | None]:
        """GET /tasks/{id}?attempt=n&anchor=A (5.3, 9.3): queued with its wait (the C4 estimate over the tasks
        ahead of it), running with its remaining time at Ê^high, or done, skipped, aborted, cancelled, not
        found (no number)."""
        now = self._clock.now
        live = self._live.get(key)
        if isinstance(live, _Queued):
            i = self._queues[live.slot].index(live)
            return "queued", self._current_finish(live.slot, i, now) - now - live.mean
        if isinstance(live, _Running):
            return "running", max(0.0, live.start + live.row.high - now)
        return self.status(key), None

    def _keep(self, key: AttemptKey, outcome: Outcome) -> None:
        """Keep (attempt key, outcome) for T_keep (8.6, 5.5)."""
        now = self._clock.now
        self._kept[key] = (outcome, now)

        def forget() -> None:
            if self._kept.get(key) == (outcome, now):
                del self._kept[key]

        self._clock.schedule(now + self._t_keep, forget)

    # --- helpers --------------------------------------------------------------------------------------------

    def _row(self, task_type: str, q: SlotType) -> RunRow:
        row = self._table.row(task_type, self._hw.name, q)
        if row is None:
            raise KeyError(f"{task_type} has no time-table row on {self._hw.name} {q}")
        return row

    def _log(self, event: str, key: AttemptKey, detail: str) -> None:
        if self._trace is not None:
            self._trace((event, self._clock.now, key, detail))
