"""One fog zone end to end in fast mode: upload, intake, TOF, placement, execution, delivery, ack, close
(final-design 6A-6E, 7.1, 8, 10.3; PLAN M2.6).

Reuses the M1 pieces: the radio uplink and backhaul are FIFO pipes, a cloud transfer is a CloudPath over the
shared WAN pipe, the zone's node and the cloud run the same Executor.

M3 PLACEHOLDERS, each the simplest correct thing behind a small interface:
- Messages are direct clock callbacks with the modelled delays (broker processing on top of the backhaul or
  WAN delay), not the bus (M3.1).
- The anchor reads the cloud's state directly, with no age (M3.3); ŵ_WAN comes from estimates.ZeroWanWait.
- A refused task is dropped and logged; there is no re-decision (M3.5).
- No message is lost, so there is no watchdog, retry, cancel or republish (M3.6, M3.7).
- Cars do not move: the anchor is always the serving zone (M3.8).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pcnme.car.car import Car, Confirm, Result, Task
from pcnme.config import Config, DeadlineMode, SlotType
from pcnme.ids import AttemptKey
from pcnme.net import hops
from pcnme.net.pipe import FifoPipe
from pcnme.net.radio import RadioUplink, shannon_rate
from pcnme.net.series import CloudPath, SeriesTransfer
from pcnme.node.executor import Accepted, Executor, Refused, Report, TaskRequest
from pcnme.node.state import SlotState
from pcnme.node.timetable import RunTimeTable
from pcnme.sim.clock import EventClock
from pcnme.sim.rng import RandomStreams
from pcnme.zone.estimates import PathEstimates
from pcnme.zone.intake import AnchorRow, AnchorTable, anchor_deadline
from pcnme.zone.tof import TofBroker

Option = Literal["here", "cloud"]


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """What the anchor knows when it places a pebble (6B): its own exact state and the cloud's."""

    task: Task
    now: float
    d_a: float
    here: dict[SlotType, SlotState]
    cloud: dict[SlotType, SlotState]
    backhaul_wait: float  # the zone's own backhaul pipe, known exactly (4.3)
    estimates: PathEstimates

    @property
    def time_left(self) -> float:
        return self.d_a - self.now


class Placement(Protocol):
    """A placement method: an option, or None when every option is masked (7.3)."""

    def choose(self, ctx: DecisionContext) -> Option | None: ...


@dataclass(slots=True)
class TaskRecord:
    """Everything that happened to one task, for energy (12.5), metrics and invariants."""

    task: Task
    anchor: str
    rate: float = 0.0  # R_car, bit/s
    upload_start: float = 0.0
    w_radio: float = 0.0
    radio_delay: float = 0.0
    t_rx: float = 0.0
    d_a: float = 0.0
    route: str = ""  # "agent", "cloud" (boulder) or "drop" (boulder the cloud cannot make)
    option: Option | None = None
    t_send: float | None = None
    t_at_executor: float | None = None  # when the input reached the executor
    budget: float | None = None  # B_k sent to the executor (6C.9)
    transfer: SeriesTransfer | None = None
    reply: str | None = None  # "accepted", "full", "late"
    slot: SlotType | None = None
    report: Report | None = None
    result_at_car: float | None = None
    confirm: Confirm | None = None
    outcome: str | None = None
    closed_at: float | None = None
    deliveries: int = 0  # result copies pushed to the car (12.2: each pays the return energy)
    parts: dict[str, float] = field(default_factory=dict)  # durations along the way, for the latency check


class Cloud:
    """The cloud: one executor shared by every zone; its reports go back to each attempt's anchor."""

    def __init__(
        self, clock: EventClock, streams: RandomStreams, config: Config, table: RunTimeTable
    ) -> None:
        self._zones: dict[str, Zone] = {}
        self.executor = Executor.from_config(clock, streams, config, "cloud", table, on_report=self._route)

    def attach(self, zone: Zone) -> None:
        self._zones[zone.zone_id] = zone

    def _route(self, report: Report) -> None:
        self._zones[report.key.anchor_id].on_report(report, executor="cloud")


class Zone:
    """One fog zone: access point, fog node, TOF-Broker, placement, anchor table (3)."""

    def __init__(
        self,
        zone_id: str,
        clock: EventClock,
        streams: RandomStreams,
        config: Config,
        table: RunTimeTable,
        wan: FifoPipe,
        cloud: Cloud,
        estimates: PathEstimates,
        placement: Placement,
        mode: DeadlineMode,
        hw_name: str = "F-GPU",
        on_close: Callable[[TaskRecord], None] | None = None,
    ) -> None:
        self.zone_id = zone_id
        self._clock = clock
        self._streams = streams
        self._cfg = config
        self._mode: DeadlineMode = mode  # the scenario's deadlines (16.2), for the TOF-Broker
        self.uplink = RadioUplink()
        self.backhaul = FifoPipe(config.links.backhaul_bw)
        self._cloud_path = CloudPath(clock, self.backhaul, wan)
        self.cloud = cloud
        self.executor = Executor.from_config(
            clock, streams, config, hw_name, table, on_report=lambda r: self.on_report(r, executor="here")
        )
        self._est = estimates
        self._tof = TofBroker(config, estimates)
        self._placement = placement
        self.anchor = AnchorTable(zone_id)
        self.records: dict[object, TaskRecord] = {}
        self._cars: dict[str, tuple[Car, float]] = {}
        self._on_close = on_close
        cloud.attach(self)
        self._mean_radio = sum(config.radio.uplink_delay) / 2
        self._broker = config.links.broker_extra_delay

    # --- cars and uploads (6A) ------------------------------------------------------------------------------

    def add_car(self, car: Car, distance: float) -> None:
        self._cars[car.car_id] = (car, shannon_rate(distance, self._cfg.radio))

    def upload(self, task: Task) -> None:
        """The car uploads its task over the zone's radio pipe (6A.2); the task exists at t_rx."""
        car, rate = self._cars[task.car_id]
        now = self._clock.now
        b_car = car.budget(task, now)
        d = hops.delay(
            self._streams, hops.RADIO_UP, self._cfg.radio.uplink_delay, task.task_id, hops.UPLOAD_ATTEMPT
        )
        tr = self.uplink.send(now=now, size_bits=task.input_bits, rate=rate, delay=d)
        rec = TaskRecord(
            task=task, anchor=self.zone_id, rate=rate, upload_start=now, w_radio=tr.wait, radio_delay=d
        )
        rec.parts["upload"] = tr.duration
        self.records[task.task_id] = rec
        self._clock.schedule(tr.delivery, lambda: self._intake(rec, b_car))

    def _intake(self, rec: TaskRecord, b_car: float) -> None:
        now = self._clock.now
        task = rec.task
        rec.t_rx = now
        rec.d_a = anchor_deadline(
            now, b_car, rec.w_radio, task.input_bits, rec.rate, self._mean_radio
        )  # 6A.3
        row = self.anchor.open(task, t_rx=now, d_a=rec.d_a)
        route = self._tof.classify(
            task.task_type,
            self._mode,
            task.cls,
            task.input_bits,
            rec.d_a - now,
            task.l_max,
            self.backhaul.wait_at(now),
        )
        rec.route = route
        if route == "drop":
            self._close(rec, "drop_intake")  # 7.1.3: no decision, no transition
            return
        if route == "cloud":
            self._send(rec, row, "cloud")
            return
        ctx = DecisionContext(
            task=task,
            now=now,
            d_a=rec.d_a,
            here=dict(self.executor.snapshot()),
            cloud=dict(self.cloud.executor.snapshot()),
            backhaul_wait=self.backhaul.wait_at(now),
            estimates=self._est,
        )
        option = self._placement.choose(ctx)
        if option is None:
            self._close(rec, "drop_masked")  # 7.3: every option masked
            return
        self._send(rec, row, option)

    # --- placement and execution (6C) -----------------------------------------------------------------------

    def _send(self, rec: TaskRecord, row: AnchorRow, option: Option) -> None:
        now = self._clock.now
        task = rec.task
        attempt = self.anchor.next_attempt(task.task_id)
        key = AttemptKey(task.task_id, self.zone_id, attempt)
        rec.option, rec.t_send = option, now
        if option == "here":
            budget = rec.d_a - self._est.t_ret_fog() - now  # 6C.9: d_k = d_A - T̂_ret,here
            self._submit(rec, self.executor, TaskRequest(key, task.task_type, task.cls, task.l_max, budget))
            return
        t_in = self._est.t_in_cloud(task.input_bits, self.backhaul.wait_at(now))
        budget = rec.d_a - now - t_in - self._est.t_ret_cloud()  # 6C.9: B_k = d_A - t_s - T̂_in - T̂_ret
        d_bh = hops.delay(self._streams, hops.BH_OUT, self._cfg.links.backhaul_delay, task.task_id, attempt)
        d_wan = hops.delay(self._streams, hops.WAN_IN, self._cfg.links.wan_delay, task.task_id, attempt)
        req = TaskRequest(key, task.task_type, task.cls, task.l_max, budget)

        def arrived(tr: SeriesTransfer) -> None:
            rec.transfer = tr
            rec.parts["input"] = tr.duration
            self._submit(rec, self.cloud.executor, req)

        self._cloud_path.send(task.input_bits, d_bh, d_wan, arrived)

    def _submit(self, rec: TaskRecord, executor: Executor, req: TaskRequest) -> None:
        rec.t_at_executor = self._clock.now
        rec.budget = req.budget
        reply = executor.submit(req)
        if isinstance(reply, Accepted):
            rec.reply, rec.slot = "accepted", reply.slot
        elif isinstance(reply, Refused):
            rec.reply = reply.reason
            self._close(rec, f"drop_refused_{reply.reason}")  # M3 PLACEHOLDER (M3.5): no re-decision

    # --- delivery, ack, close (6D, 6E, 8.6, 9.5, 10.3) ------------------------------------------------------

    def on_report(self, report: Report, executor: str) -> None:
        rec = self.records[report.key.task_id]
        rec.report = report
        attempt = report.key.attempt
        tid = rec.task.task_id
        lk = self._cfg.links
        # The executor publishes once: one backhaul (fog) or WAN (cloud) crossing to the broker, broker
        # processing, then one backhaul crossing to the serving zone (10.3, 4.2: results pay only delays).
        pub_hop, pub_iv = (
            (hops.PUB_WAN, lk.wan_delay) if executor == "cloud" else (hops.PUB_BH, lk.backhaul_delay)
        )
        to_zone = (
            hops.delay(self._streams, pub_hop, pub_iv, tid, attempt)
            + self._broker
            + hops.delay(self._streams, hops.TO_ZONE, lk.backhaul_delay, tid, attempt)
        )
        if (
            report.outcome != "done"
        ):  # skipped or aborted: a status message to the anchor, sent at once (8.6),
            # not when an abort tail frees the slot (report.finish)
            self._clock.schedule(
                self._clock.now + to_zone, lambda: self._close(rec, f"drop_{report.outcome}")
            )
            return
        down = hops.delay(self._streams, hops.RADIO_DOWN, self._cfg.radio.downlink_delay, tid, attempt)
        assert report.start is not None and rec.t_at_executor is not None
        rec.parts["queue"] = report.start - rec.t_at_executor
        rec.parts["run"] = report.busy
        rec.parts["return"] = to_zone + down
        at_car = report.finish + to_zone + down
        self._clock.schedule(at_car, lambda: self._at_car(rec, report.key))

    def _at_car(self, rec: TaskRecord, key: AttemptKey) -> None:
        car, _ = self._cars[rec.task.car_id]
        now = self._clock.now
        rec.deliveries += 1
        rec.result_at_car = now
        confirm = car.receive(Result(rec.task.task_id, key.anchor_id, key.attempt), now)
        if confirm is None:
            return
        rec.confirm = confirm
        tid, a = rec.task.task_id, key.attempt
        ack = (
            hops.delay(self._streams, hops.ACK_UP, self._cfg.radio.uplink_delay, tid, a)
            + hops.delay(self._streams, hops.ACK_PUB, self._cfg.links.backhaul_delay, tid, a)
            + self._broker
            + hops.delay(self._streams, hops.ACK_TO_ANCHOR, self._cfg.links.backhaul_delay, tid, a)
        )
        self._clock.schedule(now + ack, lambda: self._close(rec, self._outcome(rec.task, confirm.latency)))

    @staticmethod
    def _outcome(task: Task, latency: float) -> str:
        """QUESTIONS Q15: from the car's latency L. Hard: on time if L <= D, else a drop. Soft: on time if
        L <= D, late if L <= D + L_max, else a drop (9.5.5)."""
        if latency <= task.deadline:
            return "on_time"
        if task.cls == "soft" and latency <= task.deadline + task.l_max:
            return "late"
        return "drop_late"

    def _close(self, rec: TaskRecord, outcome: str) -> None:
        now = self._clock.now
        self.anchor.close(rec.task.task_id, outcome, now)  # raises if closed twice
        rec.outcome, rec.closed_at = outcome, now
        if self._on_close is not None:
            self._on_close(rec)
