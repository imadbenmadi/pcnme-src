"""One fog zone end to end in fast mode: upload, intake, TOF, placement, execution, delivery, ack, close
(final-design 6A-6E, 7.1, 8, 10.3; PLAN M2.6).

Reuses the M1 pieces: the backhaul is a FIFO pipe, a cloud transfer is a CloudPath over the shared WAN pipe,
the zone's node and the cloud run the same Executor. The radio uplink shares its airtime among the uploads in
progress (net/radio.py, CHANGES C6).

Messages go over the bus (bus/bus.py, M3.1): the executor publishes each result or status once on
car/{id}/result; the serving zone pushes results (never status messages) to the car, and the anchor keeps a
copy; the car's confirm reaches the serving zone, which publishes car/{id}/ack for the anchor (6D, 10.3).

The state plane (zone/statestore.py, M3.3): every zone and the cloud publish zone/{id}/state each state
period, and every POST reply carries a snapshot; the anchor keeps the newest by version, counts its own
pending sends (R1) and estimates ŵ_WAN from its cloud uploads (4.3). Placement (M3.5): here, a neighbour
(POST over the own backhaul pipe, reply after the backhaul delay) or the cloud; a refusal gets the one
re-decision (9.1).

M3 PLACEHOLDERS, each the simplest correct thing behind a small interface:
- No message is lost, so there is no watchdog, retry, cancel or republish (M3.6, M3.7).
- Cars do not move: the anchor is always the serving zone (M3.8).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

from pcnme.bus import topics
from pcnme.bus.bus import Bus, Client, Delivery
from pcnme.bus.calls import call
from pcnme.car.car import Car, Confirm, Result, Task
from pcnme.config import Config, DeadlineMode, SlotType
from pcnme.energy import EnergyModel
from pcnme.ids import AttemptKey, SnapshotCounter, TaskId
from pcnme.net import hops
from pcnme.net.pipe import FifoPipe
from pcnme.net.radio import RadioTransfer, RadioUplink, shannon_rate
from pcnme.net.series import CloudPath, SeriesTransfer
from pcnme.node.executor import Accepted, Executor, Refused, Reply, Report, TaskRequest
from pcnme.node.state import SlotState
from pcnme.node.timetable import RunTimeTable
from pcnme.sim.clock import EventClock
from pcnme.sim.rng import RandomStreams
from pcnme.zone.decision import DecisionContext, NeighbourView, Option, OptionView, Placement
from pcnme.zone.estimates import PathEstimates
from pcnme.zone.intake import AnchorRow, AnchorTable, anchor_deadline
from pcnme.zone.layout import CLOUD, NeighbourTable, Site
from pcnme.zone.masks import views
from pcnme.zone.statestore import Snapshot, StateStore
from pcnme.zone.tof import TofBroker


@dataclass(frozen=True, slots=True)
class EarlierAttempt:
    """An attempt that ended before the last one: refused (9.1, 9.2)."""

    attempt: int
    option: Option
    executor: str
    reply: str
    t_send: float


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
    executor: str | None = None  # the zone id or "cloud" of the last attempt
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
    earlier: list[EarlierAttempt] = field(default_factory=list)
    redecided: bool = False  # the one re-decision after a refusal is used (9.1)


@dataclass(frozen=True, slots=True)
class ResultMessage:
    """car/{id}/result (5.2): kind = result, or status (skipped or aborted, 8.6). The report carries the slot
    type, the run and the energy spent."""

    kind: Literal["result", "status"]
    executor_id: str
    report: Report

    @property
    def key(self) -> AttemptKey:
        return self.report.key


@dataclass(frozen=True, slots=True)
class ZoneStatus:
    """zone/{id}/status (5.2): retained; the last will says offline (7.2, 9.1)."""

    zone_id: str
    status: Literal["online", "offline"]
    x: float
    y: float
    radius: float

    @property
    def site(self) -> Site:
        return Site(self.zone_id, self.x, self.y, self.radius)


@dataclass(frozen=True, slots=True)
class AckMessage:
    """car/{id}/ack (5.2): the copy the car kept and its latency, from the serving zone."""

    confirm: Confirm
    zone_id: str


def publish_report(client: Client, executor_id: str, report: Report) -> None:
    """An executor publishes each result or drop once, at once (6D.12, 8.6), QoS 1."""
    kind: Literal["result", "status"] = "result" if report.outcome == "done" else "status"
    k = report.key
    client.publish(
        topics.car_result(str(k.task_id[0])),  # task_id = (car_id, epoch, seq)
        ResultMessage(kind, executor_id, report),
        qos=1,
        key=(*k.task_id, k.anchor_id, k.attempt, "result"),
    )


def busy_share(slots: Mapping[SlotType, SlotState]) -> float:
    """The "load" of a state message (5.2): busy slots / slots (QUESTIONS Q20)."""
    c = sum(s.c for s in slots.values())
    return sum(s.u for s in slots.values()) / c if c else 0.0


def start_state_loop(clock: EventClock, period: float, publish: Callable[[], None]) -> None:
    """Publish now, then every state period (5.2: every 100 ms)."""

    def tick() -> None:
        publish()
        clock.schedule(clock.now + period, tick)

    clock.schedule(clock.now, tick)


class Cloud:
    """The cloud: one executor shared by every zone, an MQTT client on the WAN (10.3: one WAN crossing to
    publish). It has no radio and never predicts."""

    def __init__(
        self, clock: EventClock, streams: RandomStreams, config: Config, table: RunTimeTable, bus: Bus
    ) -> None:
        offline = ZoneStatus(CLOUD, "offline", 0.0, 0.0, 0.0)  # the cloud has no coverage circle
        self.mqtt = bus.connect(
            CLOUD, config.links.wan_delay, lambda d: None, will=(topics.zone_status(CLOUD), offline)
        )
        self.mqtt.publish(
            topics.zone_status(CLOUD),
            ZoneStatus(CLOUD, "online", 0.0, 0.0, 0.0),
            qos=1,
            key=(CLOUD, "status"),
            retained=True,
        )
        self.executor = Executor.from_config(
            clock, streams, config, "cloud", table, on_report=lambda r: publish_report(self.mqtt, "cloud", r)
        )
        self._versions = SnapshotCounter()
        start_state_loop(clock, config.stateplane.state_period, self._publish_state)

    def snapshot(self) -> Snapshot:
        """The cloud's state: it never publishes ŵ_WAN, it cannot see the router's queue (4.3)."""
        slots = self.executor.snapshot()
        return Snapshot(CLOUD, self._versions.next(), slots, 0.0, None, busy_share(slots))

    def _publish_state(self) -> None:
        s = self.snapshot()
        self.mqtt.publish(
            topics.zone_state(CLOUD), s, qos=0, key=(CLOUD, "state", s.version.counter), retained=True
        )

    def handle(self, req: TaskRequest) -> tuple[Reply, Snapshot]:
        """POST /tasks at the cloud: the executor's reply and a snapshot (R1)."""
        return self.executor.submit(req), self.snapshot()


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
        placement: Placement,
        mode: DeadlineMode,
        hw_name: str = "F-GPU",
        *,
        bus: Bus,
        start_sites: tuple[Site, ...] = (),
        fog_classes: Mapping[str, str] | None = None,
        on_close: Callable[[TaskRecord], None] | None = None,
    ) -> None:
        """`start_sites`: the other zones active at start-up (deployment configuration, 7.5.2); none for a
        zone alone."""
        self.zone_id = zone_id
        self._bus = bus
        self.peers: dict[str, Zone] = {}  # the other zones, for direct calls (5.3); set by the city
        self._fog = dict(fog_classes or {})  # every zone's hardware class: deployment configuration (Q21)
        self._clock = clock
        self._streams = streams
        self._cfg = config
        self._mode: DeadlineMode = mode  # the scenario's deadlines (16.2), for the TOF-Broker
        self._hw_name = hw_name
        self.uplink = RadioUplink(clock)
        self.backhaul = FifoPipe(config.links.backhaul_bw)
        self._cloud_path = CloudPath(clock, self.backhaul, wan)
        self.cloud = cloud
        x, y = config.layout.zone_positions_m[zone_id]
        self.site = Site(zone_id, x, y, config.layout.zone_radius_m)
        self.neighbours = NeighbourTable.at_start(self.site, list(start_sites), config.layout.d_nb_m)  # 7.5
        offline = ZoneStatus(zone_id, "offline", x, y, self.site.radius)
        self.mqtt = bus.connect(
            zone_id,
            config.links.backhaul_delay,
            self._on_message,
            will=(topics.zone_status(zone_id), offline),
        )
        for other in (*config.layout.zones, CLOUD):  # every zone and the cloud announce their status (5.2)
            if other != zone_id:
                self.mqtt.subscribe(topics.zone_status(other))
        self.mqtt.publish(
            topics.zone_status(zone_id),
            ZoneStatus(zone_id, "online", x, y, self.site.radius),
            qos=1,
            key=(zone_id, "status"),
            retained=True,
        )
        self.executor = Executor.from_config(
            clock, streams, config, hw_name, table, on_report=lambda r: publish_report(self.mqtt, zone_id, r)
        )
        lk, sp = config.links, config.stateplane
        bh_d, wan_d = sum(lk.backhaul_delay) / 2, sum(lk.wan_delay) / 2  # mean delays
        pub = {z: bh_d + lk.broker_extra_delay + bh_d for z in config.layout.zones}  # δ̂_pub (Q20)
        pub[CLOUD] = wan_d + lk.broker_extra_delay + bh_d
        self.store = StateStore(zone_id, clock, sp.stale_after, pub, sp.wan_ewma_lambda)
        self._est = PathEstimates(config, self.store)  # ŵ_WAN from the state store (4.3)
        self._reply_cloud = (
            wan_d + bh_d
        )  # δ̂_reply,cloud: the HTTP reply crosses the WAN, then the backhaul (Q20)
        self._tof = TofBroker(config, self._est)
        self._em = EnergyModel(config)
        self._versions = SnapshotCounter()
        self._placement = placement
        self.anchor = AnchorTable(zone_id)
        self.records: dict[object, TaskRecord] = {}
        self._cars: dict[str, tuple[Car, float]] = {}
        self._on_close = on_close
        self._mean_radio = sum(config.radio.uplink_delay) / 2
        self.mqtt.subscribe(topics.zone_state(CLOUD))
        for z in self.neighbours.positions:
            if z is not None:
                self.mqtt.subscribe(topics.zone_state(z))
        start_state_loop(clock, sp.state_period, self._publish_state)

    # --- state plane (5.2, 9.2) -----------------------------------------------------------------------------

    def snapshot(self) -> Snapshot:
        slots = self.executor.snapshot()
        s = self.store
        return Snapshot(
            self.zone_id, self._versions.next(), slots, s.wan.estimate, s.own_wan_age(), busy_share(slots)
        )

    def _publish_state(self) -> None:
        s = self.snapshot()
        topic = topics.zone_state(self.zone_id)
        self.mqtt.publish(topic, s, qos=0, key=(self.zone_id, "state", s.version.counter), retained=True)

    # --- cars and uploads (6A) ------------------------------------------------------------------------------

    def add_car(self, car: Car, distance: float) -> None:
        """The car is served here and anchors its tasks here (no mobility yet: M3.8). Serving role: result
        and attach; anchor role: attach, gps, result and ack (9.6), counted per topic."""
        self._cars[car.car_id] = (car, shannon_rate(distance, self._cfg.radio))
        c = car.car_id
        for topic in (topics.car_result(c), topics.car_attach(c)):  # serving
            self.mqtt.subscribe(topic)
        for topic in (topics.car_attach(c), topics.car_gps(c), topics.car_result(c), topics.car_ack(c)):
            self.mqtt.subscribe(topic)  # anchor

    def upload(self, task: Task) -> None:
        """The car uploads its task over the zone's radio, sharing its airtime with the other uploads in
        progress (6A.2, CHANGES C6); the task exists at t_rx."""
        car, rate = self._cars[task.car_id]
        now = self._clock.now
        b_car = car.budget(task, now)
        d = hops.delay(
            self._streams, hops.RADIO_UP, self._cfg.radio.uplink_delay, task.task_id, hops.UPLOAD_ATTEMPT
        )
        rec = TaskRecord(task=task, anchor=self.zone_id, rate=rate, upload_start=now, radio_delay=d)
        self.records[task.task_id] = rec

        def arrived(tr: RadioTransfer) -> None:
            rec.w_radio = tr.wait  # known exactly at t_rx: the zone runs its uplink (6A.3)
            rec.parts["upload"] = tr.duration
            self._intake(rec, b_car)

        self.uplink.send(size_bits=task.input_bits, rate=rate, delay=d, on_delivered=arrived)

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
            self._send(rec, row, "cloud", None)  # a boulder: no decision (7.1.3)
            return
        ctx = self._context(task, now, rec.d_a)
        option = self._placement.choose(ctx)
        if option is None:
            self._close(rec, "drop_masked")  # 7.3: every option masked
            return
        self._send(rec, row, option, next(v for v in ctx.views if v.option == option))

    def _neighbour(self, zone_id: str | None) -> NeighbourView:
        """One neighbour position as the anchor knows it (7.2): its newest snapshot, its own pending sends
        there (R1), offline (its last will) and stale."""
        if zone_id is None:
            return NeighbourView(None)
        snap = self.store.snapshot(zone_id)
        return NeighbourView(
            zone_id=zone_id,
            hw=self._fog.get(zone_id, ""),
            slots=None if snap is None else snap.slots,
            pending={q: self.store.pending_on(zone_id, q) for q in ("gpu", "cpu")},
            offline=self.neighbours.online.get(zone_id) is False,
            stale=self.store.is_stale(zone_id),
        )

    def _context(self, task: Task, now: float, d_a: float) -> DecisionContext:
        """What the anchor knows (6B.7): its own exact state; the cloud's newest snapshot and its own pending
        sends there (R1); the masks' views of the 6 options."""
        cloud = self.store.snapshot(CLOUD)
        pending = {q: self.store.pending_on(CLOUD, q) for q in ("gpu", "cpu")}
        ctx = DecisionContext(
            task=task,
            now=now,
            d_a=d_a,
            here=self.executor.snapshot(),
            cloud=None if cloud is None else cloud.slots,
            backhaul_wait=self.backhaul.wait_at(now),
            estimates=self._est,
            here_hw=self._hw_name,
            cloud_pending=pending,  # type: ignore[arg-type]
            cloud_stale=self.store.is_stale(CLOUD),
            cloud_offline=self.neighbours.online.get(CLOUD) is False,
            neighbours=tuple(self._neighbour(z) for z in self.neighbours.positions),
        )
        return replace(ctx, views=tuple(views(self._cfg, self._em, ctx)))

    # --- placement and execution (6C) -----------------------------------------------------------------------

    def _send(self, rec: TaskRecord, row: AnchorRow, option: Option, view: OptionView | None) -> None:
        """One attempt (6C.9, 9.1): here, a neighbour (POST over the own backhaul pipe) or the cloud (backhaul
        then WAN). Every placement has its own attempt number (5.5); an earlier one was refused."""
        now = self._clock.now
        task = rec.task
        if rec.option is not None and rec.executor is not None and rec.t_send is not None:
            prev = EarlierAttempt(row.attempts, rec.option, rec.executor, rec.reply or "", rec.t_send)
            rec.earlier.append(prev)
        attempt = self.anchor.next_attempt(task.task_id)
        key = AttemptKey(task.task_id, self.zone_id, attempt)
        rec.option, rec.t_send = option, now
        rec.reply = rec.slot = rec.transfer = rec.t_at_executor = rec.budget = None
        rec.parts.pop("input", None)
        rec.parts["before_send"] = now - rec.t_rx  # the decision and any refusal before this send
        pend = (task.task_id, attempt)
        w_bh = self.backhaul.wait_at(now)
        if option == "here":
            rec.executor = self.zone_id
            budget = rec.d_a - self._est.t_ret_fog() - now  # 6C.9: d_k = d_A - T̂_ret,here
            req = TaskRequest(key, task.task_type, task.cls, task.l_max, budget)
            rec.t_at_executor, rec.budget = now, budget
            self._reply(rec, self.zone_id, pend, self.executor.submit(req), None, w_bh)
            return
        if option == "cloud":
            self._send_cloud(rec, key, pend, view, w_bh)
            return
        target = self.neighbours.positions[int(option[1]) - 1]
        assert target is not None and view is not None and view.slot is not None
        rec.executor = target
        t_in = self._est.t_in_neighbour(task.input_bits, w_bh)
        budget = rec.d_a - now - t_in - self._est.t_ret_fog()  # 6C.9: B_k = d_A - t_s - T̂_in - T̂_ret
        req = TaskRequest(key, task.task_type, task.cls, task.l_max, budget)
        self.store.add_pending(target, pend, view.slot, task.cls, view.run)  # R1 (9.2)
        d_bh = hops.delay(self._streams, hops.BH_OUT, self._cfg.links.backhaul_delay, task.task_id, attempt)
        tr = self.backhaul.send(now=now, size_bits=task.input_bits, delay=d_bh)  # POST /tasks + the input
        back = hops.delay(self._streams, hops.REPLY_BH, self._cfg.links.backhaul_delay, task.task_id, attempt)
        peer = self.peers[target]

        def at_peer() -> tuple[Reply, Snapshot]:
            rec.parts["input"] = tr.duration
            rec.t_at_executor, rec.budget = self._clock.now, budget
            return peer.handle(req)

        call(
            self._clock,
            tr.delivery,
            lambda: self._bus.is_alive(target),
            at_peer,
            lambda r: back,
            lambda r: self._reply(rec, target, pend, r[0], r[1], w_bh),
            caller_alive=lambda: self.mqtt.alive,
        )

    def _send_cloud(
        self, rec: TaskRecord, key: AttemptKey, pend: tuple[object, ...], view: OptionView | None, w_bh: float
    ) -> None:
        task, now, attempt = rec.task, self._clock.now, key.attempt
        rec.executor = CLOUD
        t_in = self._est.t_in_cloud(task.input_bits, w_bh)
        budget = rec.d_a - now - t_in - self._est.t_ret_cloud()  # 6C.9: B_k = d_A - t_s - T̂_in - T̂_ret
        d_bh = hops.delay(self._streams, hops.BH_OUT, self._cfg.links.backhaul_delay, task.task_id, attempt)
        d_wan = hops.delay(self._streams, hops.WAN_IN, self._cfg.links.wan_delay, task.task_id, attempt)
        req = TaskRequest(key, task.task_type, task.cls, task.l_max, budget)
        slot = (
            view.slot if view is not None and view.slot is not None else self._cloud_fastest(task.task_type)
        )
        row = self._cfg.time_table.row(task.task_type, "cloud", slot)
        assert row is not None
        self.store.add_pending(CLOUD, pend, slot, task.cls, row.mean)  # R1 (9.2)

        def arrived(tr: SeriesTransfer) -> None:
            rec.transfer = tr
            rec.parts["input"] = tr.duration
            rec.t_at_executor, rec.budget = self._clock.now, req.budget
            reply, snap = self.cloud.handle(req)
            back = hops.delay(self._streams, hops.REPLY_WAN, self._cfg.links.wan_delay, task.task_id, attempt)
            back += hops.delay(
                self._streams, hops.REPLY_BH, self._cfg.links.backhaul_delay, task.task_id, attempt
            )
            self._clock.schedule(
                self._clock.now + back, lambda: self._reply(rec, CLOUD, pend, reply, snap, w_bh)
            )

        self._cloud_path.send(task.input_bits, d_bh, d_wan, arrived)

    def _cloud_fastest(self, task_type: str) -> SlotType:
        rows = self._cfg.time_table
        return min(
            (q for q in ("gpu", "cpu") if rows.row(task_type, "cloud", q) is not None),
            key=lambda q: rows.row(task_type, "cloud", q).mean,  # type: ignore[union-attr]
        )

    def handle(self, req: TaskRequest) -> tuple[Reply, Snapshot]:
        """POST /tasks from another anchor (5.3): the executor's answer and a snapshot (R1). A zone that
        accepted a task never forwards it (N12)."""
        return self.executor.submit(req), self.snapshot()

    def _reply(
        self,
        rec: TaskRecord,
        executor_id: str,
        pend: tuple[object, ...],
        reply: Reply,
        snap: Snapshot | None,
        w_bh: float,
    ) -> None:
        """An executor's answer reaches the anchor (6C.10): its snapshot (R1), the pending send cleared, a
        ŵ_WAN sample on a cloud accept (4.3); a refusal leads to the one re-decision (9.1, 9.2)."""
        if not self.anchor.is_open(rec.task.task_id):
            return
        if snap is not None:
            self.store.on_snapshot(snap)
        self.store.clear_pending(executor_id, pend)
        if isinstance(reply, Accepted):
            if executor_id == CLOUD:
                lk = self._cfg.links
                assert rec.t_send is not None
                sample = (self._clock.now - rec.t_send) - (w_bh + sum(lk.backhaul_delay) / 2)
                sample -= rec.task.input_bits / self._cfg.derived.b_wan + sum(lk.wan_delay) / 2
                sample -= self._reply_cloud
                self.store.wan_sample(sample)
            rec.reply, rec.slot = "accepted", reply.slot
        elif isinstance(reply, Refused):
            rec.reply = reply.reason
            self._refused(rec, reply.reason)

    def _refused(self, rec: TaskRecord, reason: str) -> None:
        """9.1: refused at once; the anchor re-decides once; refused again or no slack -> hard: drop; soft:
        cloud, unless the cloud cannot finish by deadline + L_max either, then drop. A boulder had no
        decision to redo (QUESTIONS Q22)."""
        task, row = rec.task, self.anchor.row(rec.task.task_id)
        if rec.route == "agent" and not rec.redecided:
            rec.redecided = True
            ctx = self._context(task, self._clock.now, rec.d_a)
            option = self._placement.choose(ctx)
            if option is not None:
                self._send(rec, row, option, next(v for v in ctx.views if v.option == option))
                return
        if (
            task.cls == "soft"
            and rec.route == "agent"
            and rec.executor != CLOUD
            and self._cloud_can_finish(rec)
        ):
            self._send(rec, row, "cloud", None)  # 9.1: the soft fallback, a rule, not a decision
            return
        self._close(rec, f"drop_refused_{reason}")

    def _cloud_can_finish(self, rec: TaskRecord) -> bool:
        """9.1: T̂_in,cloud + Ê_cloud + T̂_ret,cloud <= time left + L_max (the cloud never queues, as 7.1.3)."""
        task, now = rec.task, self._clock.now
        t = self._est.t_in_cloud(task.input_bits, self.backhaul.wait_at(now)) + self._est.e_cloud(
            task.task_type
        )
        return t + self._est.t_ret_cloud() <= rec.d_a - now + task.l_max

    # --- delivery, ack, close (6D, 6E, 8.6, 9.5, 10.3) ------------------------------------------------------

    def _on_message(self, d: Delivery) -> None:
        scope, car_id, kind = topics.parse(d.topic)
        if scope == "zone" and kind == "status":
            st = d.payload
            assert isinstance(st, ZoneStatus)
            before = set(self.neighbours.positions)
            self.neighbours.on_status(st.site, st.status == "online")  # 7.5.2-3
            if st.zone_id in set(self.neighbours.positions) - before:
                self.mqtt.subscribe(topics.zone_state(st.zone_id))  # a new neighbour's state
            return
        if scope == "zone" and kind == "state":
            snap = d.payload
            assert isinstance(snap, Snapshot)
            self.store.on_snapshot(snap)
            return
        if scope != "car":
            return
        if kind == "result":
            msg = d.payload
            assert isinstance(msg, ResultMessage)
            if car_id in self._cars and msg.kind == "result":
                self._push(car_id, msg)  # serving role: results only, never status messages (8.6)
            if msg.key.anchor_id == self.zone_id:
                self._anchor_result(msg)  # anchor role: keep the copy, or close on a drop
        elif kind == "ack":
            ack = d.payload
            assert isinstance(ack, AckMessage)
            self._anchor_ack(ack)

    def _anchor_result(self, msg: ResultMessage) -> None:
        rec = self.records.get(msg.key.task_id)
        if rec is None or not self.anchor.is_open(rec.task.task_id):
            return
        report = msg.report
        rec.report = report
        if msg.kind == "status":  # skipped or aborted, reported at once (8.6)
            self._close(rec, f"drop_{report.outcome}")
            return
        assert report.start is not None and rec.t_at_executor is not None
        rec.parts["queue"] = report.start - rec.t_at_executor
        rec.parts["run"] = report.busy

    def _push(self, car_id: str, msg: ResultMessage) -> None:
        """The serving zone pushes a result to the car over the radio downlink (10.3)."""
        k = msg.key
        down = hops.delay(
            self._streams, hops.RADIO_DOWN, self._cfg.radio.downlink_delay, k.task_id, k.attempt
        )
        self._clock.schedule(self._clock.now + down, lambda: self._at_car(car_id, msg))

    def _at_car(self, car_id: str, msg: ResultMessage) -> None:
        car, _ = self._cars[car_id]
        now = self._clock.now
        k = msg.key
        rec = self.records.get(k.task_id)  # the anchor's record, while the car's zone is its anchor (M3.8)
        if rec is not None:
            rec.deliveries += 1
            if rec.result_at_car is None:
                rec.result_at_car = now
                rec.parts["return"] = now - msg.report.finish
        confirm = car.receive(Result(TaskId(*k.task_id), k.anchor_id, k.attempt), now)  # type: ignore[arg-type]
        if confirm is None:
            return
        # the car's confirm crosses the radio to this serving zone, which publishes the ack (6D.13)
        up = hops.delay(self._streams, hops.ACK_UP, self._cfg.radio.uplink_delay, k.task_id, k.attempt)
        ack = AckMessage(confirm, self.zone_id)
        key = (*k.task_id, k.anchor_id, k.attempt, "ack")
        self._clock.schedule(now + up, lambda: self.mqtt.publish(topics.car_ack(car_id), ack, qos=1, key=key))

    def _anchor_ack(self, ack: AckMessage) -> None:
        c = ack.confirm
        if c.anchor_id != self.zone_id:
            return
        rec = self.records.get(c.task_id)
        if rec is None or not self.anchor.is_open(c.task_id):
            return  # 9.5.4: repeat acks for a closed task are ignored
        rec.confirm = c
        self._close(rec, self._outcome(rec.task, c.latency))

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
