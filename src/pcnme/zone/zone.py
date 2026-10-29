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
re-decision (9.1). Loss detection (M3.6, 9.3-9.4): an upload timeout per send, the executor's last will,
a watchdog from the accept, a status probe before any retry, retries with the normal masks, best-effort
cancels; the first result wins; no ack by the end is a drop.

Duplicates and subscriptions (M3.7, 9.5, 9.6): the serving zone pushes a task's result at most once per radio
session; the anchor republishes its copy on every new attach until the ack, and once if no ack came within
T_ack; subscriptions are counted per role (serving: result and attach while the car is attached; anchor:
attach, gps, result and ack while it has an open task of the car).

Mobility (M3.8, 10): the serving zone gets each car's GPS every 100 ms and publishes it on car/{id}/gps
(retained, with an expiry) to the car's anchors; with no GPS for T_leave it ends the serving role. An upload
starts at the car's current rate and can be cut by a handover or a gap (9.7): nothing is left behind.
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
from pcnme.sim.clock import EventClock, EventHandle
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
class AttemptInfo:
    """One attempt as its anchor tracks it (9.3, 9.4): where it went, when, and its live timers."""

    option: Option
    executor: str
    t_send: float
    t_at_executor: float | None = None
    input: float | None = None  # the input's transfer time (0 here)
    live: bool = True  # sent and not yet ended, lost or cancelled
    timer: EventHandle | None = None  # its upload timeout, then its watchdog
    probes: int = 0
    probe_timer: EventHandle | None = None
    done_seen: bool = False  # a probe said done: if no result follows, it was lost (QUESTIONS Q23)


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
    attempts: dict[int, AttemptInfo] = field(default_factory=dict)
    retries: int = 0  # retries of lost attempts used (9.1: 1 hard, 2 soft)
    result_attempt: int | None = None  # the attempt whose result reached the anchor first (9.4.3)
    end_timer: EventHandle | None = None  # no ack by the end: a drop (9.5.5)
    copy: ResultMessage | None = None  # the anchor's copy of the result, until close (6D.12)
    republished: bool = False  # the one republish after T_ack without an ack (9.5.4)


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
class GpsMessage:
    """car/{id}/gps (5.2): position, speed, heading and serving zone; car_time is the car's own clock (only
    differences between its own samples are used, 5.4)."""

    car_id: str
    x: float
    y: float
    vx: float
    vy: float
    zone_id: str
    car_time: float


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
        ledger: dict[tuple[str, object], TaskRecord] | None = None,
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
        self._bh_d, self._wan_d = bh_d, wan_d
        tm, sc = config.timing, config.scheduler
        self._kappa = tm.kappa_up
        self._t_ack = tm.t_ack
        self._m_wd = {"fog": tm.m_wd_fog, "cloud": tm.m_wd_cloud}
        self._t_probe = {"fog": sc.probe_timeout_fog, "cloud": sc.probe_timeout_cloud}
        self._retry_limit = {"hard": sc.retries_hard, "soft": sc.retries_soft}
        self.lose_reply: Callable[[AttemptKey], bool] | None = None  # test hook: an HTTP reply is lost
        self._versions = SnapshotCounter()
        self._placement = placement
        self.anchor = AnchorTable(zone_id)
        self.records: dict[object, TaskRecord] = {}
        # The run's log of every task, keyed by (anchor, task id): a simulation record, not zone state, so a
        # serving zone can note the deliveries of a task anchored elsewhere (shared by the city).
        self.ledger = ledger if ledger is not None else {}
        self._cars: dict[str, tuple[Car, float]] = {}  # the car registry follows the serving role (9.6)
        self._confirmed: dict[str, set[object]] = {}  # per served car: task ids confirmed this radio session
        self._anchored: dict[str, int] = {}  # per car: this zone's open tasks of it (the anchor role)
        self._attaches = 0
        self._uploads: dict[object, int] = {}  # task id -> its radio upload in flight (9.7)
        self._leave: dict[str, EventHandle] = {}  # T_leave timers of the served cars (9.6)
        self.registry: dict[str, GpsMessage] = {}  # the car registry: last GPS of the cars it anchors (3)
        self._gps_seq = 0
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
        """The car attaches here (the serving role, 9.6)."""
        self.attach(car, distance)

    def attach(self, car: Car, distance: float) -> None:
        """A car attaches to this access point: subscribe car/{id}/result and attach (serving role), start a
        new radio session, and announce it on car/{id}/attach to its anchors and its previous serving zone
        (5.2, 9.6)."""
        c = car.car_id
        if c not in self._cars:
            for topic in (topics.car_result(c), topics.car_attach(c)):
                self.mqtt.subscribe(topic)
        self._cars[c] = (car, shannon_rate(distance, self._cfg.radio))
        self._confirmed[c] = set()  # a new radio session (9.5.3)
        self._attaches += 1
        self.mqtt.publish(
            topics.car_attach(c), (c, self.zone_id), qos=1, key=(c, self.zone_id, self._attaches)
        )

    def detach(self, car_id: str) -> None:
        """The serving role ends (another zone announced an attach, or no GPS for T_leave): unsubscribe and
        drop the car from the registry (9.6)."""
        if self._cars.pop(car_id, None) is None:
            return
        self._confirmed.pop(car_id, None)
        timer = self._leave.pop(car_id, None)
        if timer is not None:
            self._clock.cancel(timer)
        for topic in (topics.car_result(car_id), topics.car_attach(car_id)):
            self.mqtt.unsubscribe(topic)

    def _anchor_role(self, car_id: str, step: int) -> None:
        """9.6: the anchor subscribes at its first open task of the car and unsubscribes at its last."""
        n = self._anchored.get(car_id, 0) + step
        self._anchored[car_id] = n
        if (step, n) in ((1, 1), (-1, 0)):
            for topic in (
                topics.car_attach(car_id),
                topics.car_gps(car_id),
                topics.car_result(car_id),
                topics.car_ack(car_id),
            ):
                (self.mqtt.subscribe if step > 0 else self.mqtt.unsubscribe)(topic)
        if n == 0:
            del self._anchored[car_id]
            self.registry.pop(car_id, None)  # 9.6: the anchor drops the car's predictor state

    def upload(self, task: Task, rate: float | None = None) -> None:
        """The car uploads its task over the zone's radio, sharing its airtime with the other uploads in
        progress (6A.2, CHANGES C6), at its current rate (`rate`, from its distance now; by default the rate
        at its attach); the task exists at t_rx."""
        car, attach_rate = self._cars[task.car_id]
        rate = attach_rate if rate is None else rate
        now = self._clock.now
        b_car = car.budget(task, now)
        d = hops.delay(
            self._streams, hops.RADIO_UP, self._cfg.radio.uplink_delay, task.task_id, hops.UPLOAD_ATTEMPT
        )
        rec = TaskRecord(task=task, anchor=self.zone_id, rate=rate, upload_start=now, radio_delay=d)
        self.records[task.task_id] = rec
        self.ledger[(self.zone_id, task.task_id)] = rec

        def arrived(tr: RadioTransfer) -> None:
            self._uploads.pop(task.task_id, None)
            rec.w_radio = tr.wait  # known exactly at t_rx: the zone runs its uplink (6A.3)
            rec.parts["upload"] = tr.duration
            self._intake(rec, b_car)

        self._uploads[task.task_id] = self.uplink.send(
            size_bits=task.input_bits, rate=rate, delay=d, on_delivered=arrived
        )

    def cut_upload(self, task_id: object) -> bool:
        """9.7: the car's connection ended before the upload's last bit: no task exists here, nothing is left
        behind. False if the last bit was already sent (it arrives, and this zone is the anchor)."""
        up = self._uploads.pop(task_id, None)
        if up is None or not self.uplink.cancel(up):
            return False
        del self.records[task_id]
        del self.ledger[(self.zone_id, task_id)]
        return True

    # --- GPS (10.1, 9.6) ------------------------------------------------------------------------------------

    def gps(self, car_id: str, x: float, y: float, vx: float, vy: float, car_time: float) -> None:
        """The serving zone gets the car's GPS (every 100 ms) and publishes it to the car's anchors: QoS 0,
        retained with an MQTT 5 expiry (5.2, 9.6). It restarts the car's T_leave timer."""
        if car_id not in self._cars:
            return
        self._gps_seq += 1
        msg = GpsMessage(car_id, x, y, vx, vy, self.zone_id, car_time)
        tm = self._cfg.timing
        self.mqtt.publish(
            topics.car_gps(car_id),
            msg,
            qos=0,
            key=(car_id, "gps", self.zone_id, self._gps_seq),
            retained=True,
            expiry=tm.gps_expiry,
        )
        old = self._leave.pop(car_id, None)
        if old is not None:
            self._clock.cancel(old)
        self._leave[car_id] = self._clock.schedule(self._clock.now + tm.t_leave, lambda: self._left(car_id))

    def _left(self, car_id: str) -> None:
        """9.6: no GPS for T_leave: the car left (a gap): the serving role ends."""
        self._leave.pop(car_id, None)
        self.detach(car_id)

    def _intake(self, rec: TaskRecord, b_car: float) -> None:
        now = self._clock.now
        task = rec.task
        rec.t_rx = now
        rec.d_a = anchor_deadline(
            now, b_car, rec.w_radio, task.input_bits, rec.rate, self._mean_radio
        )  # 6A.3
        row = self.anchor.open(task, t_rx=now, d_a=rec.d_a)
        self._anchor_role(task.car_id, +1)  # 6A.4: register
        end = rec.d_a + (task.l_max if task.cls == "soft" else 0.0) + self._t_ack  # 9.5.5, QUESTIONS Q23
        rec.end_timer = self._clock.schedule(end, lambda: self._no_ack(rec))
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
            info = rec.attempts[attempt] = AttemptInfo(option, self.zone_id, now, now, 0.0)
            budget = rec.d_a - self._est.t_ret_fog() - now  # 6C.9: d_k = d_A - T̂_ret,here
            req = TaskRequest(key, task.task_type, task.cls, task.l_max, budget)
            rec.t_at_executor, rec.budget = now, budget
            self._reply(rec, self.zone_id, pend, self.executor.submit(req), None, w_bh)
            return
        if option == "cloud":
            rec.attempts[attempt] = AttemptInfo(option, CLOUD, now)
            self._send_cloud(rec, key, pend, view, w_bh)
            return
        target = self.neighbours.positions[int(option[1]) - 1]
        assert target is not None and view is not None and view.slot is not None
        rec.executor = target
        info = rec.attempts[attempt] = AttemptInfo(option, target, now)
        t_in = self._est.t_in_neighbour(task.input_bits, w_bh)
        self._upload_timer(rec, attempt, t_in, self._bh_d)
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
            info.t_at_executor, info.input = self._clock.now, tr.duration
            return peer.handle(req)

        def replied(r: tuple[Reply, Snapshot]) -> None:
            if self.lose_reply is None or not self.lose_reply(key):
                self._reply(rec, target, pend, r[0], r[1], w_bh)

        call(
            self._clock,
            tr.delivery,
            lambda: self._bus.is_alive(target),
            at_peer,
            lambda r: back,
            replied,
            caller_alive=lambda: self.mqtt.alive,
        )

    def _send_cloud(
        self, rec: TaskRecord, key: AttemptKey, pend: tuple[object, ...], view: OptionView | None, w_bh: float
    ) -> None:
        task, now, attempt = rec.task, self._clock.now, key.attempt
        rec.executor = CLOUD
        info = rec.attempts[attempt]
        t_in = self._est.t_in_cloud(task.input_bits, w_bh)
        self._upload_timer(rec, attempt, t_in, self._reply_cloud)
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
            info.t_at_executor, info.input = self._clock.now, tr.duration
            if not self._bus.is_alive(CLOUD):
                return  # a dead cloud never replies
            reply, snap = self.cloud.handle(req)
            back = hops.delay(self._streams, hops.REPLY_WAN, self._cfg.links.wan_delay, task.task_id, attempt)
            back += hops.delay(
                self._streams, hops.REPLY_BH, self._cfg.links.backhaul_delay, task.task_id, attempt
            )
            if self.lose_reply is not None and self.lose_reply(key):
                return
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
        if snap is not None:
            self.store.on_snapshot(snap)
        self.store.clear_pending(executor_id, pend)
        attempt = int(pend[1])  # type: ignore[call-overload]
        info = rec.attempts.get(attempt)
        if not self.anchor.is_open(rec.task.task_id) or info is None or not info.live:
            return  # the task closed, or this attempt timed out and was cancelled
        if info.timer is not None:
            self._clock.cancel(info.timer)
            info.timer = None
        if isinstance(reply, Accepted):
            if executor_id == CLOUD:
                lk = self._cfg.links
                assert rec.t_send is not None
                sample = (self._clock.now - rec.t_send) - (w_bh + sum(lk.backhaul_delay) / 2)
                sample -= rec.task.input_bits / self._cfg.derived.b_wan + sum(lk.wan_delay) / 2
                sample -= self._reply_cloud
                self.store.wan_sample(sample)
            rec.reply, rec.slot = "accepted", reply.slot
            self._arm(
                rec, attempt, self._clock.now, reply.wait + self._row_high(rec, executor_id, reply.slot)
            )
        elif isinstance(reply, Refused):
            rec.reply = reply.reason
            info.live = False
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

    # --- loss detection, retry and cancel (9.1, 9.3, 9.4) ---------------------------------------------------

    def _kind(self, executor_id: str) -> str:
        return "cloud" if executor_id == CLOUD else "fog"

    def _hw(self, executor_id: str) -> str:
        if executor_id == CLOUD:
            return CLOUD
        return self._hw_name if executor_id == self.zone_id else self._fog.get(executor_id, self._hw_name)

    def _executor(self, executor_id: str) -> Executor:
        if executor_id == self.zone_id:
            return self.executor
        return self.cloud.executor if executor_id == CLOUD else self.peers[executor_id].executor

    def _reply_delay(self, executor_id: str) -> float:
        """δ̂_reply,k (Q20): 0 here, the backhaul to a neighbour, WAN + backhaul from the cloud."""
        if executor_id == self.zone_id:
            return 0.0
        return self._reply_cloud if executor_id == CLOUD else self._bh_d

    def _row_high(self, rec: TaskRecord, executor_id: str, slot: SlotType) -> float:
        row = self._cfg.time_table.row(rec.task.task_type, self._hw(executor_id), slot)
        assert row is not None
        return row.p95  # Ê^high (14.4, 18)

    def _cap(self, rec: TaskRecord) -> float:
        """9.3: the watchdog is capped at d_A (hard) or d_A + L_max (soft): after that the task is dropped."""
        return rec.d_a + (rec.task.l_max if rec.task.cls == "soft" else 0.0)

    def _upload_timer(self, rec: TaskRecord, attempt: int, t_in: float, reply: float) -> None:
        """9.3: T_up,k = min(κ_up (T̂_in,k + δ̂_reply,k), d_A - t_s + ℓ), ℓ = 0 hard, L_max soft."""
        now = self._clock.now
        ell = rec.task.l_max if rec.task.cls == "soft" else 0.0
        t_up = min(self._kappa * (t_in + reply), rec.d_a - now + ell)
        info = rec.attempts[attempt]
        info.timer = self._clock.schedule(now + max(0.0, t_up), lambda: self._upload_timeout(rec, attempt))

    def _upload_timeout(self, rec: TaskRecord, attempt: int) -> None:
        """9.1: no reply: treated as not placed. Re-decide (it counts as the one re-decision) and cancel the
        first target, in case only its reply was lost. The pending send ends (R1)."""
        info = rec.attempts[attempt]
        info.timer = None
        if not info.live or not self.anchor.is_open(rec.task.task_id):
            return
        self.store.clear_pending(info.executor, (rec.task.task_id, attempt))
        self._cancel_attempt(rec, attempt)
        rec.reply = "no_reply"
        self._refused(rec, "no_reply")

    def _arm(self, rec: TaskRecord, attempt: int, t_reply: float, rest: float) -> None:
        """9.3: t_wd = t_reply - δ̂_reply + (the time it still needs: Ŵ + Ê^high from an accept) + δ̂_pub
        + m_wd, capped (after the cap the end timer drops the task)."""
        info = rec.attempts[attempt]
        e = info.executor
        pub = self.store.pub_delay.get(e, self.store.pub_delay.get(self.zone_id, 0.0))
        t_wd = t_reply - self._reply_delay(e) + rest + pub + self._m_wd[self._kind(e)]
        if t_wd < self._cap(rec):
            info.timer = self._clock.schedule(t_wd, lambda: self._watchdog(rec, attempt))

    def _watchdog(self, rec: TaskRecord, attempt: int) -> None:
        """9.3: a fired watchdog does not prove a loss: probe first if the executor is alive, as far as the
        anchor knows (no last will). Whether it answers is up to the executor."""
        info = rec.attempts[attempt]
        info.timer = None
        if not info.live or not self.anchor.is_open(rec.task.task_id) or rec.result_attempt is not None:
            return
        e = info.executor
        if info.done_seen or self.neighbours.online.get(e) is False:
            self._lost(rec, attempt)  # its will said offline, or done but its result never came (Q23)
            return
        info.probes += 1
        n = info.probes
        key = AttemptKey(rec.task.task_id, self.zone_id, attempt)
        go = self._link(e, rec, attempt, f"probe{n}")
        back = self._link(e, rec, attempt, f"probe{n}_back")
        info.probe_timer = self._clock.schedule(
            self._clock.now + self._t_probe[self._kind(e)], lambda: self._probe_answer(rec, attempt, n, None)
        )
        call(
            self._clock,
            self._clock.now + go,
            lambda: self._bus.is_alive(e),
            lambda: self._executor(e).probe(key),
            lambda r: back,
            lambda r: self._probe_answer(rec, attempt, n, r),
            caller_alive=lambda: self.mqtt.alive,
        )

    def _link(self, e: str, rec: TaskRecord, attempt: int, hop: str) -> float:
        """One-way delay of a GET or DELETE to executor e: none here, the backhaul to a neighbour, backhaul
        and WAN to the cloud; keyed by task, attempt and hop (C3)."""
        if e == self.zone_id:
            return 0.0
        tid, lk = rec.task.task_id, self._cfg.links
        d = hops.delay(self._streams, hops.CALL_BH, lk.backhaul_delay, (*tid, hop), attempt)
        if e == CLOUD:
            d += hops.delay(self._streams, hops.CALL_WAN, lk.wan_delay, (*tid, hop), attempt)
        return d

    def _probe_answer(
        self, rec: TaskRecord, attempt: int, n: int, answer: tuple[str, float | None] | None
    ) -> None:
        """9.3: queued -> re-arm with the new estimate (a hard task that would now finish after d_A is retried
        elsewhere if the retry mask allows, and this attempt cancelled); running -> re-arm with the remaining
        Ê^high; done -> wait δ̂_pub + m_wd; skipped, aborted, not found or no reply (by T_probe) -> lost."""
        info = rec.attempts[attempt]
        if info.probes != n or info.probe_timer is None:
            return  # a later probe, or this one was already answered or timed out
        self._clock.cancel(info.probe_timer)
        info.probe_timer = None
        if not info.live or not self.anchor.is_open(rec.task.task_id) or rec.result_attempt is not None:
            return
        status, value = answer if answer is not None else ("no reply", None)
        now = self._clock.now
        if status == "queued" and value is not None:
            row = self._cfg.time_table.row(rec.task.task_type, self._hw(info.executor), rec.slot or "gpu")
            mean = row.mean if row is not None else 0.0
            t_ret = self._est.t_ret_cloud() if info.executor == CLOUD else self._est.t_ret_fog()
            if rec.task.cls == "hard" and now + value + mean + t_ret > rec.d_a:
                self._lost(rec, attempt)  # it would finish too late: retry elsewhere, cancel this one
                return
            self._arm(rec, attempt, now, value + self._row_high(rec, info.executor, rec.slot or "gpu"))
        elif status == "running" and value is not None:
            self._arm(rec, attempt, now, value)
        elif status == "done":
            info.done_seen = True  # wait δ̂_pub + m_wd for its result
            self._arm(rec, attempt, now + self._reply_delay(info.executor), 0.0)
        else:
            self._lost(rec, attempt)

    def _lost(self, rec: TaskRecord, attempt: int) -> None:
        """A lost attempt: cancel it (best effort) and retry if the retry mask allows, else drop (9.1)."""
        self._cancel_attempt(rec, attempt)
        rec.reply = "lost"
        self._retry(rec)

    def _retry(self, rec: TaskRecord) -> None:
        """9.1: the whole task again from the stored input, at most 1 retry (hard) or 2 (soft). A retry is a
        new decision with the normal masks (the retry mask): none open -> drop."""
        if rec.retries >= self._retry_limit[rec.task.cls] or rec.route != "agent":
            self._close(rec, "drop_lost")
            return
        rec.retries += 1
        ctx = self._context(rec.task, self._clock.now, rec.d_a)
        option = self._placement.choose(ctx)
        if option is None:
            self._close(rec, "drop_lost")
            return
        row = self.anchor.row(rec.task.task_id)
        self._send(rec, row, option, next(v for v in ctx.views if v.option == option))

    def _executor_died(self, zone_id: str) -> None:
        """9.1: its last will: every attempt this anchor placed there is handled at once."""
        for rec in list(self.records.values()):
            if not self.anchor.is_open(rec.task.task_id):
                continue
            for a, info in list(rec.attempts.items()):
                if info.live and info.executor == zone_id and rec.result_attempt is None:
                    self.store.clear_pending(zone_id, (rec.task.task_id, a))
                    self._lost(rec, a)
                    break

    def _end_attempt(self, info: AttemptInfo) -> None:
        info.live = False
        for timer in (info.timer, info.probe_timer):
            if timer is not None:
                self._clock.cancel(timer)
        info.timer = info.probe_timer = None

    def _cancel_attempt(self, rec: TaskRecord, attempt: int) -> None:
        """9.4.2: DELETE /tasks/{id}?attempt=n&anchor=A, best effort: queued -> removed, running -> aborted,
        unknown or finished -> ignored; twice is harmless. The attempt ends here at once."""
        info = rec.attempts[attempt]
        self._end_attempt(info)
        e = info.executor
        key = AttemptKey(rec.task.task_id, self.zone_id, attempt)
        call(
            self._clock,
            self._clock.now + self._link(e, rec, attempt, "cancel"),
            lambda: self._bus.is_alive(e),
            lambda: self._executor(e).cancel(key),
            lambda r: 0.0,
            lambda r: None,
        )

    def _no_ack(self, rec: TaskRecord) -> None:
        """9.5.5: no ack by the end (deadline, or deadline + L_max) is a drop (QUESTIONS Q23: + T_ack)."""
        rec.end_timer = None
        if self.anchor.is_open(rec.task.task_id):
            self._close(rec, "drop_no_ack")

    # --- delivery, ack, close (6D, 6E, 8.6, 9.5, 10.3) ------------------------------------------------------

    def _on_message(self, d: Delivery) -> None:
        scope, car_id, kind = topics.parse(d.topic)
        if scope == "zone" and kind == "status":
            st = d.payload
            assert isinstance(st, ZoneStatus)
            before = set(self.neighbours.positions)
            self.neighbours.on_status(st.site, st.status == "online")  # 7.5.2-3
            if st.status == "offline":
                self._executor_died(st.zone_id)  # 9.1: every attempt placed there, at once
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
        elif kind == "gps":
            g = d.payload
            assert isinstance(g, GpsMessage)
            self.registry[g.car_id] = g  # 10.1: the anchor always knows the car's current zone
        elif kind == "attach":
            payload = d.payload
            assert isinstance(payload, tuple)
            car, zone = str(payload[0]), str(payload[1])  # (car_id, zone_id), 5.2
            if zone != self.zone_id and car in self._cars:
                self.detach(car)  # 9.6: another zone serves the car now
            self._republish_for(car)  # 6D.14, 9.5: until the ack, republish on every new attach

    def _anchor_result(self, msg: ResultMessage) -> None:
        rec = self.records.get(msg.key.task_id)
        if rec is None or not self.anchor.is_open(rec.task.task_id):
            return
        report = msg.report
        a = msg.key.attempt
        info = rec.attempts.get(a)
        if msg.kind == "status":  # skipped, aborted (8.6) or cancelled by this anchor
            if info is None or not info.live:
                return  # an attempt this anchor already gave up or cancelled
            self._end_attempt(info)
            rec.report = report
            if not any(i.live for i in rec.attempts.values()) and rec.result_attempt is None:
                self._close(rec, f"drop_{report.outcome}")
            return
        if rec.result_attempt is not None:
            return  # 9.4.3: the first result wins; the car filters the copy
        rec.result_attempt, rec.report, rec.copy = a, report, msg
        self._clock.schedule(self._clock.now + self._t_ack, lambda: self._republish_once(rec))  # 9.5.4
        if info is not None:
            self._end_attempt(info)
        for other, i in rec.attempts.items():
            if other != a and i.live:
                self._cancel_attempt(rec, other)  # 9.4.3: cancel any attempt still alive

    def _push(self, car_id: str, msg: ResultMessage) -> None:
        """The serving zone pushes a result to the car over the radio downlink (10.3), unless the car already
        confirmed that task id in this radio session (9.5.3)."""
        k = msg.key
        if k.task_id in self._confirmed.get(car_id, ()):
            return
        down = hops.delay(
            self._streams, hops.RADIO_DOWN, self._cfg.radio.downlink_delay, k.task_id, k.attempt
        )
        self._clock.schedule(self._clock.now + down, lambda: self._at_car(car_id, msg))

    def _at_car(self, car_id: str, msg: ResultMessage) -> None:
        if car_id not in self._cars:
            return  # the car left this access point before the copy reached it: the anchor republishes
        car, _ = self._cars[car_id]
        now = self._clock.now
        k = msg.key
        rec = self.ledger.get((k.anchor_id, k.task_id))  # the run's log (the anchor may be another zone)
        if rec is not None:
            rec.deliveries += 1
            info = rec.attempts.get(k.attempt)
            if rec.result_at_car is None and info is not None and msg.report.start is not None:
                # the latency parts of the copy that reached the car first (the one it keeps, 9.5.1)
                rec.result_at_car = now
                rec.parts["before_send"] = info.t_send - rec.t_rx
                rec.parts.pop("input", None)
                if info.input:
                    rec.parts["input"] = info.input
                assert info.t_at_executor is not None
                rec.parts["queue"] = msg.report.start - info.t_at_executor
                rec.parts["run"] = msg.report.busy
                rec.parts["return"] = now - msg.report.finish
        confirm = car.receive(Result(TaskId(*k.task_id), k.anchor_id, k.attempt), now)  # type: ignore[arg-type]
        if confirm is None:
            return
        self._confirmed.setdefault(car_id, set()).add(k.task_id)  # 9.5.3: this session has it
        # the car's confirm crosses the radio to this serving zone, which publishes the ack (6D.13)
        up = hops.delay(self._streams, hops.ACK_UP, self._cfg.radio.uplink_delay, k.task_id, k.attempt)
        ack = AckMessage(confirm, self.zone_id)
        key = (*k.task_id, k.anchor_id, k.attempt, "ack")
        self._clock.schedule(now + up, lambda: self.mqtt.publish(topics.car_ack(car_id), ack, qos=1, key=key))

    def _republish(self, rec: TaskRecord) -> None:
        """The anchor publishes its copy of the result again (QoS 1), keyed as a new message (9.5)."""
        assert rec.copy is not None
        k = rec.copy.key
        n = rec.deliveries + int(rec.republished) + 1
        topic = topics.car_result(rec.task.car_id)
        self.mqtt.publish(topic, rec.copy, qos=1, key=(*k.task_id, k.anchor_id, k.attempt, "republish", n))

    def _republish_once(self, rec: TaskRecord) -> None:
        """9.5.4: the anchor saw the result but got no ack within T_ack: it republishes once (this timer is
        set once, at the result; an ack closes the task)."""
        if self.anchor.is_open(rec.task.task_id):
            rec.republished = True
            self._republish(rec)

    def _republish_for(self, car_id: str) -> None:
        """6D.14: on a new attach, republish the copy of every open task of the car that has a result but
        no ack yet."""
        for rec in self.records.values():
            open_ = self.anchor.is_open(rec.task.task_id)
            if open_ and rec.task.car_id == car_id and rec.copy is not None and rec.confirm is None:
                self._republish(rec)

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
        """6E.15: close; cancel any other live attempt; stop every timer."""
        now = self._clock.now
        self.anchor.close(rec.task.task_id, outcome, now)  # raises if closed twice
        self._anchor_role(rec.task.car_id, -1)  # 9.6
        rec.copy = None  # 6E.15: the result copy is deleted
        if rec.end_timer is not None:
            self._clock.cancel(rec.end_timer)
        for a, info in rec.attempts.items():
            if info.live:
                self._cancel_attempt(rec, a)
        rec.outcome, rec.closed_at = outcome, now
        if self._on_close is not None:
            self._on_close(rec)
