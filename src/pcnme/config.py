"""Config loader and derived values (final-design 18).

`load_config` reads `config/default.yaml` into typed, frozen dataclasses and computes the derived values of
section 18 (tag C) the same way as `tools/calc_defaults.py`. Every key is required and every unknown key is
refused, so a missing or misspelled key fails at load time with its dotted path (`ConfigError.key`).

Units are converted here, at the edge (CLAUDE.md): seconds, bits, bits per second, watts, joules, metres,
metres per second, and linear gains. KB in the config are 1 000 bytes.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Any, Literal, TypeVar, cast

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

SlotType = Literal["gpu", "cpu"]
DeadlineMode = Literal["loose", "tight"]
TaskClass = Literal["hard", "soft"]
Mobility = Literal["low", "high"]

SLOT_TYPES: tuple[SlotType, ...] = ("gpu", "cpu")
DEADLINE_MODES: tuple[DeadlineMode, ...] = ("loose", "tight")
CLOUD = "cloud"  # the hardware class of the cloud (final-design 3); every other class is a fog class

# Unit conversions (not model numbers).
_BITS_PER_KB = 8_000
_J_PER_NJ = 1e-9
_MPS_PER_KMH = 1_000 / 3_600

# final-design 18 (Links): B_WAN is set from the cloud-eligible bits per second at S2.
_WAN_CALIBRATION_SCENARIO = "S2"

K = TypeVar("K")
V = TypeVar("V")


class ConfigError(ValueError):
    """A config value is missing, misspelled, unknown, of the wrong type or out of range."""

    def __init__(self, key: str, problem: str) -> None:
        super().__init__(f"config key '{key}': {problem}")
        self.key = key


# --- Reading YAML -------------------------------------------------------------------------------------------

# YAML 1.2 floats. PyYAML follows YAML 1.1, which reads 40.0e6 or 1e9 (no exponent sign) as strings.
# The pattern needs a dot or an exponent, so integers stay integers.
_FLOAT_12 = re.compile(r"^[-+]?(?:(?:\d[\d_]*\.[\d_]*|\.\d[\d_]*)(?:[eE][-+]?\d+)?|\d[\d_]*[eE][-+]?\d+)$")


class _Yaml12Loader(yaml.SafeLoader):
    pass


_Yaml12Loader.add_implicit_resolver("tag:yaml.org,2002:float", _FLOAT_12, list("-+0123456789."))


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file with YAML 1.2 float rules (safe loader: plain data only)."""
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=_Yaml12Loader)
    if not isinstance(data, dict):
        raise ConfigError("<root>", f"{path} must hold a mapping, got {_kind(data)}")
    return data


def _kind(value: object) -> str:
    return "nothing" if value is None else type(value).__name__


def _frozen(items: Mapping[K, V]) -> Mapping[K, V]:
    return MappingProxyType(dict(items))


class _Node:
    """One mapping of the YAML tree. Every key must be read; `done` refuses the keys nobody read.

    Used as a context manager: leaving the `with` block without an error calls `done`.
    """

    def __init__(self, data: object, path: str) -> None:
        if not isinstance(data, dict):
            raise ConfigError(path, f"must be a mapping, got {_kind(data)}")
        for k in data:
            if not isinstance(k, str):
                raise ConfigError(f"{path}.{k}" if path else str(k), "keys must be text")
        self._data: dict[str, object] = data
        self.path = path
        self._read: set[str] = set()

    def __enter__(self) -> _Node:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if exc_type is None:
            self.done()

    def key(self, name: str) -> str:
        return f"{self.path}.{name}" if self.path else name

    def names(self) -> list[str]:
        return list(self._data)

    def has(self, name: str) -> bool:
        return name in self._data

    def raw(self, name: str) -> object:
        if name not in self._data:
            unread = [k for k in self._data if k not in self._read]
            close = difflib.get_close_matches(name, unread, n=1, cutoff=0.6)
            hint = f" (found '{close[0]}': misspelled?)" if close else ""
            raise ConfigError(self.key(name), f"missing{hint}")
        self._read.add(name)
        return self._data[name]

    def done(self) -> None:
        unread = [k for k in self._data if k not in self._read]
        if unread:
            raise ConfigError(self.key(unread[0]), "unknown key (misspelled, or not in final-design 18?)")

    def node(self, name: str) -> _Node:
        return _Node(self.raw(name), self.key(name))

    def number(
        self, name: str, *, gt: float | None = None, ge: float | None = None, le: float | None = None
    ) -> float:
        return _number(self.raw(name), self.key(name), gt=gt, ge=ge, le=le)

    def integer(self, name: str, *, ge: int = 0) -> int:
        value = self.raw(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(self.key(name), f"must be a whole number, got {_kind(value)} {value!r}")
        if value < ge:
            raise ConfigError(self.key(name), f"must be >= {ge}, got {value}")
        return value

    def text(self, name: str, choices: tuple[str, ...] | None = None) -> str:
        value = self.raw(name)
        if not isinstance(value, str):
            raise ConfigError(self.key(name), f"must be text, got {_kind(value)} {value!r}")
        if choices is not None and value not in choices:
            raise ConfigError(self.key(name), f"must be one of {', '.join(choices)}; got {value!r}")
        return value

    def flag(self, name: str) -> bool:
        value = self.raw(name)
        if not isinstance(value, bool):
            raise ConfigError(self.key(name), f"must be true or false, got {_kind(value)} {value!r}")
        return value

    def marker(self, name: str, expected: str) -> None:
        """A value computed by the code (tag C in final-design 18): the config must say so, nothing else."""
        value = self.raw(name)
        if value != expected:
            raise ConfigError(
                self.key(name), f"is computed by the loader: must be {expected!r}, got {value!r}"
            )

    def items(self, name: str) -> list[object]:
        value = self.raw(name)
        if not isinstance(value, list) or not value:
            raise ConfigError(self.key(name), f"must be a non-empty list, got {_kind(value)} {value!r}")
        return value

    def numbers(self, name: str, *, gt: float | None = None, ge: float | None = None) -> tuple[float, ...]:
        return tuple(
            _number(v, f"{self.key(name)}[{i}]", gt=gt, ge=ge) for i, v in enumerate(self.items(name))
        )

    def interval(self, name: str) -> tuple[float, float]:
        """A uniform range [low, high] with 0 <= low <= high."""
        values = self.numbers(name, ge=0.0)
        if len(values) != 2 or values[0] > values[1]:
            raise ConfigError(self.key(name), f"must be [low, high] with low <= high, got {list(values)}")
        return values[0], values[1]

    def texts(self, name: str) -> tuple[str, ...]:
        out = []
        for i, v in enumerate(self.items(name)):
            if not isinstance(v, str):
                raise ConfigError(f"{self.key(name)}[{i}]", f"must be text, got {_kind(v)} {v!r}")
            out.append(v)
        return tuple(out)


def _number(
    value: object, key: str, *, gt: float | None = None, ge: float | None = None, le: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(key, f"must be a number, got {_kind(value)} {value!r}")
    x = float(value)
    if gt is not None and not x > gt:
        raise ConfigError(key, f"must be > {gt}, got {x}")
    if ge is not None and not x >= ge:
        raise ConfigError(key, f"must be >= {ge}, got {x}")
    if le is not None and not x <= le:
        raise ConfigError(key, f"must be <= {le}, got {x}")
    return x


def _fraction(node: _Node, name: str) -> float:
    return node.number(name, ge=0.0, le=1.0)


def _db_to_linear(db: float) -> float:
    return float(10 ** (db / 10))


# --- Sections -----------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Layout:
    """Zone layout (final-design 18, Layout; 7.5)."""

    area_m: tuple[float, float]
    zone_radius_m: float
    zone_positions_m: Mapping[str, tuple[float, float]]
    joining_zones: tuple[str, ...]
    d_nb_m: float
    handover_hysteresis_m: float
    map_dev: str
    map_final: str

    @property
    def zones(self) -> tuple[str, ...]:
        """Every zone, in config order, the joining ones included."""
        return tuple(self.zone_positions_m)

    @property
    def active_zones(self) -> tuple[str, ...]:
        """The zones online at the start (every zone but the joining ones)."""
        return tuple(z for z in self.zone_positions_m if z not in self.joining_zones)


@dataclass(frozen=True, slots=True)
class Cars:
    count: tuple[int, int]
    gps_period: float
    speed_low_mps: tuple[float, ...]
    speed_high_mps: tuple[float, ...]
    hotspot_share: float
    grid: Grid
    calibration: Calibration
    coverage: Mapping[str, float]  # per mobility: the share of task creations with coverage (Q29, C)
    hotspot_h: Mapping[str, float]  # per mobility: the share of cars on z1's roads (Q30, C)


@dataclass(frozen=True, slots=True)
class Calibration:
    seconds: float
    seed: int


@dataclass(frozen=True, slots=True)
class Grid:
    """The dev map: the VFC grid (final-design 16.2; stand-ins for lights and stops: QUESTIONS Q24)."""

    block_m: float
    complex: bool
    light_cycle_s: float
    stop_probability: float
    stop_s: tuple[float, float]


@dataclass(frozen=True, slots=True)
class Radio:
    """Radio uplink (final-design 4.2, 18; CHANGES C1). Gains are linear, noise in W/Hz."""

    bandwidth_hz: float
    tx_power_w: float
    gain_tx: float
    gain_rx: float
    noise_w_per_hz: float
    carrier_hz: float
    uplink_delay: tuple[float, float]
    downlink_delay: tuple[float, float]


@dataclass(frozen=True, slots=True)
class Links:
    """Backhaul and WAN (final-design 4.2). B_WAN is derived: `Derived.b_wan`."""

    backhaul_bw: float
    backhaul_delay: tuple[float, float]
    wan_delay: tuple[float, float]
    broker_extra_delay: float


@dataclass(frozen=True, slots=True)
class TaskType:
    """One task type (final-design 14.1, 18). Deadlines are resolved: T2's tight D is a number here."""

    name: str
    model: str
    cls: TaskClass
    share: float
    input_bits: float
    result_bits: float
    deadline: Mapping[DeadlineMode, float]
    l_max: Mapping[DeadlineMode, float]


@dataclass(frozen=True, slots=True)
class HardwareClass:
    """Slots, queues and powers of one hardware class (final-design 8.1, 8.2, 18)."""

    name: str
    c_gpu: int
    c_cpu: int
    queue: Mapping[SlotType, int]
    p_dyn_w: Mapping[SlotType, float]
    p_idle_w: float

    @property
    def is_cloud(self) -> bool:
        return self.name == CLOUD

    @property
    def slot_types(self) -> tuple[SlotType, ...]:
        return tuple(q for q in SLOT_TYPES if self.slots(q) > 0)

    def slots(self, q: SlotType) -> int:
        return self.c_gpu if q == "gpu" else self.c_cpu


@dataclass(frozen=True, slots=True)
class TimeRow:
    """One time-table row (final-design 14.4), in seconds."""

    mean: float
    p5: float
    p95: float
    longest_step: float
    p5_measured: bool = False  # False: the placeholder factor filled p5 (QUESTIONS Q2, CHANGES C5)


@dataclass(frozen=True, slots=True)
class TimeTable:
    """Run times per (task type, hardware class, slot type). A missing row = cannot run there (14.4)."""

    rows: Mapping[tuple[str, str, SlotType], TimeRow]

    def row(self, task: str, hw: str, q: SlotType) -> TimeRow | None:
        return self.rows.get((task, hw, q))

    def slot_types(self, task: str, hw: str) -> tuple[SlotType, ...]:
        return tuple(q for q in SLOT_TYPES if (task, hw, q) in self.rows)

    def fastest(self, task: str, hw: str) -> tuple[SlotType, TimeRow]:
        """The slot type with the lowest mean run (ties: GPU first)."""
        options = [(q, row) for q in SLOT_TYPES if (row := self.row(task, hw, q)) is not None]
        if not options:
            raise KeyError(f"{task} cannot run on {hw}")
        return min(options, key=lambda option: option[1].mean)


@dataclass(frozen=True, slots=True)
class Energy:
    """Energy constants (final-design 12, 18). Per-bit energies are in J/bit."""

    pue_fog: float
    pue_cloud: float
    pue_network: float
    e_bh_device: float
    e_wan_device: float
    e_down_device: float
    car_tx_w: float
    reference_class: str

    @property
    def e_bh(self) -> float:
        """Backhaul J/bit, x PUE_network (18)."""
        return self.e_bh_device * self.pue_network

    @property
    def e_wan(self) -> float:
        """WAN J/bit, x PUE_network (18)."""
        return self.e_wan_device * self.pue_network

    @property
    def e_down(self) -> float:
        """Radio downlink J/bit. Not x PUE_network: the access point is part of the edge node (Q3)."""
        return self.e_down_device

    def pue(self, hw: HardwareClass) -> float:
        return self.pue_cloud if hw.is_cloud else self.pue_fog


@dataclass(frozen=True, slots=True)
class StatePlane:
    state_period: float
    stale_after: float
    wan_ewma_lambda: float


@dataclass(frozen=True, slots=True)
class Scheduler:
    reserve_hard_r: int
    headroom_h: int
    soft_promotion_a_max: float
    probe_timeout_fog: float
    probe_timeout_cloud: float
    t_keep: float
    retries_hard: int
    retries_soft: int


@dataclass(frozen=True, slots=True)
class Timing:
    kappa_up: float
    m_wd_fog: float
    m_wd_cloud: float
    quantile_low: float
    quantile_high: float
    t_ack: float
    t_leave: float
    gps_expiry: float
    reward_grace_g: float
    encode: float  # the car's encode time, inside L; the upload starts after it (14.3, QUESTIONS Q14)


@dataclass(frozen=True, slots=True)
class Tof:
    """TOF-Broker (final-design 7.1)."""

    alpha: float
    alpha_sweep: tuple[float, ...]
    strongest_class: str


@dataclass(frozen=True, slots=True)
class Reward:
    """Reward constants (final-design 11.4). w = w1, the time weight (w2 = 1 - w): the median knee weight
    w* (13.3) once M5 computes it; until then a stand-in from the config (QUESTIONS Q32)."""

    b: float
    p: float
    w3: float
    c_drop: float
    w: float


@dataclass(frozen=True, slots=True)
class EpsilonSchedule:
    """Linear decay from `start` to `end` over the first `fraction` of the budget."""

    start: float
    end: float
    fraction: float


@dataclass(frozen=True, slots=True)
class Ppo:
    clip: float
    lr: float
    gae_lambda: float
    epochs: int
    minibatch: int
    rollout: int  # decisions per zone between two updates (M7)
    vf_coef: float
    ent_coef: float


@dataclass(frozen=True, slots=True)
class Rl:
    gamma_per_tau: float
    tau: float
    hidden: tuple[int, ...]
    dueling: bool
    double: bool
    optimizer: Literal["adam"]
    lr: float
    batch: int
    replay_agent: int
    target_update_every: int
    updates_per_transition: float
    epsilon_online: float
    epsilon_baselines: EpsilonSchedule
    loss: Literal["mse", "huber"]
    lstm_history_h: int
    ppo: Ppo


@dataclass(frozen=True, slots=True)
class Per:
    """Prioritized replay (final-design 13.5, 18)."""

    alpha: float
    beta0: float
    beta_final: float
    eps_demo: float
    eps_agent: float


@dataclass(frozen=True, slots=True)
class Dqfd:
    n_step: int
    lambda_n: float
    lambda_margin: float
    lambda_l2: float
    margin: float
    per: Per
    pretrain_updates: int
    eval_every: int  # held-out expert agreement every this many updates (M6.4)
    patience: int  # early stop after this many measurements without a rise
    holdout_share: float  # share of the demonstration windows held out


@dataclass(frozen=True, slots=True)
class Nsga2:
    """NSGA-II (final-design 13.2, 18). The mutation rate is 1 / number of genes, set per window."""

    population: int
    generations: int
    crossover_prob: float
    t_win: float
    t_warmup: float
    windows_per_scenario: int
    expert_filter_margin: float


@dataclass(frozen=True, slots=True)
class FastMode:
    episode_length: float
    eval_seeds: int
    testbed_seeds: int
    train_decisions: int  # the learning methods' training budget in decisions (16.1; QUESTIONS Q41)
    early_decisions: int  # N of early-stage performance (16.3; QUESTIONS Q46)
    seed_split: Mapping[str, range]


@dataclass(frozen=True, slots=True)
class Scenario:
    """One scenario (final-design 16.2). `fog` gives every zone's fog class, joining zones included."""

    name: str
    rho_fog: float
    deadlines: DeadlineMode
    mobility: Mobility
    fog: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class FedAvg:
    t_fl_fast_mode: float
    t_fl_testbed: float
    t_fl_real_world: float
    upload_deadline: float
    finetune_k: int


@dataclass(frozen=True, slots=True)
class Derived:
    """Values computed when the config loads (tag C in final-design 18)."""

    mean_gpu_run: float  # s: the whole mix on the reference class's GPU slot (16.2)
    zone_capacity: float  # tasks/s: c_GPU / mean_gpu_run
    e_strong: Mapping[str, float]  # s: mean run on the strongest fog class, fastest slot type (7.1)
    b_wan: float  # bit/s (18, Links)
    arrival_rate_per_zone: Mapping[str, float]  # tasks/s, the city mean, per scenario (18, Workload)
    ti_wan: Mapping[str, float]  # WAN traffic intensity per scenario (16.2)
    e_ref: Mapping[str, float]  # J per task type (12.3)
    e_bar: float  # the cap on energy ratios (12.3)


@dataclass(frozen=True, slots=True)
class Config:
    layout: Layout
    cars: Cars
    radio: Radio
    links: Links
    task_types: Mapping[str, TaskType]  # enabled types, config order
    disabled_task_types: tuple[str, ...]
    hardware: Mapping[str, HardwareClass]
    time_table: TimeTable
    energy: Energy
    stateplane: StatePlane
    scheduler: Scheduler
    timing: Timing
    tof: Tof
    reward: Reward
    rl: Rl
    dqfd: Dqfd
    nsga2: Nsga2
    fast_mode: FastMode
    scenarios: Mapping[str, Scenario]
    fedavg: FedAvg
    derived: Derived

    def is_boulder(self, task: str, mode: DeadlineMode) -> bool:
        """final-design 7.1: boulder iff E_strong >= alpha x D."""
        return _is_boulder(self.derived.e_strong[task], self.tof.alpha, self.task_types[task].deadline[mode])

    def arrival_rate_per_car(self, scenario: str, n_cars: int) -> float:
        """lambda_car = rho_fog x c_GPU / (mean GPU run x cars per zone) (final-design 18, Workload)."""
        if n_cars <= 0:
            raise ValueError(f"n_cars must be > 0, got {n_cars}")
        cars_per_zone = n_cars / len(self.layout.active_zones)
        return self.derived.arrival_rate_per_zone[scenario] / cars_per_zone


def _is_boulder(e_strong: float, alpha: float, deadline: float) -> bool:
    return e_strong >= alpha * deadline  # final-design 7.1 step 1


# --- Loading ------------------------------------------------------------------------------------------------


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load a config file and compute its derived values."""
    return config_from_mapping(load_yaml(path))


def config_from_mapping(data: Mapping[str, object]) -> Config:
    """Build a Config from already-parsed YAML data (tests edit a copy of the default)."""
    root = _Node(dict(data), "")
    with root:
        layout = _layout(root.node("layout"))
        cars = _cars(root.node("cars"))
        radio = _radio(root.node("radio"))
        links = _links(root.node("links"))
        hardware = _hardware(root.node("hardware"))
        energy = _energy(root.node("energy"), hardware)
        tof = _tof(root.node("tof"), hardware)
        types_node = root.node("task_types")
        type_specs, disabled = _task_type_specs(types_node)
        time_table = _time_table(root.node("time_table_placeholder"), type_specs, hardware)
        e_strong = _e_strong(type_specs, time_table, tof.strongest_class)
        task_types = _resolve_task_types(type_specs, e_strong, types_node.path)
        scenarios = _scenarios(root.node("scenarios"), layout, hardware)
        config = Config(
            layout=layout,
            cars=cars,
            radio=radio,
            links=links,
            task_types=task_types,
            disabled_task_types=disabled,
            hardware=hardware,
            time_table=time_table,
            energy=energy,
            stateplane=_stateplane(root.node("stateplane")),
            scheduler=_scheduler(root.node("scheduler")),
            timing=_timing(root.node("timing")),
            tof=tof,
            reward=_reward(root.node("reward")),
            rl=_rl(root.node("rl")),
            dqfd=_dqfd(root.node("dqfd")),
            nsga2=_nsga2(root.node("nsga2")),
            fast_mode=_fast_mode(root.node("fast_mode")),
            scenarios=scenarios,
            fedavg=_fedavg(root.node("fedavg")),
            derived=_derive(layout, task_types, hardware, time_table, energy, tof, scenarios, e_strong),
        )
    return config


def _layout(n: _Node) -> Layout:
    with n:
        area = n.numbers("area_m", gt=0.0)
        if len(area) != 2:
            raise ConfigError(n.key("area_m"), f"must be [width, height], got {list(area)}")
        positions: dict[str, tuple[float, float]] = {}
        with n.node("zone_positions_m") as zp:
            for zone in zp.names():
                xy = zp.numbers(zone)
                if len(xy) != 2:
                    raise ConfigError(zp.key(zone), f"must be [x, y], got {list(xy)}")
                positions[zone] = (xy[0], xy[1])
        joining = n.texts("joining_zones")
        for zone in joining:
            if zone not in positions:
                raise ConfigError(n.key("joining_zones"), f"{zone!r} is not in zone_positions_m")
        if len(set(joining)) == len(positions):
            raise ConfigError(n.key("joining_zones"), "at least one zone must be active at the start")
        with n.node("maps") as maps:
            map_dev, map_final = maps.text("dev"), maps.text("final")
        return Layout(
            area_m=(area[0], area[1]),
            zone_radius_m=n.number("zone_radius_m", gt=0.0),
            zone_positions_m=_frozen(positions),
            joining_zones=joining,
            d_nb_m=n.number("d_nb_m", ge=0.0),
            handover_hysteresis_m=n.number("handover_hysteresis_m", ge=0.0),
            map_dev=map_dev,
            map_final=map_final,
        )


def _calibration(n: _Node) -> Calibration:
    with n:
        return Calibration(seconds=n.number("seconds", gt=0.0), seed=n.integer("seed"))


def _calibrated(n: _Node) -> dict[str, Mapping[str, float]]:
    """cars.calibrated (tag C, Q29, Q30): per mobility, coverage and hotspot_h."""
    out: dict[str, Mapping[str, float]] = {}
    with n:
        for name in ("coverage", "hotspot_h"):
            with n.node(name) as c:
                out[name] = _frozen({m: c.number(m, ge=0.0, le=1.0) for m in ("low", "high")})
    return out


def _grid(n: _Node) -> Grid:
    with n:
        stop = n.numbers("stop_s", ge=0.0)
        if len(stop) != 2 or stop[0] > stop[1]:
            raise ConfigError(n.key("stop_s"), f"must be [shortest, longest], got {list(stop)}")
        flag = n.raw("complex")
        if not isinstance(flag, bool):
            raise ConfigError(n.key("complex"), f"must be true or false, got {flag!r}")
        return Grid(
            block_m=n.number("block_m", gt=0.0),
            complex=flag,
            light_cycle_s=n.number("light_cycle_s", gt=0.0),
            stop_probability=_fraction(n, "stop_probability"),
            stop_s=(stop[0], stop[1]),
        )


def _cars(n: _Node) -> Cars:
    with n:
        count = n.items("count")
        if len(count) != 2 or not all(isinstance(c, int) and not isinstance(c, bool) for c in count):
            raise ConfigError(n.key("count"), f"must be [fewest, most] whole numbers, got {count}")
        low, high = cast(int, count[0]), cast(int, count[1])
        if not 0 < low <= high:
            raise ConfigError(n.key("count"), f"must satisfy 0 < fewest <= most, got {count}")
        with n.node("speed_kmh") as speeds:
            slow = tuple(v * _MPS_PER_KMH for v in speeds.numbers("low", gt=0.0))
            fast = tuple(v * _MPS_PER_KMH for v in speeds.numbers("high", gt=0.0))
        calibrated = _calibrated(n.node("calibrated"))
        return Cars(
            count=(low, high),
            gps_period=n.number("gps_period", gt=0.0),
            speed_low_mps=slow,
            speed_high_mps=fast,
            hotspot_share=_fraction(n, "hotspot_share"),
            grid=_grid(n.node("grid")),
            calibration=_calibration(n.node("calibration")),
            coverage=calibrated["coverage"],
            hotspot_h=calibrated["hotspot_h"],
        )


def _radio(n: _Node) -> Radio:
    with n:
        return Radio(
            bandwidth_hz=n.number("bandwidth", gt=0.0),
            tx_power_w=n.number("tx_power_w", gt=0.0),
            gain_tx=_db_to_linear(n.number("gain_tx_dbi")),
            gain_rx=_db_to_linear(n.number("gain_rx_dbi")),
            noise_w_per_hz=_db_to_linear(n.number("noise_dbm_per_hz") - 30),  # dBm -> dBW (CHANGES C1)
            carrier_hz=n.number("carrier_hz", gt=0.0),
            uplink_delay=n.interval("delay"),
            downlink_delay=n.interval("downlink_delay"),
        )


def _links(n: _Node) -> Links:
    with n:
        n.marker("wan_bw", "calibrated")  # final-design 18: B_WAN is derived
        return Links(
            backhaul_bw=n.number("backhaul_bw", gt=0.0),
            backhaul_delay=n.interval("backhaul_delay"),
            wan_delay=n.interval("wan_delay"),
            broker_extra_delay=n.number("broker_extra_delay", ge=0.0),
        )


@dataclass(frozen=True, slots=True)
class _TaskSpec:
    """A task type before its deadlines are resolved: a tight D may be `factor x E_strong` (18)."""

    name: str
    model: str
    cls: TaskClass
    share: float
    input_bits: float
    result_bits: float
    deadline: Mapping[DeadlineMode, float | tuple[Literal["x_E_strong"], float]]
    key: str


_E_STRONG_RULE = re.compile(r"^(\d+(?:\.\d+)?)x_E_strong$")


def _task_type_specs(n: _Node) -> tuple[dict[str, _TaskSpec], tuple[str, ...]]:
    with n:
        specs: dict[str, _TaskSpec] = {}
        disabled: list[str] = []
        for name in n.names():
            if name == "L_max":
                n.marker("L_max", "equal_to_D")  # final-design 2: L_max = D
                continue
            with n.node(name) as t:
                if t.has("enabled") and not t.flag("enabled"):
                    disabled.append(name)
                    continue
                specs[name] = _task_spec(t, name)
        if not specs:
            raise ConfigError(n.path, "no enabled task type")
        total = sum(s.share for s in specs.values())
        if abs(total - 1.0) > 1e-9:
            raise ConfigError(n.path, f"the shares of the enabled task types sum to {total}, not 1")
        return specs, tuple(disabled)


def _task_spec(t: _Node, name: str) -> _TaskSpec:
    deadline: dict[DeadlineMode, float | tuple[Literal["x_E_strong"], float]] = {}
    with t.node("deadline") as d:
        for mode in DEADLINE_MODES:
            value = d.raw(mode)
            if isinstance(value, str):
                match = _E_STRONG_RULE.match(value)
                if not match or float(match.group(1)) <= 0:
                    raise ConfigError(d.key(mode), f"must be seconds or '<factor>x_E_strong', got {value!r}")
                deadline[mode] = ("x_E_strong", float(match.group(1)))
            else:
                deadline[mode] = _number(value, d.key(mode), gt=0.0)
    return _TaskSpec(
        name=name,
        model=t.text("model"),
        cls=cast(TaskClass, t.text("cls", ("hard", "soft"))),
        share=_fraction(t, "share"),
        input_bits=t.number("input_kb", gt=0.0) * _BITS_PER_KB,
        result_bits=t.number("result_kb", gt=0.0) * _BITS_PER_KB,
        deadline=deadline,
        key=t.path,
    )


def _resolve_task_types(
    specs: Mapping[str, _TaskSpec], e_strong: Mapping[str, float], path: str
) -> Mapping[str, TaskType]:
    out: dict[str, TaskType] = {}
    for name, s in specs.items():
        deadline: dict[DeadlineMode, float] = {}
        for mode, rule in s.deadline.items():
            # final-design 18 (Workload): T2's tight D = 1.1 x E_strong(T2)
            deadline[mode] = rule[1] * e_strong[name] if isinstance(rule, tuple) else rule
        frozen = _frozen(deadline)
        out[name] = TaskType(
            name=name,
            model=s.model,
            cls=s.cls,
            share=s.share,
            input_bits=s.input_bits,
            result_bits=s.result_bits,
            deadline=frozen,
            l_max=frozen,  # final-design 2: L_max = D
        )
    return _frozen(out)


def _hardware(n: _Node) -> Mapping[str, HardwareClass]:
    with n:
        classes: dict[str, HardwareClass] = {}
        for name in n.names():
            with n.node(name) as h:
                c_gpu, c_cpu = h.integer("c_gpu"), h.integer("c_cpu")
                counts = (c_gpu, c_cpu)
                present = tuple(q for q, c in zip(SLOT_TYPES, counts, strict=True) if c > 0)
                if not present:
                    raise ConfigError(h.path, "needs at least one GPU or CPU slot")
                with h.node("queue") as qn:
                    queue = {q: qn.integer(q, ge=1) for q in present}
                with h.node("p_dyn_w") as pn:
                    power = {q: pn.number(f"{q}_slot", gt=0.0) for q in present}
                classes[name] = HardwareClass(
                    name=name,
                    c_gpu=c_gpu,
                    c_cpu=c_cpu,
                    queue=_frozen(queue),
                    p_dyn_w=_frozen(power),
                    p_idle_w=h.number("p_idle_w", ge=0.0),
                )
        if CLOUD not in classes:
            raise ConfigError(n.key(CLOUD), "missing (the cloud's hardware class)")
        if len(classes) == 1:
            raise ConfigError(n.path, "needs at least one fog class")
        return _frozen(classes)


def _fog_class(n: _Node, name: str, hardware: Mapping[str, HardwareClass]) -> str:
    value = n.text(name)
    if value not in hardware or hardware[value].is_cloud:
        fog = ", ".join(h for h in hardware if not hardware[h].is_cloud)
        raise ConfigError(n.key(name), f"must be a fog hardware class ({fog}), got {value!r}")
    return value


def _time_table(n: _Node, specs: Mapping[str, _TaskSpec], hardware: Mapping[str, HardwareClass]) -> TimeTable:
    with n:
        with n.node("factors") as f:
            p5_f = f.number("p5", gt=0.0, le=1.0)
            p95_f = f.number("p95", ge=1.0)
            step_f = f.number("longest_step", gt=0.0, le=1.0)
        rows: dict[tuple[str, str, SlotType], TimeRow] = {}
        for task in n.names():
            if task == "factors":
                continue
            if task not in specs:
                raise ConfigError(n.key(task), "is not an enabled task type")
            with n.node(task) as t:
                for hw in t.names():
                    if hw not in hardware:
                        raise ConfigError(t.key(hw), "is not a hardware class")
                    with t.node(hw) as h:
                        for q in h.names():
                            if q not in hardware[hw].slot_types:
                                raise ConfigError(h.key(q), f"{hw} has no {q.upper()} slot")
                            slot = cast(SlotType, q)
                            rows[(task, hw, slot)] = _time_row(h, q, p5_f, p95_f, step_f)
        return TimeTable(rows=_frozen(rows))


def _time_row(h: _Node, q: str, p5_f: float, p95_f: float, step_f: float) -> TimeRow:
    """A mean alone, or measured values; the placeholder factors fill only what is not measured (Q2)."""
    value = h.raw(q)
    key = h.key(q)
    if not isinstance(value, dict):
        mean = _number(value, key, gt=0.0)
        return TimeRow(mean=mean, p5=p5_f * mean, p95=p95_f * mean, longest_step=step_f * mean)
    with _Node(value, key) as r:
        mean = r.number("mean", gt=0.0)
        measured_p5 = r.has("p5")
        p5 = r.number("p5", gt=0.0) if measured_p5 else p5_f * mean
        p95 = r.number("p95", gt=0.0) if r.has("p95") else p95_f * mean
        step = r.number("longest_step", gt=0.0) if r.has("longest_step") else step_f * mean
    if not p5 <= mean <= p95:
        raise ConfigError(key, f"needs p5 <= mean <= p95, got {p5}, {mean}, {p95}")
    return TimeRow(mean=mean, p5=p5, p95=p95, longest_step=step, p5_measured=measured_p5)


def _e_strong(specs: Mapping[str, _TaskSpec], table: TimeTable, strongest: str) -> Mapping[str, float]:
    """final-design 7.1: mean run on the strongest fog class, on its fastest slot type."""
    out: dict[str, float] = {}
    for task in specs:
        if not table.slot_types(task, strongest):
            raise ConfigError(f"time_table_placeholder.{task}.{strongest}", "missing (the strongest class)")
        out[task] = table.fastest(task, strongest)[1].mean
    return _frozen(out)


def _energy(n: _Node, hardware: Mapping[str, HardwareClass]) -> Energy:
    with n:
        n.marker("e_bar", "computed")  # final-design 12.3: derived
        return Energy(
            pue_fog=n.number("pue_fog", ge=1.0),
            pue_cloud=n.number("pue_cloud", ge=1.0),
            pue_network=n.number("pue_network", ge=1.0),
            e_bh_device=n.number("e_bh_nj_per_bit", ge=0.0) * _J_PER_NJ,
            e_wan_device=n.number("e_wan_nj_per_bit", ge=0.0) * _J_PER_NJ,
            e_down_device=n.number("e_down_nj_per_bit", ge=0.0) * _J_PER_NJ,
            car_tx_w=n.number("car_tx_w", ge=0.0),
            reference_class=_fog_class(n, "reference_class", hardware),
        )


def _stateplane(n: _Node) -> StatePlane:
    with n:
        return StatePlane(
            state_period=n.number("state_period", gt=0.0),
            stale_after=n.number("stale_after", gt=0.0),
            wan_ewma_lambda=n.number("wan_ewma_lambda", gt=0.0, le=1.0),
        )


def _scheduler(n: _Node) -> Scheduler:
    with n:
        with n.node("probe_timeout") as probe:
            probe_fog, probe_cloud = probe.number("fog", gt=0.0), probe.number("cloud", gt=0.0)
        with n.node("retries") as retries:
            retries_hard, retries_soft = retries.integer("hard"), retries.integer("soft")
        return Scheduler(
            reserve_hard_r=n.integer("reserve_hard_r"),
            headroom_h=n.integer("headroom_h"),
            soft_promotion_a_max=n.number("soft_promotion_A_max", gt=0.0),
            probe_timeout_fog=probe_fog,
            probe_timeout_cloud=probe_cloud,
            t_keep=n.number("T_keep", gt=0.0),
            retries_hard=retries_hard,
            retries_soft=retries_soft,
        )


def _timing(n: _Node) -> Timing:
    with n:
        with n.node("m_wd") as m:
            m_fog, m_cloud = m.number("fog", ge=0.0), m.number("cloud", ge=0.0)
        with n.node("quantiles") as qn:
            low, high = qn.number("low", gt=0.0), qn.number("high", gt=0.0)
            if not low < high < 1.0:
                raise ConfigError(qn.path, f"needs 0 < low < high < 1, got {low}, {high}")
        return Timing(
            kappa_up=n.number("kappa_up", gt=0.0),
            m_wd_fog=m_fog,
            m_wd_cloud=m_cloud,
            quantile_low=low,
            quantile_high=high,
            t_ack=n.number("T_ack", gt=0.0),
            t_leave=n.number("T_leave", gt=0.0),
            gps_expiry=n.number("gps_expiry", gt=0.0),
            reward_grace_g=n.number("reward_grace_G", gt=0.0),
            encode=n.number("encode", ge=0.0),
        )


def _tof(n: _Node, hardware: Mapping[str, HardwareClass]) -> Tof:
    with n:
        return Tof(
            alpha=n.number("alpha", gt=0.0),
            alpha_sweep=n.numbers("alpha_sweep", gt=0.0),
            strongest_class=_fog_class(n, "strongest_class", hardware),
        )


def _reward(n: _Node) -> Reward:
    with n:
        return Reward(
            b=n.number("B", gt=0.0),
            p=n.number("P", ge=0.0),
            w3=n.number("w3", ge=0.0),
            c_drop=n.number("C_drop", gt=0.0),
            w=n.number("w", gt=0.0, le=1.0),  # 11.4: w in (0, 1]; STAND-IN until M5 (QUESTIONS Q32)
        )


def _rl(n: _Node) -> Rl:
    with n:
        eps = n.numbers("epsilon_baselines", ge=0.0)
        if len(eps) != 3 or not (eps[0] <= 1.0 and eps[1] <= eps[0] and 0.0 < eps[2] <= 1.0):
            raise ConfigError(
                n.key("epsilon_baselines"),
                f"must be [start, end, fraction] with end <= start <= 1, got {eps}",
            )
        hidden = n.numbers("hidden", gt=0.0)
        if not all(h.is_integer() for h in hidden):
            raise ConfigError(n.key("hidden"), f"must be whole layer widths, got {list(hidden)}")
        with n.node("ppo") as p:
            ppo = Ppo(
                clip=p.number("clip", gt=0.0),
                lr=p.number("lr", gt=0.0),
                gae_lambda=_fraction(p, "gae_lambda"),
                epochs=p.integer("epochs", ge=1),
                minibatch=p.integer("minibatch", ge=1),
                rollout=p.integer("rollout", ge=1),
                vf_coef=p.number("vf_coef", ge=0.0),
                ent_coef=p.number("ent_coef", ge=0.0),
            )
        return Rl(
            gamma_per_tau=n.number("gamma_per_tau", ge=0.0, le=1.0),
            tau=n.number("tau", gt=0.0),
            hidden=tuple(int(h) for h in hidden),
            dueling=n.flag("dueling"),
            double=n.flag("double"),
            optimizer=cast(Literal["adam"], n.text("optimizer", ("adam",))),
            lr=n.number("lr", gt=0.0),
            batch=n.integer("batch", ge=1),
            replay_agent=n.integer("replay_agent", ge=1),
            target_update_every=n.integer("target_update_every", ge=1),
            updates_per_transition=n.number("updates_per_transition", gt=0.0),
            epsilon_online=_fraction(n, "epsilon_online"),
            epsilon_baselines=EpsilonSchedule(start=eps[0], end=eps[1], fraction=eps[2]),
            loss=cast(Literal["mse", "huber"], n.text("loss", ("mse", "huber"))),
            lstm_history_h=n.integer("lstm_history_H", ge=1),
            ppo=ppo,
        )


def _dqfd(n: _Node) -> Dqfd:
    with n:
        with n.node("per") as p:
            per = Per(
                alpha=_fraction(p, "alpha"),
                beta0=_fraction(p, "beta0"),
                beta_final=_fraction(p, "beta_final"),
                eps_demo=p.number("eps_demo", ge=0.0),
                eps_agent=p.number("eps_agent", ge=0.0),
            )
        return Dqfd(
            n_step=n.integer("n_step", ge=1),
            lambda_n=n.number("lambda_n", ge=0.0),
            lambda_margin=n.number("lambda_margin", ge=0.0),
            lambda_l2=n.number("lambda_l2", ge=0.0),
            margin=n.number("margin", ge=0.0),
            per=per,
            pretrain_updates=n.integer("pretrain_updates", ge=1),
            eval_every=n.integer("eval_every", ge=1),
            patience=n.integer("patience", ge=1),
            holdout_share=n.number("holdout_share", gt=0.0, le=0.5),
        )


def _nsga2(n: _Node) -> Nsga2:
    with n:
        n.marker("mutation_per_gene", "inverse_n")  # final-design 13.2: 1 / number of genes
        return Nsga2(
            population=n.integer("population", ge=2),
            generations=n.integer("generations", ge=1),
            crossover_prob=_fraction(n, "crossover_prob"),
            t_win=n.number("T_win", gt=0.0),
            t_warmup=n.number("T_warmup", ge=0.0),
            windows_per_scenario=n.integer("windows_per_scenario", ge=1),
            expert_filter_margin=n.number("expert_filter_margin", ge=0.0),
        )


_SEED_RANGE = re.compile(r"^(\d+)-(\d+)$")


def _fast_mode(n: _Node) -> FastMode:
    with n:
        split: dict[str, range] = {}
        with n.node("seed_split") as s:
            for purpose in s.names():
                text = s.text(purpose)
                match = _SEED_RANGE.match(text)
                if not match or int(match.group(1)) > int(match.group(2)):
                    raise ConfigError(
                        s.key(purpose), f"must be 'first-last' with first <= last, got {text!r}"
                    )
                seeds = range(int(match.group(1)), int(match.group(2)) + 1)  # both ends included
                for other, taken in split.items():
                    if seeds.start < taken.stop and taken.start < seeds.stop:
                        raise ConfigError(s.key(purpose), f"overlaps {other} (Maus et al.: disjoint ranges)")
                split[purpose] = seeds
        return FastMode(
            episode_length=n.number("episode_length", gt=0.0),
            eval_seeds=n.integer("eval_seeds", ge=1),
            testbed_seeds=n.integer("testbed_seeds", ge=1),
            train_decisions=n.integer("train_decisions", ge=1),
            early_decisions=n.integer("early_decisions", ge=1),
            seed_split=_frozen(split),
        )


def _scenarios(n: _Node, layout: Layout, hardware: Mapping[str, HardwareClass]) -> Mapping[str, Scenario]:
    with n:
        n.marker("arrival_rate", "calibrated")  # final-design 18: lambda_car is derived
        out: dict[str, Scenario] = {}
        for name in n.names():
            if name == "arrival_rate":
                continue
            with n.node(name) as s:
                out[name] = Scenario(
                    name=name,
                    rho_fog=s.number("rho_fog", gt=0.0),
                    deadlines=cast(DeadlineMode, s.text("deadlines", DEADLINE_MODES)),
                    mobility=cast(Mobility, s.text("mobility", ("low", "high"))),
                    fog=_scenario_fog(s, layout, hardware),
                )
        if _WAN_CALIBRATION_SCENARIO not in out:
            raise ConfigError(n.key(_WAN_CALIBRATION_SCENARIO), "missing (B_WAN is set at this scenario, 18)")
        return _frozen(out)


def _scenario_fog(s: _Node, layout: Layout, hardware: Mapping[str, HardwareClass]) -> Mapping[str, str]:
    """`[class]` = that class in every zone; a mapping gives each zone's class."""
    value = s.raw("fog")
    key = s.key("fog")
    if isinstance(value, list):
        if len(value) != 1:
            raise ConfigError(key, "a list must hold one class (every zone); use a mapping zone -> class")
        holder = _Node({"fog": value[0]}, s.path)
        return _frozen(dict.fromkeys(layout.zones, _fog_class(holder, "fog", hardware)))
    with _Node(value, key) as zones:
        classes = {z: _fog_class(zones, z, hardware) for z in layout.zones}
    return _frozen(classes)


def _fedavg(n: _Node) -> FedAvg:
    with n:
        with n.node("T_fl") as t:
            fast = t.number("fast_mode", gt=0.0)
            testbed = t.number("testbed", gt=0.0)
            real_world = t.number("real_world", gt=0.0)
        return FedAvg(
            t_fl_fast_mode=fast,
            t_fl_testbed=testbed,
            t_fl_real_world=real_world,
            upload_deadline=n.number("upload_deadline", gt=0.0),
            finetune_k=n.integer("finetune_K", ge=1),
        )


# --- Derived values (tag C in final-design 18) --------------------------------------------------------------


def _derive(
    layout: Layout,
    types: Mapping[str, TaskType],
    hardware: Mapping[str, HardwareClass],
    table: TimeTable,
    energy: Energy,
    tof: Tof,
    scenarios: Mapping[str, Scenario],
    e_strong: Mapping[str, float],
) -> Derived:
    ref = hardware[energy.reference_class]
    if ref.c_gpu == 0:
        raise ConfigError("energy.reference_class", f"{ref.name} has no GPU slot (rho_fog needs one, 16.2)")
    mean_gpu = _mean_gpu_run(types, table, ref.name)
    capacity = ref.c_gpu / mean_gpu
    zones = len(layout.active_zones)

    # final-design 18 (Workload): lambda per zone = rho_fog x c_GPU / mean GPU run; the same mix in every
    # scenario, boulders included, so S4-S6 keep S2's rate (16.2, QUESTIONS Q1).
    per_zone = {s.name: s.rho_fog * capacity for s in scenarios.values()}

    def eligible_bits(mode: DeadlineMode) -> float:
        """Mean cloud-eligible input bits per task: soft tasks and boulders (16.2)."""
        return sum(
            t.share * t.input_bits
            for t in types.values()
            if t.cls == "soft" or _is_boulder(e_strong[t.name], tof.alpha, t.deadline[mode])
        )

    # final-design 18 (Links): cloud-eligible bits/s at S2 / 0.9. The 0.9 is S2's rho_fog: one B_WAN gives
    # TI_WAN = rho_fog in S1-S3 (16.2).
    cal = scenarios[_WAN_CALIBRATION_SCENARIO]
    b_wan = per_zone[cal.name] * zones * eligible_bits(cal.deadlines) / cal.rho_fog
    ti_wan = {
        s.name: per_zone[s.name] * zones * eligible_bits(s.deadlines) / b_wan for s in scenarios.values()
    }

    e_ref = {name: _e_ref(t, table, ref, energy) for name, t in types.items()}
    return Derived(
        mean_gpu_run=mean_gpu,
        zone_capacity=capacity,
        e_strong=e_strong,
        b_wan=b_wan,
        arrival_rate_per_zone=_frozen(per_zone),
        ti_wan=_frozen(ti_wan),
        e_ref=_frozen(e_ref),
        e_bar=_e_bar(types, table, hardware, energy, e_ref),
    )


def _mean_gpu_run(types: Mapping[str, TaskType], table: TimeTable, ref: str) -> float:
    """final-design 16.2: the mix's mean GPU run on the reference class (F-GPU)."""
    total = 0.0
    for name, t in types.items():
        row = table.row(name, ref, "gpu")
        if row is None:
            raise ConfigError(
                f"time_table_placeholder.{name}.{ref}.gpu", "missing (needed for rho_fog, 16.2)"
            )
        total += t.share * row.mean
    return total


def _e_ret(task: TaskType, executor: HardwareClass, energy: Energy) -> float:
    """final-design 12.2 (Return): fog executor -> broker -> serving zone = 2 backhaul crossings; cloud ->
    broker = 1 WAN + 1 backhaul; then the radio downlink. One delivery, no republish."""
    network = energy.e_wan + energy.e_bh if executor.is_cloud else 2 * energy.e_bh
    return (network + energy.e_down) * task.result_bits


def _e_ref(task: TaskType, table: TimeTable, ref: HardwareClass, energy: Energy) -> float:
    """final-design 12.3: PUE_fog x p x T on the reference class's fastest slot type + E_ret here.

    p = dynamic energy per run / run time; with placeholder rows it is the slot's dynamic power (18).
    """
    q_ref, row = table.fastest(task.name, ref.name)
    return energy.pue_fog * ref.p_dyn_w[q_ref] * row.mean + _e_ret(task, ref, energy)


def _e_bar(
    types: Mapping[str, TaskType],
    table: TimeTable,
    hardware: Mapping[str, HardwareClass],
    energy: Energy,
    e_ref: Mapping[str, float],
) -> float:
    """final-design 12.3: the largest E_k / E_ref over every task type and option, with p95 run times.

    Options: every fog class as here or a neighbour, and the cloud, on every slot type with a row.
    """
    worst = 0.0
    for name, t in types.items():
        for hw in hardware.values():
            # 12.2 (Forwarding): a neighbour pays e_bh x S, more than here (0), so it is the fog worst case.
            fwd = (energy.e_bh + energy.e_wan) * t.input_bits if hw.is_cloud else energy.e_bh * t.input_bits
            for q in table.slot_types(name, hw.name):
                row = table.rows[(name, hw.name, q)]
                e_k = fwd + energy.pue(hw) * hw.p_dyn_w[q] * row.p95 + _e_ret(t, hw, energy)
                worst = max(worst, e_k / e_ref[name])
    return worst
