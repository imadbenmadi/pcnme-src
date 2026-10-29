"""NSGA-II windows and their evaluation (final-design 13.2; PLAN M5.1). THIS TOUCHES THE AGENT: it decides
what the expert's demonstrations are made of (13.4).

A window is every pebble arriving in the whole city during T_win of one scenario, with its seed (13.2.1);
"arriving" = its upload is in (t_rx, its first decision) in [T_warmup, T_warmup + T_win). Before it, T_warmup
of greedy decisions fills the queues and pipes; tasks after it are placed by greedy (13.2.3). The set of
window tasks, their anchors and their t_rx do not depend on the decisions (arrivals, mobility and the radio
uplink are decision-free, and every draw is keyed by task, CHANGES C3), so one greedy discovery run finds
them for every plan.

Genome (13.2.2): one gene per window task, the index of its action among the anchor's options here, its
non-empty neighbour positions, cloud. Encoding only the non-empty positions is the repair "a gene pointing at
an empty position is repaired to here", done once by construction.

Evaluation (13.2.3): one fast-mode run of the window with the genes as the window tasks' first decisions, the
same seed for every plan (common random numbers). A gene whose option is masked at its decision is replaced by
the greedy choice (as at replay, 13.4.3); a re-decision or a retry inside a plan has no gene and is greedy's
(QUESTIONS Q37 stand-in). The run stops once every window task has settled (close + G, 11.3.3): the objectives
depend on nothing later.

Objectives (13.2.4), both minimised, averaged over the window's N tasks: f1 = mean c^time (L/D + w3 max(0,
L - D)/D if delivered, C_drop if dropped); f2 = mean ê (min(E/E_ref, ē) with E over all its attempts, 12.5;
ē if dropped).
"""

from __future__ import annotations

import math
import os
import pickle
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from pcnme.agent.reward import c_time
from pcnme.config import Config
from pcnme.experiments.city import City
from pcnme.ids import TaskId
from pcnme.methods.simple import Greedy
from pcnme.sim.rng import RandomStreams
from pcnme.zone.decision import OPTIONS, DecisionContext, Option
from pcnme.zone.zone import TaskRecord

STEP = 0.1  # s: the clock runs in steps of this size, checking after each whether every window task settled
USAGE = "nsga2_and_demos"  # fast_mode.seed_split: NSGA-II and demonstrations only (16.3, Maus et al.)
Key = tuple[str, TaskId]  # (anchor, task id)


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """One window: scenario, seed and its number among the scenario's windows."""

    scenario: str
    seed: int
    index: int
    n_cars: int | None = None  # the fleet; None = the config's smallest (as City)


def window_seed(cfg: Config, scenario: str, index: int) -> int:
    """The seed of window `index` of a scenario: 1000 + 100 (s - 1) + index for S<s>, so every (scenario,
    window) has its own seed inside the NSGA-II and demonstrations range (an engineering choice)."""
    s = list(cfg.scenarios).index(scenario)
    if not 0 <= index < 100:
        raise ValueError(f"window index {index} must be in 0-99")
    seed = cfg.fast_mode.seed_split[USAGE].start + 100 * s + index
    if seed not in cfg.fast_mode.seed_split[USAGE]:
        raise ValueError(f"window seed {seed} is outside the {USAGE} range")
    return seed


@dataclass(frozen=True, slots=True)
class WindowTask:
    anchor: str
    task_id: TaskId
    task_type: str
    t_rx: float
    options: tuple[Option, ...]  # here, the anchor's non-empty neighbour positions, cloud (13.2.2)

    @property
    def key(self) -> Key:
        return (self.anchor, self.task_id)


@dataclass(frozen=True, slots=True)
class Objectives:
    f1: float  # mean c^time
    f2: float  # mean ê
    n: int  # window tasks
    drops: int


@dataclass(frozen=True, slots=True)
class Window:
    """A discovered window: its tasks in (t_rx, anchor, task id) order, the greedy plan's genome and the
    greedy plan's objectives."""

    spec: WindowSpec
    tasks: tuple[WindowTask, ...]
    greedy_genome: tuple[int, ...]
    greedy: Objectives

    def genes(self, genome: Sequence[int]) -> dict[Key, Option]:
        if len(genome) != len(self.tasks):
            raise ValueError(f"a genome of {len(genome)} genes for {len(self.tasks)} window tasks")
        out = {}
        for t, g in zip(self.tasks, genome, strict=True):
            if not 0 <= g < len(t.options):
                raise ValueError(f"gene {g} out of range for {t.key}: {len(t.options)} options")
            out[t.key] = t.options[g]
        return out


@dataclass
class PlanPlacement:
    """One zone's placement under a plan: a window task's first decision takes its gene if that option is
    open, else greedy's choice (13.4.3); every other decision is greedy's (13.2.3; re-decisions: Q37)."""

    cfg: Config
    zone_id: str
    genes: Mapping[TaskId, Option]
    rule: Callable[[DecisionContext], Option | None] | None = None  # first decisions of `ruled` tasks
    ruled: frozenset[TaskId] = frozenset()
    greedy: Greedy = field(init=False)
    first: dict[TaskId, Option | None] = field(default_factory=dict)  # every task's first decision
    applied: set[TaskId] = field(default_factory=set)  # window tasks that took their gene
    replaced: set[TaskId] = field(default_factory=set)  # window tasks whose gene was masked

    def __post_init__(self) -> None:
        self.greedy = Greedy(self.cfg)

    def choose(self, ctx: DecisionContext) -> Option | None:
        tid = ctx.task.task_id
        first = tid not in self.first
        option: Option | None
        gene = self.genes.get(tid) if first else None
        if gene is not None and ctx.views[OPTIONS.index(gene)].masked is None:
            self.applied.add(tid)
            option = gene
        else:
            if gene is not None:
                self.replaced.add(tid)
            if first and self.rule is not None and tid in self.ruled:
                option = self.rule(ctx)
            else:
                option = self.greedy.choose(ctx)
        if first:
            self.first[tid] = option
        return option


@dataclass
class Run:
    """One simulated plan: the city and each zone's placement."""

    city: City
    placements: dict[str, PlanPlacement]
    settled: set[Key] = field(default_factory=set)
    t: float = 0.0  # how far the clock has run

    def records(self, keys: Iterable[Key]) -> list[TaskRecord]:
        return [self.city.zones[a].records[tid] for a, tid in keys]


def horizon(cfg: Config) -> float:
    """The latest a window task can settle: the window's end + the longest D + L_max + T_ack + G."""
    nsga, tm = cfg.nsga2, cfg.timing
    longest = max(t.deadline[m] + t.l_max[m] for t in cfg.task_types.values() for m in t.deadline)
    return nsga.t_warmup + nsga.t_win + longest + tm.t_ack + tm.reward_grace_g


def start(
    cfg: Config,
    spec: WindowSpec,
    genes: Mapping[Key, Option],
    *,
    logging: bool = False,
    rule: Callable[[DecisionContext], Option | None] | None = None,
    ruled: Iterable[Key] = (),
) -> Run:
    """Build the scenario with the plan's genes at time 0; the cars make tasks until the horizon in every
    plan."""
    placements: dict[str, PlanPlacement] = {}
    ruled_keys = tuple(ruled)

    def factory(c: Config, streams: RandomStreams, z: str) -> PlanPlacement:
        mine = frozenset(tid for a, tid in ruled_keys if a == z)
        p = PlanPlacement(c, z, {tid: o for (a, tid), o in genes.items() if a == z}, rule, mine)
        placements[z] = p
        return p

    city = City(cfg, spec.scenario, factory, spec.seed, spec.n_cars, usage=USAGE, logging=logging)
    run = Run(city, placements)
    for z, zone in city.zones.items():
        zone.on_settle = lambda rec, z=z: run.settled.add((z, rec.task.task_id))
    city.start(horizon(cfg))
    return run


def advance(run: Run, end: float, stop: Callable[[Run], bool] | None = None) -> Run:
    """Run the clock in steps of STEP until `stop` says so after a step, or to `end`. Where the steps fall
    never changes the events (the clock runs them in order)."""
    while run.t < end:
        run.t = min(run.t + STEP, end)
        run.city.clock.run_until(run.t)
        if stop is not None and stop(run):
            break
    return run


def simulate(
    cfg: Config,
    spec: WindowSpec,
    genes: Mapping[Key, Option],
    *,
    logging: bool = False,
    stop: Callable[[Run], bool] | None = None,
    rule: Callable[[DecisionContext], Option | None] | None = None,
    ruled: Iterable[Key] = (),
) -> Run:
    """Run the scenario with the plan's genes until `stop` says so after a step, or to the horizon."""
    run = start(cfg, spec, genes, logging=logging, rule=rule, ruled=ruled)
    return advance(run, horizon(cfg), stop)


def window_tasks(cfg: Config, run: Run) -> list[WindowTask]:
    """The pebbles whose upload was in during the window (13.2.1), in (t_rx, anchor, task id) order."""
    t0 = cfg.nsga2.t_warmup
    t1 = t0 + cfg.nsga2.t_win
    out = []
    for z, zone in run.city.zones.items():
        positions = zone.neighbours.positions
        options: tuple[Option, ...] = (
            "here",
            *(o for o, n in zip(OPTIONS[1:5], positions, strict=True) if n is not None),
            "cloud",
        )
        for rec in zone.records.values():
            if rec.route == "agent" and t0 <= rec.t_rx < t1:
                out.append(WindowTask(z, rec.task.task_id, rec.task.task_type, rec.t_rx, options))
    return sorted(out, key=lambda w: (w.t_rx, w.anchor, w.task_id))


def objectives(cfg: Config, records: Sequence[TaskRecord]) -> Objectives:
    """13.2.4: f1 = mean c^time, f2 = mean ê over the window tasks (a drop: C_drop and ē)."""
    r, e_bar = cfg.reward, cfg.derived.e_bar
    f1 = f2 = 0.0
    drops = 0
    for rec in records:
        task = rec.task
        if rec.outcome in ("on_time", "late"):
            assert rec.confirm is not None
            f1 += c_time(rec.confirm.latency, task.deadline, r.w3)
            energy = sum(a.energy for a in rec.rewards.values())  # every attempt (12.5)
            f2 += min(energy / cfg.derived.e_ref[task.task_type], e_bar)
        else:
            drops += 1
            f1 += r.c_drop
            f2 += e_bar
    n = len(records)
    return Objectives(f1 / n, f2 / n, n, drops) if n else Objectives(0.0, 0.0, 0, 0)


def discover(cfg: Config, spec: WindowSpec) -> Window:
    """The greedy run: the window's tasks, the greedy plan as a genome, and its objectives."""
    end_win = cfg.nsga2.t_warmup + cfg.nsga2.t_win
    found: list[WindowTask] = []

    def stop(run: Run) -> bool:
        if run.city.clock.now < end_win:
            return False
        if not found:
            found.extend(window_tasks(cfg, run))
        return all(t.key in run.settled for t in found)

    run = simulate(cfg, spec, {}, stop=stop)
    if not found:
        found.extend(window_tasks(cfg, run))
    genome = []
    for t in found:
        choice = run.placements[t.anchor].first.get(t.task_id)
        genome.append(t.options.index(choice) if choice in t.options else 0)  # all masked: any gene
    greedy = objectives(cfg, run.records(t.key for t in found))
    return Window(spec, tuple(found), tuple(genome), greedy)


def evaluate(cfg: Config, window: Window, genome: Sequence[int]) -> tuple[Objectives, Run]:
    """13.2.3: one fast-mode run of the window with the genes as decisions; stops once every window task
    has settled."""
    keys = [t.key for t in window.tasks]
    run = simulate(cfg, window.spec, window.genes(genome), stop=lambda r: all(k in r.settled for k in keys))
    if not all(k in run.settled for k in keys):
        raise RuntimeError(f"window {window.spec}: not every window task settled by the horizon")
    return objectives(cfg, run.records(keys)), run


def energy_greedy(ctx: DecisionContext) -> Option | None:
    """The energy-greedy choice (CHANGES C9): the open option with the lowest Ê (12.4), i.e. the cheapest
    one that meets the deadline; ties by the earlier finish, then option order."""
    free = [v for v in ctx.views if v.masked is None]
    if not free:
        return None
    pick = min(free, key=lambda v: (v.energy, v.finish, OPTIONS.index(v.option)))
    return pick.option  # type: ignore[return-value]


def plan_of(
    cfg: Config, window: Window, rule: Callable[[DecisionContext], Option | None]
) -> tuple[tuple[int, ...], Objectives]:
    """The genome of a rule and its objectives: the window tasks' first decisions made by `rule`, every other
    one by greedy (as in any plan); a task with every option masked gets gene 0 (unused). Evaluating the
    genome gives the same objectives."""
    keys = [t.key for t in window.tasks]
    run = simulate(
        cfg, window.spec, {}, stop=lambda r: all(k in r.settled for k in keys), rule=rule, ruled=keys
    )
    genome = []
    for t in window.tasks:
        choice = run.placements[t.anchor].first.get(t.task_id)
        genome.append(t.options.index(choice) if choice in t.options else 0)
    return tuple(genome), objectives(cfg, run.records(keys))


FORK = sys.platform.startswith("linux")  # warm-up reuse by fork: Linux and WSL only (Windows has no fork)


class WarmStart:
    """Speed-up that changes no result: the warm-up before T_warmup is the same in every plan of a window
    (greedy decisions, decision-free arrivals, keyed draws), so it is simulated once; each evaluation forks
    the warmed-up process, sets the plan's genes and runs on (copy-on-write, exact). Linux only."""

    def __init__(self, cfg: Config, window: Window) -> None:
        if not FORK:
            raise RuntimeError("WarmStart needs fork (Linux or WSL)")
        self.cfg, self.window = cfg, window
        self.keys = [t.key for t in window.tasks]
        self.run = start(cfg, window.spec, {})
        # every event strictly before T_warmup: no window task has a decision yet (its t_rx >= T_warmup)
        self.run.t = math.nextafter(cfg.nsga2.t_warmup, 0.0)
        self.run.city.clock.run_until(self.run.t)

    def _finish(self, genome: Sequence[int]) -> Objectives:
        genes = self.window.genes(genome)
        for z, p in self.run.placements.items():
            p.genes = {tid: o for (a, tid), o in genes.items() if a == z}
        keys = self.keys
        advance(self.run, horizon(self.cfg), lambda r: all(k in r.settled for k in keys))
        if not all(k in self.run.settled for k in keys):
            raise RuntimeError(f"window {self.window.spec}: not every window task settled by the horizon")
        return objectives(self.cfg, self.run.records(keys))

    def evaluate(self, genome: Sequence[int]) -> Objectives:
        """13.2.3 from the shared warm-up: the same objectives as evaluate(), exactly."""
        read, write = os.pipe()
        pid = os.fork()
        if pid == 0:  # the child: finish this plan, send the objectives, leave without cleanup
            code = 1
            try:
                os.close(read)
                with os.fdopen(write, "wb") as out:
                    pickle.dump(self._finish(genome), out)
                code = 0
            finally:
                os._exit(code)
        os.close(write)
        with os.fdopen(read, "rb") as inp:
            data = inp.read()
        _, status = os.waitpid(pid, 0)
        if status != 0 or not data:
            raise RuntimeError(f"the evaluation of a plan of {self.window.spec} failed in its fork")
        result: Objectives = pickle.loads(data)
        return result
