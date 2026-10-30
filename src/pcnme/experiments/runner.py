"""The scenario runner: train a method, then evaluate it on evaluation seeds (final-design 16.1, 16.3; PLAN
M7). THIS TOUCHES THE AGENT: every learning method runs in the agent's seat (methods/learners.py).

One run = one scenario and one method over a list of evaluation seeds:

1. A learning method trains first, on training seeds from 2000 (16.3), on the shared budget in decisions
   (fast_mode.train_decisions, QUESTIONS Q41). PCNME pretrains on the knee demonstrations given (files from
   tools/make_demos.py); D3QN + greedy expert on greedy demonstrations it makes from the scenario's first
   `windows` NSGA-II windows (seeds 1000-1999). VFC PPO runs with w = 1 (delay only, Q43).
2. Every evaluation seed is its own episode: the trained weights deployed in every zone, learning online
   (Q42); a rule as it is; the knee reference with its plan for that seed's window (Q45).
3. Each episode writes results/<run_id>/ (METRICS.md) with run.json holding the method, the seed's usage and
   the training budget; a learning method also writes <scenario>-<method>-train/ (training.json, model.pt).

Seeds: evaluation 3000-3999 only, training 2000-2999 only; a seed outside its range is refused before
anything runs (16.3, Maus et al.).

The knee reference (16.1: "the offline knee plan replayed on the same windows and seeds, fast mode only";
QUESTIONS Q45 stand-in): NSGA-II plans the window of the evaluation seed itself (T_warmup to T_warmup + T_win
of the episode), and the episode replays its knee plan there, greedy everywhere else (13.2.3). It plans on an
evaluation seed but trains nothing: a clairvoyant reference line, not a competitor. window.json lists the
window tasks and the knee's objectives, so the report can put every method on the same window tasks.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pcnme.config import CLOUD, Config, config_from_mapping
from pcnme.experiments.city import City
from pcnme.methods.learners import D3QN_FAMILY, LEARNING, LSTM_FAMILY, METHODS, RULES, Demos, Trained
from pcnme.sim.rng import RandomStreams, SeedError
from pcnme.zone.zone import Placement

Factory = Callable[[Config, RandomStreams, str], Placement]
_SEEDS = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$")


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One scenario and method over evaluation seeds. None: the config's value."""

    scenario: str
    method: str
    seeds: tuple[int, ...]
    seconds: float | None = None  # an evaluation episode (fast_mode.episode_length)
    n_cars: int | None = None  # the fleet (cars.count's smallest)
    budget: int | None = None  # training decisions (fast_mode.train_decisions)
    train_seconds: float | None = None  # a training episode (fast_mode.episode_length)
    train_seed: int | None = None  # the first training seed (the range's first)
    demos: tuple[Path, ...] = ()  # pcnme: knee demonstrations (tools/make_demos.py)
    windows: int | None = None  # d3qn-greedy: windows for its demonstrations (nsga2.windows_per_scenario)
    pretrain_updates: int | None = None  # dqfd.pretrain_updates
    nsga_population: int | None = None  # knee-reference (nsga2.population)
    nsga_generations: int | None = None  # knee-reference (nsga2.generations)
    workers: int = 1  # knee-reference: NSGA-II evaluations in parallel
    teacher: str = "knee"  # pcnme: "random" = the 16.4 random-plan expert, made from `windows` windows
    overrides: tuple[tuple[str, Any], ...] = ()  # (dotted config key, value): ablations and sweeps (16.4)
    variant: str = ""  # the ablation's or sweep point's label: runs go to out/<variant>/ (empty: the design)


@dataclass
class RunResult:
    runs: list[Path] = field(default_factory=list)  # results/<run_id>/, one per evaluation seed
    trained: Trained | None = None


def parse_seeds(text: str) -> tuple[int, ...]:
    """ "3000-3004" or "3000,3002,3010-3012": every seed once, in the order given."""
    out: list[int] = []
    for part in text.split(","):
        m = _SEEDS.match(part)
        if m is None:
            raise ValueError(f"not a seed or seed range: {part!r}")
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) is not None else lo
        if hi < lo:
            raise ValueError(f"empty seed range {part.strip()!r}")
        out += range(lo, hi + 1)
    if len(set(out)) != len(out):
        raise ValueError(f"a seed is listed twice in {text!r}")
    return tuple(out)


def check_seeds(cfg: Config, seeds: Sequence[int], usage: str) -> None:
    """16.3: every seed in its usage's range (fast_mode.seed_split); none refused silently."""
    allowed = cfg.fast_mode.seed_split[usage]
    if not seeds:
        raise SeedError(f"no {usage} seeds given")
    bad = [s for s in seeds if s not in allowed]
    if bad:
        listed = ", ".join(map(str, bad))
        raise SeedError(f"seeds {listed} are outside the {usage} range {allowed.start}-{allowed.stop - 1}")


def method_config(cfg: Config, method: str) -> Config:
    """The config a method runs with: VFC PPO optimises delay only, w = 1 (16.1; Q43)."""
    from pcnme.expert.demos import with_w

    return with_w(cfg, 1.0) if method == "vfc-ppo" else cfg


def budget_of(cfg: Config, spec: RunSpec) -> int:
    """16.1: the same training budget in decisions for every learning method (Q41)."""
    return cfg.fast_mode.train_decisions if spec.budget is None else spec.budget


def _check(cfg: Config, spec: RunSpec) -> None:
    if spec.method not in METHODS:
        raise ValueError(f"unknown method {spec.method!r}; known: {', '.join(METHODS)}")
    if spec.teacher not in TEACHERS or (spec.teacher != "knee" and spec.method != "pcnme"):
        raise ValueError(f"teacher {spec.teacher!r}: pcnme only, one of {', '.join(TEACHERS)}")
    if not cfg.ablation.online and spec.method not in ("pcnme", "d3qn-greedy"):
        raise ValueError(f"pretraining only: {spec.method} does not pretrain (pcnme, d3qn-greedy do)")
    if (spec.overrides or spec.teacher != "knee") and not spec.variant:
        raise ValueError(
            "an ablation or a sweep point needs a variant label (its runs would mix with the design's)"
        )
    if spec.scenario not in cfg.scenarios:
        raise ValueError(f"unknown scenario {spec.scenario!r}; known: {', '.join(cfg.scenarios)}")
    check_seeds(cfg, spec.seeds, "evaluation")
    if spec.method in LEARNING and spec.train_seed is not None:
        check_seeds(cfg, (spec.train_seed,), "training")
    if spec.method == "knee-reference":
        # Q45: only these scenarios and the first evaluation seeds (one NSGA-II run per seed)
        nsga = cfg.nsga2
        if spec.scenario not in nsga.knee_scenarios:
            raise ValueError(
                f"the knee reference runs on {', '.join(nsga.knee_scenarios)} only (nsga2.knee_reference)"
            )
        first = cfg.fast_mode.seed_split["evaluation"].start
        allowed = range(first, first + nsga.knee_seeds)
        if any(s not in allowed for s in spec.seeds):
            raise ValueError(
                f"the knee reference runs on seeds {first}-{allowed.stop - 1} only (nsga2.knee_reference)"
            )
        seconds = cfg.fast_mode.episode_length if spec.seconds is None else spec.seconds
        end = cfg.nsga2.t_warmup + cfg.nsga2.t_win
        if seconds < end:
            raise ValueError(f"the episode ({seconds} s) must hold the whole NSGA-II window (to {end} s)")


# --- training -----------------------------------------------------------------------------------------------


def plan_demos(cfg: Config, scenario: str, windows: int, n_cars: int | None, teacher: str) -> Demos:
    """A rule's plan of each of the scenario's first `windows` windows (NSGA-II and demonstrations seeds),
    replayed as demonstrations (13.4) with w*: greedy (D3QN + greedy expert's teacher, 16.1) or a random
    valid plan (16.4 random-plan expert; QUESTIONS Q49 for its w)."""
    from pcnme.expert.demos import greedy_demonstrations, random_plan, replay
    from pcnme.expert.window import WindowSpec, discover, window_seed

    out = []
    for k in range(windows):
        w = discover(cfg, WindowSpec(scenario, window_seed(cfg, scenario, k), k, n_cars))
        demo = (
            greedy_demonstrations(cfg, w)
            if teacher == "greedy"
            else replay(cfg, w, random_plan(cfg, w), cfg.reward.w)
        )
        out += [((scenario, k), t.sample) for t in demo.transitions]
    return out


def train(cfg: Config, spec: RunSpec) -> Trained | None:
    """A learning method trained on the training seeds; None for a rule or the knee reference."""
    if spec.method not in LEARNING:
        return None
    zones = cfg.layout.active_zones
    budget = budget_of(cfg, spec)
    common: dict[str, Any] = {"seconds": spec.train_seconds, "seed": spec.train_seed, "n_cars": spec.n_cars}
    if spec.method in D3QN_FAMILY:
        from pcnme.expert.demos import load_samples
        from pcnme.methods.learners import train_d3qn

        demos: Demos = ()
        k = cfg.nsga2.windows_per_scenario if spec.windows is None else spec.windows
        if spec.method == "pcnme" and spec.teacher == "random":
            demos = plan_demos(cfg, spec.scenario, k, spec.n_cars, "random")
        elif spec.method == "pcnme":
            if not spec.demos:
                raise ValueError("pcnme needs its knee demonstrations (--demos, from tools/make_demos.py)")
            demos = load_samples(spec.demos, cfg.nsga2.expert_filter_margin)  # 16.4 expert filter
        elif spec.method == "d3qn-greedy":
            demos = plan_demos(cfg, spec.scenario, k, spec.n_cars, "greedy")
        return train_d3qn(
            cfg, spec.method, spec.scenario, zones, budget, demos=demos,
            pretrain_updates=spec.pretrain_updates, **common,
        )  # fmt: skip
    if spec.method in LSTM_FAMILY:
        from pcnme.methods.lstm import train_lstm

        return train_lstm(cfg, spec.method, spec.scenario, zones, budget, **common)
    from pcnme.methods.ppo import train_ppo

    return train_ppo(cfg, spec.scenario, zones, budget, **common)


def deployed(cfg: Config, trained: Trained, seed: int) -> Factory:
    """The trained method in every zone for one evaluation episode (Q42)."""
    zones = cfg.layout.active_zones
    agents: Mapping[str, Placement]
    if trained.method in D3QN_FAMILY:
        from pcnme.methods.learners import deployed_d3qn

        agents = deployed_d3qn(cfg, trained, zones, seed)
    elif trained.method in LSTM_FAMILY:
        from pcnme.methods.lstm import deployed_lstm

        agents = deployed_lstm(cfg, trained, zones, seed)
    else:
        from pcnme.methods.ppo import deployed_ppo

        agents = deployed_ppo(cfg, trained, zones, seed)
    return lambda c, s, z: agents[z]


def training_info(trained: Trained | None) -> dict[str, Any] | None:
    if trained is None:
        return None
    return {
        "method": trained.method,
        "budget": {"total": trained.budget.total, "used": trained.budget.used},
        "episodes": [
            {
                "seed": e.seed,
                "on_time": e.on_time,
                "mean_e_hat": e.mean_e_hat,
                "tasks": e.tasks,
                "updates": e.updates,
                "mean_loss": e.mean_loss,
            }
            for e in trained.episodes
        ],
        "expert_transitions": len(trained.expert),
        "agreement": trained.agreement,
        "epsilon_schedule": None if trained.schedule is None else repr(trained.schedule),
    }


def save_training(out: Path, spec: RunSpec, trained: Trained) -> Path:
    import torch

    d = out / f"{spec.scenario}-{spec.method}-train"
    d.mkdir(parents=True, exist_ok=True)
    (d / "training.json").write_text(json.dumps(training_info(trained), indent=1), encoding="utf-8")
    torch.save(trained.states, d / "model.pt")
    return d


# --- the knee reference (Q45) -------------------------------------------------------------------------------


@dataclass
class KneeReference:
    """The knee plan of one evaluation seed's window and the zones' placements that replay it."""

    plan: dict[str, Any]
    genes: dict[tuple[str, Any], str]
    placements: dict[str, Any] = field(default_factory=dict)

    def factory(self, c: Config, streams: RandomStreams, z: str) -> Placement:
        from pcnme.expert.window import PlanPlacement

        p = PlanPlacement(c, z, {tid: o for (a, tid), o in self.genes.items() if a == z})  # type: ignore[misc]
        self.placements[z] = p
        return p


def knee_reference(source: Mapping[str, object], cfg: Config, spec: RunSpec, seed: int) -> KneeReference:
    """NSGA-II on the window of this evaluation seed (Q45) and its knee plan; a front without a knee (fewer
    than 2 distinct points) falls back to its first plan."""
    from pcnme.expert.knee import knee
    from pcnme.expert.nsga import run_nsga
    from pcnme.expert.window import WindowSpec, discover

    window = discover(cfg, WindowSpec(spec.scenario, seed, 0, spec.n_cars, usage="evaluation"))
    front = run_nsga(
        source, window, population=spec.nsga_population, generations=spec.nsga_generations,
        workers=spec.workers,
    )  # fmt: skip
    k = knee(front.F)
    i = front.F.index(k.point) if k is not None else 0
    plan = {
        "usage": "evaluation",
        "seed": seed,
        "f1": front.F[i][0],
        "f2": front.F[i][1],
        "w": None if k is None else k.w,
        "is_knee": k is not None,
        "front": [list(p) for p in front.F],
        "evaluations": front.evaluations,
        "tasks": [[t.anchor, str(t.task_id)] for t in window.tasks],
        "genome": list(front.genomes[i]),
    }
    return KneeReference(plan, window.genes(front.genomes[i]))  # type: ignore[arg-type]


def _window_check(cfg: Config, city: City, ref: KneeReference) -> dict[str, Any]:
    """The replayed window's objectives (13.2.4), which equal the planned point: the episode up to the
    horizon is the plan's evaluation."""
    from pcnme.expert.window import objectives

    recs = [city.zones[a].records[tid] for a, tid in ref.genes]
    obj = objectives(cfg, recs)
    applied = sum(len(p.applied) for p in ref.placements.values())
    return {**ref.plan, "replayed": [obj.f1, obj.f2], "applied": applied}


def placement(
    source: Mapping[str, object], cfg: Config, spec: RunSpec, trained: Trained | None, seed: int
) -> tuple[str | Factory, KneeReference | None]:
    """What runs in every zone for one evaluation seed: a rule by name, the trained method deployed (Q42),
    or the knee reference with its plan for this seed's window (Q45)."""
    if spec.method in RULES:
        return spec.method, None
    if spec.method == "knee-reference":
        ref = knee_reference(source, cfg, spec, seed)
        return ref.factory, ref
    if trained is None:
        raise ValueError(f"{spec.method} must be trained before it is deployed")
    return deployed(cfg, trained, seed), None


def report_constants(cfg: Config, scenario: str) -> dict[str, Any]:
    """What the report needs besides the logs (metrics/stats.py): the idle power of the scenario's nodes for
    the total energy (12: every active fog node and one cloud server, Q46) and the 13.2.4 cost constants for
    the window objectives."""
    hw = cfg.hardware
    fog = cfg.scenarios[scenario].fog
    return {
        "idle_w": sum(hw[fog[z]].p_idle_w for z in cfg.layout.active_zones) + hw[CLOUD].p_idle_w,
        "objective": {
            "w3": cfg.reward.w3,
            "c_drop": cfg.reward.c_drop,
            "e_bar": cfg.derived.e_bar,
            "e_ref": dict(cfg.derived.e_ref),
        },
    }


# --- ablations and sweeps (16.4) ------------------------------------------------------------------------

ABLATIONS: dict[str, tuple[tuple[str, Any], ...]] = {
    "gamma-0": (("rl.gamma_per_tau", 0.0),),  # placement as a bandit problem
    "no-n-step": (("dqfd.lambda_n", 0.0),),
    "no-margin": (("dqfd.lambda_margin", 0.0),),
    "no-neighbours": (("ablation.neighbours", False),),
    "no-predictor": (("ablation.predictor", False),),
    "no-forwarding": (("ablation.result_forwarding", False),),
    "no-tof": (("ablation.tof", False),),
    "r3": (("stateplane.r3.enabled", True),),
    "expert-fade": (("ablation.expert", "fade"),),  # shrinking share + fading margin (Fading-BC)
    "expert-clear": (("ablation.expert", "clear"),),
    "pretrain-only": (("ablation.online", False),),  # Only-BC
}
TEACHERS = ("knee", "random")  # pcnme's teacher: its knee demonstrations, or a random valid plan (16.4)
SENSITIVITY = ("arrival_rate", "deadline", "input_size", "cars", "fog_capacity", "wan_bandwidth")
FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5)  # 16.4: 50 % to 150 % of the default


def with_overrides(source: Mapping[str, object], overrides: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """A copy of the config data with each dotted key set; a key that is not in the config is refused."""
    out: dict[str, Any] = copy.deepcopy(dict(source))
    for dotted, value in overrides:
        *head, last = dotted.split(".")
        node: Any = out
        for part in head:
            if not isinstance(node, dict) or part not in node:
                raise ValueError(f"unknown config key {dotted!r}")
            node = node[part]
        if not isinstance(node, dict) or last not in node:
            raise ValueError(f"unknown config key {dotted!r}")
        node[last] = value
    return out


def sweep_points(source: Mapping[str, object], name: str) -> list[tuple[str, tuple[tuple[str, Any], ...]]]:
    """The (variant, overrides) points of a 16.4 sweep: state-period (10 / 100 / 500 ms, R3 off and on;
    stale_after stays 3 state periods), alpha (tof.alpha_sweep), or a sensitivity parameter (FACTORS)."""
    if name == "state-period":
        return [
            (
                f"state-{round(p * 1000)}ms{'-r3' if r3 else ''}",
                (
                    ("stateplane.state_period", p),
                    ("stateplane.stale_after", 3 * p),
                    ("stateplane.r3.enabled", r3),
                ),
            )
            for p in (0.010, 0.100, 0.500)
            for r3 in (False, True)
        ]
    if name == "alpha":
        alphas: Sequence[float] = source["tof"]["alpha_sweep"]  # type: ignore[index]
        return [(f"alpha-{a}", (("tof.alpha", a),)) for a in alphas]
    if name in SENSITIVITY:
        return [(f"{name}-x{f}", ((f"sweep.{name}", f),)) for f in FACTORS]
    raise ValueError(f"unknown sweep {name!r}; known: state-period, alpha, {', '.join(SENSITIVITY)}")


def w_points(ws: Sequence[float]) -> list[tuple[str, tuple[tuple[str, Any], ...]]]:
    """16.4 w sweep: the w-conditioned agent run at the front's w_p values (input 48 and the reward)."""
    return [(f"w-{w:g}", (("reward.w", w),)) for w in ws]


# --- running ------------------------------------------------------------------------------------------------


def run(source: Mapping[str, object], spec: RunSpec, out: Path) -> RunResult:
    """Train (a learning method), then one episode per evaluation seed; each writes out/<run_id>/, or
    out/<variant>/<run_id>/ for an ablation or a sweep point (16.4)."""
    source = with_overrides(source, spec.overrides)
    out = out / spec.variant if spec.variant else out
    base = config_from_mapping(source)
    _check(base, spec)
    cfg = method_config(base, spec.method)
    seconds = cfg.fast_mode.episode_length if spec.seconds is None else spec.seconds
    result = RunResult(trained=train(cfg, spec))
    out.mkdir(parents=True, exist_ok=True)
    if result.trained is not None:
        save_training(out, spec, result.trained)
    tinfo = training_info(result.trained)
    for seed in spec.seeds:
        extra: dict[str, Any] = {
            "method": spec.method,
            "usage": "evaluation",
            "reward_w": cfg.reward.w,
            "training": tinfo,
            "variant": spec.variant,
            "overrides": [list(o) for o in spec.overrides],
            "broker_messages": None,
            **report_constants(cfg, spec.scenario),
        }
        method, ref = placement(source, cfg, spec, result.trained, seed)
        city = City(
            cfg, spec.scenario, method, seed, spec.n_cars, usage="evaluation", method_name=spec.method
        ).run(seconds)
        extra["broker_messages"] = city.bus.published  # 9.2: broker load (the state-period experiment)
        if ref is not None:
            window = _window_check(cfg, city, ref)
            extra["knee"] = {k: window[k] for k in ("f1", "f2", "w", "is_knee", "replayed", "applied")}
        d = city.write(out, extra)
        if ref is not None:
            (d / "window.json").write_text(json.dumps(window, indent=1), encoding="utf-8")
        result.runs.append(d)
    return result
