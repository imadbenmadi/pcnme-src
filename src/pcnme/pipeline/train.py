"""train.py: everything that is trained, per map, in order (final-design 13, 16.1):

1. traces (trace maps only): the SUMO traces per time of day if missing, then the map's coverage measured on
   them (data/<map>/calibrated.yaml, merged into that map's config).
2. NSGA-II windows per scenario (13.2), their knee points and the map's w* = the median knee weight (13.3).
3. Demonstrations: every front plan of every window (13.4; PCNME's) and greedy plans of the same windows
   (D3QN + greedy expert's, 16.1).
4. Learning: PCNME (DQfD pretraining on its demonstrations, then online, 13.5) and the learning baselines on
   the same budget and the same reward with that w* (16.1): D3QN from scratch, D3QN + greedy expert
   (pretrained on the greedy demonstrations), VFC PPO (w = 1, Q43), Tang & Wong, QECO (no demonstrations,
   as in their papers). Greedy and Random are not trained.

Everything lands in data/<map>/ (pcnme/paths.py, data/DATA.md); it resumes where it stopped (finished files
are skipped, a training continues from its last checkpoint), and refuses to mix outputs of another config or
commit (pipeline/runinfo.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pcnme.paths import REPO, MapPaths

TRAINED = ("pcnme", "d3qn-scratch", "d3qn-greedy", "vfc-ppo", "tang-wong", "qeco")  # 16.1; rules need none

# Measured on the home desktop (3 Nov): core-seconds per NSGA-II evaluation (S2, no fork) and training
# decisions per second for one job. Only for the estimate printed before a run starts.
CORE_S_PER_EVALUATION = 3.0
DECISIONS_PER_S = 110.0
TRACE_S_PER_SIMULATED_S = 0.6


@dataclass(frozen=True)
class Plan:
    """What one train.py run does on one map. None: the config's value."""

    map: str
    scenarios: tuple[str, ...]
    windows: int
    population: int | None
    generations: int | None
    budget: int | None
    train_seconds: float | None
    n_cars: int | None
    pretrain_updates: int | None
    workers: int
    checkpoint_every_s: float
    smoke: bool = False


# --- stage 1: traces ----------------------------------------------------------------------------------------


def _trace_job(map_name: str, time_of_day: str, smoke: bool, report: Any) -> dict[str, float]:
    from pcnme.maps import map_config, trace_source
    from pcnme.mobility.lust import make_trace
    from pcnme.mobility.traces import trace_file

    cfg = map_config(map_name)
    paths = MapPaths(map_name)
    out = trace_file(paths.traces, time_of_day)
    report(status="running", phase="SUMO", done=0, total=1, unit="trace")
    kw: dict[str, float] = {"trace_s": 120, "warmup_s": 120} if smoke else {}
    stats = make_trace(
        cfg,
        trace_source(cfg),
        time_of_day,
        out,
        paths.traces / "work",
        on_stage=lambda s: report(phase=s[:40]),
        **kw,
    )
    out.with_suffix(".json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    report(
        status="done", done=1, numbers={"vehicles": stats["vehicles"], "mean at once": stats["mean_at_once"]}
    )
    return stats


def stage_traces(screen: Any, plan: Plan) -> None:
    from pcnme.maps import map_config
    from pcnme.mobility.traces import trace_file
    from pcnme.pipeline.screen import run_jobs

    cfg = map_config(plan.map)
    paths = MapPaths(plan.map)
    times = sorted({cfg.map.scenario_time[s] for s in plan.scenarios})
    todo = [t for t in times if not trace_file(paths.traces, t).exists()]
    screen.stage("traces: SUMO runs per time of day", [f"trace {t}" for t in todo])
    if todo:
        res = run_jobs(
            screen, _trace_job, [(f"trace {t}", (plan.map, t, plan.smoke)) for t in todo], plan.workers
        )
        failed = [k for k, v in res.items() if isinstance(v, Exception)]
        if failed:
            raise RuntimeError(
                f"trace generation failed: {failed} (see the screen and data/{plan.map}/traces/work)"
            )
    if not paths.calibrated.exists():
        screen.say("coverage: measuring it on the traces (positions only)")
        _measure_coverage(plan, times)


def _measure_coverage(plan: Plan, times: list[str]) -> None:
    """Q29 on traces: each mobility's coverage = the mean over its scenarios' times of day."""
    import yaml

    from pcnme.experiments.calibrate import measure_trace
    from pcnme.maps import map_config
    from pcnme.mobility.traces import trace_file

    cfg = map_config(plan.map)
    paths = MapPaths(plan.map)
    cal = cfg.cars.calibration
    fleet = plan.n_cars or cfg.cars.count[0]
    seconds = 20.0 if plan.smoke else cal.seconds
    by_time = {
        t: measure_trace(cfg, trace_file(paths.traces, t), seconds, [cal.seed, cal.seed - 1], fleet)
        for t in times
    }
    coverage: dict[str, float] = {}
    for mobility in ("low", "high"):
        ts = [cfg.map.scenario_time[s] for s, sc in cfg.scenarios.items() if sc.mobility == mobility]
        vals = [by_time[t] for t in ts if t in by_time]
        coverage[mobility] = round(statistics.mean(vals), 4) if vals else cfg.cars.coverage[mobility]
    paths.calibrated.write_text(
        "# Measured by train.py on this map's traces (positions only; QUESTIONS Q29). Merged into the map's\n"
        "# config by pcnme.maps. Per time of day: "
        + json.dumps({k: round(v, 4) for k, v in by_time.items()})
        + "\n"
        + yaml.safe_dump({"cars": {"calibrated": {"coverage": coverage}}}),
        encoding="utf-8",
    )


# --- stage 2: NSGA-II, knees, w* ----------------------------------------------------------------------------


def _nsga_job(map_name: str, scenario: str, k: int, plan: Plan, report: Any) -> dict[str, Any]:
    from pcnme.config import config_from_mapping
    from pcnme.expert.knee import knee, weights
    from pcnme.expert.nsga import run_nsga
    from pcnme.expert.window import WindowSpec, discover, window_seed
    from pcnme.maps import map_source

    source = map_source(map_name)
    cfg = config_from_mapping(source)
    spec = WindowSpec(scenario, window_seed(cfg, scenario, k), k, plan.n_cars)
    report(status="running", phase="discover window", done=0, total=1, unit="window")
    window = discover(cfg, spec)
    report(phase=f"NSGA-II on {len(window.tasks)} genes")
    front = run_nsga(source, window, population=plan.population, generations=plan.generations, workers=1)
    kn, wp = knee(front.F), weights(front.F)
    data = {
        "scenario": scenario,
        "index": k,
        "seed": spec.seed,
        "population": plan.population or cfg.nsga2.population,
        "generations": plan.generations or cfg.nsga2.generations,
        "genes": len(window.tasks),
        "greedy": [window.greedy.f1, window.greedy.f2, window.greedy.drops],
        "genomes": front.genomes,
        "F": front.F,
        "w_p": [wp.get(p) for p in front.F],
        "distinct": front.distinct,
        "evaluations": front.evaluations,
        "seconds": front.seconds,
        "knee": None if kn is None else {"point": kn.point, "w": kn.w},
    }
    path = MapPaths(map_name).nsga / f"{scenario}-w{k}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    report(
        status="done", done=1, numbers={"distinct": front.distinct, "knee w": None if kn is None else kn.w}
    )
    return {k2: v for k2, v in data.items() if k2 != "genomes"}


def stage_nsga(screen: Any, plan: Plan) -> float:
    from pcnme.pipeline.screen import run_jobs

    paths = MapPaths(plan.map)
    keys = [(s, k) for s in plan.scenarios for k in range(plan.windows)]
    todo = [(s, k) for s, k in keys if not (paths.nsga / f"{s}-w{k}.json").exists()]
    screen.stage("NSGA-II windows (13.2)", [f"nsga {s} w{k}" for s, k in todo])
    res = run_jobs(
        screen, _nsga_job, [(f"nsga {s} w{k}", (plan.map, s, k, plan)) for s, k in todo], plan.workers
    )
    failed = [k for k, v in res.items() if isinstance(v, Exception)]
    if failed:
        raise RuntimeError(f"NSGA-II failed on {failed}; rerun train.py to retry them")
    return write_fronts_and_w_star(paths, keys, screen)


def write_fronts_and_w_star(paths: MapPaths, keys: list[tuple[str, int]], screen: Any) -> float:
    """fronts.csv (one row per window) and w_star.json: the median of the knee weights (13.3)."""
    rows, knees = [], []
    for s, k in keys:
        d = json.loads((paths.nsga / f"{s}-w{k}.json").read_text(encoding="utf-8"))
        kn = d["knee"]
        rows.append(
            {
                "scenario": s,
                "window": k,
                "seed": d["seed"],
                "genes": d["genes"],
                "greedy_f1": d["greedy"][0],
                "greedy_f2": d["greedy"][1],
                "greedy_drops": d["greedy"][2],
                "points": len(d["F"]),
                "distinct": d["distinct"],
                "knee_f1": kn["point"][0] if kn else "",
                "knee_f2": kn["point"][1] if kn else "",
                "knee_w": kn["w"] if kn else "",
                "evaluations": d["evaluations"],
                "seconds": round(d["seconds"], 1),
            }
        )
        if kn:
            knees.append(kn["w"])
    _write_csv(paths.nsga / "fronts.csv", rows)
    if not knees:
        raise RuntimeError("no window has a front with 2 distinct points: no knee, no w* (13.3)")
    w = statistics.median(knees)
    paths.w_star.write_text(
        json.dumps(
            {"w_star": w, "knee_weights": knees, "windows": len(keys), "with_knee": len(knees)}, indent=1
        ),
        encoding="utf-8",
    )
    screen.say(f"w* = {w:.4f} (median of {len(knees)} knee weights over {len(keys)} windows)")
    return w


# --- stage 3: demonstrations --------------------------------------------------------------------------------


def _demo_job(
    map_name: str, kind: str, scenario: str, k: int, w_star: float, plan: Plan, report: Any
) -> dict:
    from pcnme.config import config_from_mapping
    from pcnme.expert.demos import demonstrate_front, greedy_demonstrations, write
    from pcnme.expert.window import WindowSpec, discover
    from pcnme.maps import map_source

    paths = MapPaths(map_name)
    cfg = config_from_mapping(with_w_star(map_source(map_name), w_star))
    front = json.loads((paths.nsga / f"{scenario}-w{k}.json").read_text(encoding="utf-8"))
    report(status="running", phase=f"{kind}: replay", done=0, total=1, unit="window")
    window = discover(cfg, WindowSpec(scenario, front["seed"], k, plan.n_cars))
    if kind == "nsga":
        F = [tuple(p) for p in front["F"]]
        demos = demonstrate_front(cfg, window, front["genomes"], F)  # 13.4.4: every plan, its own w_p
        for d in demos:
            if (d.objectives.f1, d.objectives.f2) != F[d.plan]:
                raise RuntimeError(
                    f"{scenario} w{k} plan {d.plan}: the replay does not reproduce its objectives"
                )
    else:
        demos = [greedy_demonstrations(cfg, window)]  # 16.1: the greedy expert's plan with w*
    meta = {
        "kind": kind,
        "scenario": scenario,
        "window": k,
        "plans": len(demos),
        "transitions": sum(len(d.transitions) for d in demos),
        "expert": sum(t.expert for d in demos for t in d.transitions),
        "w": ";".join(f"{d.w:.4f}" for d in demos),
    }
    if demos:
        write(paths.demos / f"{kind}-{scenario}-w{k}.parquet", demos)
    report(status="done", done=1, numbers={"plans": len(demos), "transitions": meta["transitions"]})
    return meta


def stage_demos(screen: Any, plan: Plan, w_star: float) -> None:
    from pcnme.pipeline.screen import run_jobs

    paths = MapPaths(plan.map)
    jobs = []
    for kind in ("nsga", "greedy"):
        for s in plan.scenarios:
            for k in range(plan.windows):
                if not (paths.demos / f"{kind}-{s}-w{k}.parquet").exists() and not _no_demos(
                    paths, kind, s, k
                ):
                    jobs.append((f"demos {kind} {s} w{k}", (plan.map, kind, s, k, w_star, plan)))
    screen.stage("demonstrations (13.4; greedy for D3QN + greedy expert)", [n for n, _ in jobs])
    res = run_jobs(screen, _demo_job, jobs, plan.workers)
    failed = [k for k, v in res.items() if isinstance(v, Exception)]
    if failed:
        raise RuntimeError(f"demonstrations failed: {failed}")
    meta_path = paths.demos / "demos.csv"
    old = _read_csv(meta_path)
    new = [v for v in res.values() if isinstance(v, dict)]
    _write_csv(meta_path, sorted(old + new, key=lambda r: (r["kind"], r["scenario"], int(r["window"]))))


def _no_demos(paths: MapPaths, kind: str, scenario: str, k: int) -> bool:
    """A front with fewer than 2 distinct points gives no demonstrations (13.4.4): known from demos.csv."""
    return any(
        r["kind"] == kind and r["scenario"] == scenario and int(r["window"]) == k and int(r["plans"]) == 0
        for r in _read_csv(paths.demos / "demos.csv")
    )


# --- stage 4: learning --------------------------------------------------------------------------------------


def with_w_star(source: dict[str, Any], w: float) -> dict[str, Any]:
    """The map's config with its w*: the online reward and input 48 use it (13.3)."""
    out = json.loads(json.dumps(source))
    out["reward"]["w"] = round(w, 6)
    return out


def _train_job(map_name: str, scenario: str, method: str, w_star: float, plan: Plan, report: Any) -> str:
    from pcnme.config import config_from_mapping
    from pcnme.experiments.runner import method_config, training_info
    from pcnme.maps import map_source
    from pcnme.pipeline import checkpoint

    paths = MapPaths(map_name)
    d = paths.train(scenario, method)
    d.mkdir(parents=True, exist_ok=True)
    done, ckpt, log = d / "trained.pt", d / "checkpoint.pt", d / "train.log"
    if done.exists():
        report(status="skipped", phase="trained before", message="already trained (trained.pt): skipped")
        return "skipped"
    cfg = method_config(config_from_mapping(with_w_star(map_source(map_name), w_star)), method)
    started = time.monotonic()

    def note(text: str) -> None:
        with log.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {text}\n")

    if ckpt.exists():
        session = checkpoint.load(ckpt)
        note(f"resumed at episode {len(session.out.episodes)} (budget used {session.out.budget.used})")
        report(status="running", message=f"resumed at episode {len(session.out.episodes)}")
        _rewrite_episodes(d, session)
    else:
        note(
            f"start: {method} on {scenario}, w = {cfg.reward.w}, "
            f"budget {plan.budget or cfg.fast_mode.train_decisions}"
        )
        session = _new_session(cfg, paths, scenario, method, plan, report, note)
        checkpoint.save(session, ckpt)  # pretraining done: never redone
        _rewrite_episodes(d, session)
    last_save = time.monotonic()

    def on_episode(ep: Any) -> None:
        b = session.out.budget
        row = _episode_row(len(session.out.episodes), ep, b.used, time.monotonic() - started)
        _append_csv(d / "episodes.csv", row)
        note(
            f"episode {row['episode']}: on time {ep.on_time:.3f}, mean e_hat {ep.mean_e_hat:.3f}, "
            f"loss {ep.mean_loss:.4g}, decisions {b.used}/{b.total}"
        )
        report(
            phase="online",
            done=b.used,
            total=b.total,
            unit="decisions",
            numbers={
                "episode": row["episode"],
                "on time": round(ep.on_time, 4),
                "mean ê": round(ep.mean_e_hat, 3),
                "loss": float(f"{ep.mean_loss:.4g}"),
            },
        )

    def save(s: Any) -> None:
        nonlocal last_save
        if time.monotonic() - last_save >= plan.checkpoint_every_s:
            checkpoint.save(s, ckpt)
            last_save = time.monotonic()

    from pcnme.methods.learners import run_session

    report(
        status="running",
        phase="online",
        done=session.out.budget.used,
        total=session.out.budget.total,
        unit="decisions",
    )
    trained = run_session(cfg, session, on_episode=on_episode, checkpoint=save)
    checkpoint.save(trained, done)  # evaluate.py deploys this (weights, expert data, schedule)
    checkpoint.save(trained.states, d / "model.pt")  # the zones' weights alone
    (d / "training.json").write_text(json.dumps(training_info(trained), indent=1), encoding="utf-8")
    ckpt.unlink(missing_ok=True)
    note(f"done: {len(trained.episodes)} episodes in {time.monotonic() - started:.0f} s")
    report(status="done", phase="trained", message=f"trained: {len(trained.episodes)} episodes")
    return "trained"


def _new_session(
    cfg: Any, paths: MapPaths, scenario: str, method: str, plan: Plan, report: Any, note: Any
) -> Any:
    from pcnme.methods.learners import d3qn_session
    from pcnme.methods.lstm import lstm_session
    from pcnme.methods.ppo import ppo_session

    zones = cfg.layout.active_zones
    budget = plan.budget or cfg.fast_mode.train_decisions
    seconds, n_cars = plan.train_seconds, plan.n_cars
    if method in ("pcnme", "d3qn-scratch", "d3qn-greedy"):
        demos: Any = ()
        if method != "d3qn-scratch":
            from pcnme.expert.demos import load_samples

            kind = "nsga" if method == "pcnme" else "greedy"
            files = sorted(paths.demos.glob(f"{kind}-{scenario}-w*.parquet"))
            if len(files) < 2:  # STAND-IN (QUESTIONS Q59): the held-out split needs 2 windows
                files = sorted(paths.demos.glob(f"{kind}-*-w*.parquet"))
                note(
                    f"{scenario} has fewer than 2 windows with {kind} demonstrations: "
                    f"using the map's ({len(files)})"
                )
                report(message=f"fewer than 2 {kind} demo windows: pretraining on the map's (Q59)")
            if len(files) < 2:
                raise RuntimeError(f"{method} on {scenario}: fewer than 2 windows of {kind} demonstrations")
            demos = load_samples(files, cfg.nsga2.expert_filter_margin)
            note(f"{len(demos)} demonstration transitions from {len(files)} windows")

        def on_pretrain(u: int, total: int, agreement: float, loss: float) -> None:
            report(
                status="running",
                phase="pretraining (DQfD)",
                done=u,
                total=total,
                unit="updates",
                numbers={"agreement": round(agreement, 4), "loss": float(f"{loss:.4g}")},
            )
            note(f"pretraining update {u}/{total}: held-out agreement {agreement:.4f}, loss {loss:.4g}")

        if method != "d3qn-scratch":
            report(
                status="running",
                phase="pretraining (DQfD)",
                done=0,
                total=plan.pretrain_updates or cfg.dqfd.pretrain_updates,
                unit="updates",
            )
        return d3qn_session(
            cfg,
            method,
            scenario,
            zones,
            budget,
            demos=demos,
            pretrain_updates=plan.pretrain_updates,
            on_pretrain=on_pretrain,
            seconds=seconds,
            n_cars=n_cars,
        )
    if method in ("tang-wong", "qeco"):
        return lstm_session(cfg, method, scenario, zones, budget, seconds=seconds, n_cars=n_cars)
    return ppo_session(cfg, scenario, zones, budget, seconds=seconds, n_cars=n_cars)


def _episode_row(n: int, ep: Any, used: int, wall: float) -> dict[str, Any]:
    return {
        "episode": n,
        "seed": ep.seed,
        "on_time": ep.on_time,
        "mean_e_hat": ep.mean_e_hat,
        "tasks": ep.tasks,
        "updates": ep.updates,
        "mean_loss": ep.mean_loss,
        "decisions_used": used,
        "wall_s": round(wall, 1),
    }


def _rewrite_episodes(d: Path, session: Any) -> None:
    """episodes.csv from the session itself (the checkpoint is the truth after a restart)."""
    rows = [_episode_row(i + 1, ep, 0, 0.0) for i, ep in enumerate(session.out.episodes)]
    _write_csv(d / "episodes.csv", rows, fields=list(_episode_row(0, _Blank(), 0, 0.0)))


class _Blank:
    seed = on_time = mean_e_hat = tasks = updates = mean_loss = 0


def stage_learning(screen: Any, plan: Plan, w_star: float) -> None:
    from pcnme.pipeline.screen import run_jobs

    jobs = [(f"{s} {m}", (plan.map, s, m, w_star, plan)) for s in plan.scenarios for m in TRAINED]
    screen.stage(
        "learning: PCNME (pretraining + online) and the learning baselines (16.1)", [n for n, _ in jobs]
    )
    res = run_jobs(screen, _train_job, jobs, plan.workers)
    failed = {k: v for k, v in res.items() if isinstance(v, Exception)}
    if failed:
        raise RuntimeError(
            f"{len(failed)} trainings failed: {sorted(failed)} (their train.log); rerun to retry"
        )


# --- csv helpers --------------------------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --- the run ------------------------------------------------------------------------------------------------


def estimate_hours(plan: Plan, cfg: Any, traces_missing: int) -> dict[str, float]:
    pop = plan.population or cfg.nsga2.population
    gen = plan.generations or cfg.nsga2.generations
    windows = len(plan.scenarios) * plan.windows
    nsga = windows * pop * gen * CORE_S_PER_EVALUATION / min(plan.workers, max(windows, 1))
    budget = plan.budget or cfg.fast_mode.train_decisions
    jobs = len(plan.scenarios) * len(TRAINED)
    learn = jobs * budget / DECISIONS_PER_S / plan.workers
    trace = (cfg.map.warmup_s + cfg.map.trace_s) * TRACE_S_PER_SIMULATED_S if traces_missing else 0.0
    demos = windows * 2 * 60.0 / plan.workers
    return {"traces": trace / 3600, "nsga": nsga / 3600, "demos": demos / 3600, "learning": learn / 3600}


def map_ready(name: str) -> tuple[bool, str]:
    from pcnme.maps import map_config, trace_source

    cfg = map_config(name)
    if cfg.map.kind == "grid":
        return True, "grid map (built in code)"
    src = trace_source(cfg)
    if not (src / "lust.net.xml").exists():
        return False, f"{name}: the SUMO scenario is not at {src} (download: data/DATA.md)"
    try:
        from pcnme.mobility.lust import sumo_binary

        if not sumo_binary().exists():
            raise ImportError
    except ImportError:
        return False, f"{name}: SUMO is not installed (uv sync --locked --extra maps)"
    return True, "trace map (SUMO scenario found)"


def parse(argv: list[str] | None, *, evaluate: bool = False) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--maps", nargs="+", help="maps from the config (vfc, lust)")
    g.add_argument("--all", action="store_true", help="every map in the config")
    ap.add_argument("--scenarios", nargs="+", default=None, help="default: S1-S6")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--smoke", action="store_true", help="a 2-3 minute end-to-end check in data/smoke/")
    ap.add_argument("--yes", action="store_true", help="start without waiting for Enter")
    ap.add_argument("--fresh", action="store_true", help="archive this map's earlier outputs and start over")
    if not evaluate:
        ap.add_argument(
            "--windows",
            type=int,
            default=None,
            help="NSGA-II windows per scenario (nsga2.windows_per_scenario)",
        )
        ap.add_argument("--population", type=int, default=None, help="nsga2.population")
        ap.add_argument("--generations", type=int, default=None, help="nsga2.generations")
        ap.add_argument(
            "--budget",
            type=int,
            default=None,
            help="training decisions per method and scenario (fast_mode.train_decisions)",
        )
        ap.add_argument("--pretrain-updates", type=int, default=None, help="dqfd.pretrain_updates")
        ap.add_argument("--checkpoint-minutes", type=float, default=10.0)
    return ap.parse_args(argv)


def smoke_env() -> None:
    """--smoke: outputs in data/smoke/, the downloaded maps where they are."""
    os.environ["PCNME_DATA_DIR"] = str(REPO / "data" / "smoke")
    os.environ.setdefault("PCNME_MAPS_DIR", str(REPO / "data" / "maps"))


def archive(paths: MapPaths) -> Path | None:
    """--fresh: move the map's outputs aside (data/_archive/<map>-<time>/); nothing is deleted."""
    if not paths.root.exists():
        return None
    dest = paths.root.parent / "_archive" / f"{paths.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    paths.root.rename(dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    if args.smoke:
        smoke_env()
    from pcnme.config import DEFAULT_CONFIG_PATH, load_yaml
    from pcnme.maps import config_hash, map_config, map_source
    from pcnme.mobility.traces import trace_file
    from pcnme.pipeline import runinfo
    from pcnme.pipeline.screen import Screen, hms

    base = load_yaml(DEFAULT_CONFIG_PATH)
    names = list(base["maps"]) if args.all else args.maps
    plans: list[Plan] = []
    print(
        f"PCNME train.py   commit {runinfo.commit()}   "
        f"data {os.environ.get('PCNME_DATA_DIR', REPO / 'data')}\n"
    )
    for name in names:
        if name not in base["maps"]:
            print(f"  {name}: no such map in config/default.yaml (maps: {', '.join(base['maps'])}): skipped")
            continue
        ok, why = map_ready(name)
        if not ok:
            print(f"  {name}: NOT READY, skipped: {why}")
            continue
        cfg = map_config(name)
        smoke = args.smoke
        plan = Plan(
            map=name,
            scenarios=tuple(args.scenarios or (["S1"] if smoke else list(cfg.scenarios))),
            windows=2 if smoke else (args.windows or cfg.nsga2.windows_per_scenario),
            population=6 if smoke else args.population,
            generations=2 if smoke else args.generations,
            budget=400 if smoke else args.budget,
            train_seconds=3.0 if smoke else None,
            n_cars=40 if smoke else None,
            pretrain_updates=60 if smoke else args.pretrain_updates,
            workers=args.workers,
            checkpoint_every_s=0.0 if smoke else 60 * args.checkpoint_minutes,
            smoke=smoke,
        )
        paths = MapPaths(name)
        missing = (
            0
            if cfg.map.kind == "grid"
            else sum(
                not trace_file(paths.traces, t).exists()
                for t in {cfg.map.scenario_time[s] for s in plan.scenarios}
            )
        )
        est = estimate_hours(plan, cfg, missing)
        print(f"  {name} ({why}) -> {paths.root}")
        print(
            f"    scenarios {', '.join(plan.scenarios)}; NSGA-II {plan.windows} windows each, "
            f"{plan.population or cfg.nsga2.population} x {plan.generations or cfg.nsga2.generations}; "
            f"budget {plan.budget or cfg.fast_mode.train_decisions:,} decisions for each of "
            f"{', '.join(TRAINED)}; {plan.workers} workers"
        )
        print(
            "    estimate: "
            + ", ".join(f"{k} {v:.1f} h" for k, v in est.items() if v)
            + f"; total about {sum(est.values()):.1f} h"
        )
        plans.append(plan)
    if not plans:
        print("\nnothing to train.")
        return 1
    if not args.yes:
        input("\nPress Enter to start (Ctrl+C to stop) ")
    for plan in plans:
        paths = MapPaths(plan.map)
        if args.fresh and (dest := archive(paths)):
            print(f"{plan.map}: earlier outputs moved to {dest}")
        source = map_source(plan.map, measured=False)
        identity = {
            "config_hash": config_hash(source),
            "commit": runinfo.commit(),
            "plan": asdict(plan) | {"workers": None, "checkpoint_every_s": None},
        }
        try:
            runinfo.check_or_start(
                paths.manifest, identity, {"map": plan.map, "seeds": "training 2000-, NSGA-II 1000-"}
            )
        except runinfo.Mismatch as e:
            print(f"\n{plan.map}: {e}")
            return 2
        paths.logs.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        with Screen(f"train.py  map {plan.map}{'  (smoke)' if plan.smoke else ''}") as screen:
            if map_config(plan.map).map.kind == "trace":
                stage_traces(screen, plan)
            w = stage_nsga(screen, plan)
            stage_demos(screen, plan, w)
            stage_learning(screen, plan, w)
            screen.say(f"map {plan.map}: training finished in {hms(time.monotonic() - t0)}")
        runinfo.update(
            paths.manifest, finished=runinfo.now(), w_star=w, hours=round((time.monotonic() - t0) / 3600, 2)
        )
        print(f"\n{plan.map}: done. Outputs: {paths.root}. Next: python evaluate.py --maps {plan.map}")
    return 0
