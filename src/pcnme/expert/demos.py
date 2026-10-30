"""Demonstrations from NSGA-II plans (final-design 13.4; PLAN M5.4). THIS TOUCHES THE AGENT: these are its
pretraining data (13.5).

A plan is replayed in fast mode with the agent's input builder on (13.4.1-13.4.2): the real simulator, so s
holds exactly what the agent would see (stale state, pending sends, masks). The 48 inputs carry w = the plan's
own weight (the knee's w, or w_p for another front point, 13.4.4), and the rewards use the same w (QUESTIONS
Q40 stand-in); s', Δt and r come from the same replay (11.3, 11.4). A transition is an expert transition when
its action is the plan's gene: a window task's first decision whose gene was open. A masked gene replaced by
greedy, and every greedy re-decision, are stored without the flag (13.4.3; a refused planned action keeps its
flag: QUESTIONS Q39). Only window tasks' transitions are stored; the anchors' other decisions still serve
as s'.
Boulders make no transition (13.4.5). Replaying a plan gives the objectives its evaluation gave, exactly.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pcnme.agent.inputs import INPUT_NAMES
from pcnme.agent.nstep import Sample, anchor_samples
from pcnme.agent.transitions import Transition
from pcnme.config import Config
from pcnme.expert.knee import knee, weights
from pcnme.expert.window import Objectives, Window, objectives, simulate
from pcnme.zone.decision import OPTIONS


@dataclass(frozen=True, slots=True)
class DemoTransition:
    transition: Transition
    expert: bool
    sample: Sample  # the learner's view: the 1-step and n-step parts over the anchor's whole chain (13.5)


@dataclass(frozen=True, slots=True)
class Demonstration:
    """One replayed plan: its weight, its objectives (as evaluated) and its window tasks' transitions."""

    window: Window
    plan: int  # its index on the front
    genome: tuple[int, ...]
    w: float
    objectives: Objectives
    transitions: tuple[DemoTransition, ...]
    is_knee: bool = False


def is_expert(t: Transition, applied: Collection[object]) -> bool:
    """13.4.3: the plan's own action: a window task's first decision that took its gene."""
    return t.attempt == 1 and t.task_id in applied


def with_w(cfg: Config, w: float) -> Config:
    """The config with the plan's weight: input 48 and the rewards' w1 (13.4.1, Q40)."""
    if not 0.0 < w <= 1.0:
        raise ValueError(f"w must be in (0, 1], got {w}")
    return replace(cfg, reward=replace(cfg.reward, w=w))


def replay(
    cfg: Config, window: Window, genome: Sequence[int], w: float, plan: int = 0, is_knee: bool = False
) -> Demonstration:
    """13.4.1: replay a plan to the horizon (every window task settled, every s' seen), then end the
    episode (11.3.4: the last transitions are truncated)."""
    cw = with_w(cfg, w)
    run = simulate(cw, window.spec, window.genes(genome))
    run.city.end()
    keys = {t.key for t in window.tasks}
    rl = cw.rl
    out = []
    for z, zone in run.city.zones.items():
        applied = run.placements[z].applied
        # the n-step chain runs over every decision of the anchor, window task or not (13.5)
        order = [(d.task_id, d.attempt) for d in zone.decisions]
        samples = anchor_samples(
            zone.transitions.replay,
            order,
            cw.dqfd.n_step,
            rl.gamma_per_tau,
            rl.tau,
            lambda t, applied=applied: is_expert(t, applied),  # type: ignore[misc]
        )
        for t in zone.transitions.replay:
            if (z, t.task_id) in keys:
                out.append(DemoTransition(t, is_expert(t, applied), samples[(t.task_id, t.attempt)]))
    out.sort(key=lambda d: (d.transition.time, d.transition.anchor))
    obj = objectives(cw, run.records(t.key for t in window.tasks))
    return Demonstration(window, plan, tuple(genome), w, obj, tuple(out), is_knee)


def demonstrate_front(
    cfg: Config, window: Window, genomes: Sequence[Sequence[int]], F: Sequence[tuple[float, float]]
) -> list[Demonstration]:
    """13.4.4: every distinct front plan, each with its own w_p; the knee marked. A front with fewer than 2
    distinct points has no weight: no demonstrations."""
    wp = weights(F)
    kn = knee(F)
    out, seen = [], set()
    for i, (g, p) in enumerate(zip(genomes, F, strict=True)):
        if p in seen or p not in wp:
            continue
        seen.add(p)
        out.append(replay(cfg, window, g, wp[p], i, kn is not None and kn.point == p))
    return out


def rows(demos: Sequence[Demonstration]) -> list[dict[str, Any]]:
    """One row per transition: the window, the plan and its w, the transition and its expert flag; s as
    x_<input>, s' as xn_<input> (empty when truncated with no s'), the valid actions of s and s'; the n-step
    part (13.5): has1, g1, hasn, m, rn, gn, s_m as xm_<input> and its valid actions as valid_m_<option>."""
    out = []
    for d in demos:
        spec = d.window.spec
        for dt in d.transitions:
            t = dt.transition
            row: dict[str, Any] = {
                "scenario": spec.scenario,
                "window": spec.index,
                "seed": spec.seed,
                "plan": d.plan,
                "knee": d.is_knee,
                "w": d.w,
                "anchor": t.anchor,
                "task_id": str(t.task_id),
                "attempt": t.attempt,
                "time_s": t.time,
                "expert": dt.expert,
                "action": OPTIONS[t.a],
                "a": t.a,
                "reward": t.r,
                "dt_s": t.dt,
                "truncated": t.truncated,
            }
            row.update({f"x_{n}": v for n, v in zip(INPUT_NAMES, t.s, strict=True)})
            nxt = t.s_next if t.s_next is not None else (None,) * len(INPUT_NAMES)
            row.update({f"xn_{n}": v for n, v in zip(INPUT_NAMES, nxt, strict=True)})
            row.update({f"valid_{o}": v for o, v in zip(OPTIONS, t.valid, strict=True)})
            vn = t.valid_next if t.valid_next is not None else (None,) * len(OPTIONS)
            row.update({f"valid_next_{o}": v for o, v in zip(OPTIONS, vn, strict=True)})
            smp = dt.sample
            row.update(
                {"has1": smp.has1, "g1": smp.g1, "hasn": smp.hasn, "m": smp.m, "rn": smp.rn, "gn": smp.gn}
            )
            sn = smp.sn if smp.sn is not None else (None,) * len(INPUT_NAMES)
            row.update({f"xm_{n}": v for n, v in zip(INPUT_NAMES, sn, strict=True)})
            vm = smp.validn if smp.validn is not None else (None,) * len(OPTIONS)
            row.update({f"valid_m_{o}": v for o, v in zip(OPTIONS, vm, strict=True)})
            out.append(row)
    return out


def write(path: Path, demos: Sequence[Demonstration]) -> Path:
    """The demonstrations as one Parquet file (the store M6 pretrains from)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows(demos)), path)
    return path


def load_samples(paths: Sequence[Path]) -> list[tuple[tuple[str, int], Sample]]:
    """The demonstrations as the learner's samples, each with its (scenario, window) for the held-out
    split."""
    import pyarrow.parquet as pq

    def opt(row: dict[str, Any], prefix: str, names: Sequence[str]) -> tuple[Any, ...] | None:
        vals = tuple(row[f"{prefix}{n}"] for n in names)
        return None if vals[0] is None else vals

    out = []
    for path in paths:
        for row in pq.read_table(path).to_pylist():
            smp = Sample(
                s=tuple(row[f"x_{n}"] for n in INPUT_NAMES),
                a=row["a"],
                r=row["reward"],
                valid=tuple(row[f"valid_{o}"] for o in OPTIONS),
                expert=row["expert"],
                has1=row["has1"],
                s1=opt(row, "xn_", INPUT_NAMES),
                valid1=opt(row, "valid_next_", OPTIONS),
                g1=row["g1"],
                hasn=row["hasn"],
                m=row["m"],
                rn=row["rn"],
                sn=opt(row, "xm_", INPUT_NAMES),
                validn=opt(row, "valid_m_", OPTIONS),
                gn=row["gn"],
            )
            out.append(((row["scenario"], row["window"]), smp))
    return out


def greedy_demonstrations(cfg: Config, window: Window) -> Demonstration:
    """The greedy expert as teacher (16.1, D3QN + greedy expert): the window's greedy plan replayed with the
    deployment w (no front, so no w_p), every first decision flagged as the plan's own."""
    return replay(cfg, window, window.greedy_genome, cfg.reward.w)
