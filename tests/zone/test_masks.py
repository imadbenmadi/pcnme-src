"""Masks, q*, Greedy and Random in one zone (PLAN M2.8; final-design 7.2, 7.3, 16.1, 12.4)."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from scipy import stats

from pcnme.car.car import Task
from pcnme.config import Config, load_config
from pcnme.energy import EnergyModel
from pcnme.ids import TaskId
from pcnme.methods.simple import Greedy, Random, greedy_pick
from pcnme.node.state import SlotState
from pcnme.sim.rng import RandomStreams
from pcnme.zone.estimates import PathEstimates, ZeroWanWait
from pcnme.zone.masks import OptionView, views
from pcnme.zone.zone import DecisionContext
from tests.zone.harness import fixed_delay_config, one_zone, task_now


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


def empty(cfg: Config, hw: str) -> dict[str, SlotState]:
    node = cfg.hardware[hw]
    return {
        q: SlotState(
            c=node.slots(q),
            capacity=node.queue[q],
            n_hard=0,
            n_soft=0,
            b_hard=0.0,
            b_soft=0.0,
            remaining=(0.0,) * node.slots(q),
            u=0,
        )
        for q in node.slot_types
    }


def ctx(
    cfg: Config,
    kind: str,
    time_left: float | None = None,
    here: dict | None = None,
    hw: str = "F-GPU",
    cloud: dict | None = None,
) -> DecisionContext:
    t = cfg.task_types[kind]
    task = Task(
        TaskId("c01", 0, 1),
        "c01",
        kind,
        t.cls,
        t.deadline["loose"],
        t.l_max["loose"],
        t.input_bits,
        t.result_bits,
        0.0,
    )
    left = t.deadline["loose"] if time_left is None else time_left
    return DecisionContext(
        task=task,
        now=0.0,
        d_a=left,
        here=here or empty(cfg, hw),
        cloud=cloud or empty(cfg, "cloud"),
        backhaul_wait=0.0,
        estimates=PathEstimates(cfg, ZeroWanWait()),
        here_hw=hw,
    )  # type: ignore[arg-type]


def by_option(cfg: Config, c: DecisionContext) -> dict[str, OptionView]:
    return {v.option: v for v in views(cfg, EnergyModel(cfg), c)}


# --- masks (7.2) --------------------------------------------------------------------------------------------


def test_in_one_zone_the_neighbour_positions_are_empty(cfg: Config) -> None:
    v = by_option(cfg, ctx(cfg, "T1"))
    assert [v[n].masked for n in ("n1", "n2", "n3", "n4")] == ["empty"] * 4
    assert list(v) == ["here", "n1", "n2", "n3", "n4", "cloud"]


def test_q_star_and_the_estimates_of_an_empty_system(cfg: Config) -> None:
    # T1 here: q* = GPU (0 + 8 ms < 0 + 90 ms); finish = 0 + 0 + 8 + 10 = 18 ms. Its energy estimate (12.4) =
    # E_ref (here, mean run on the reference class).
    v = by_option(cfg, ctx(cfg, "T1"))
    assert (v["here"].masked, v["here"].slot) == (None, "gpu")
    assert v["here"].finish == pytest.approx(0.018)
    assert v["here"].energy == pytest.approx(cfg.derived.e_ref["T1"])


def test_cannot_run_when_no_slot_type_has_a_row(cfg: Config) -> None:
    v = by_option(cfg, ctx(cfg, "T2", hw="F-CPU"))
    assert v["here"].masked == "cannot_run"  # T2 is GPU only
    assert v["cloud"].masked is None


def test_cannot_meet_the_deadline(cfg: Config) -> None:
    # A hard T1 to the cloud: 67.8 + 0 + 4 + 69.5 = 141.3 ms > 100 ms left (16.2's "about 140 ms").
    v = by_option(cfg, ctx(cfg, "T1"))
    assert v["cloud"].masked == "cannot_meet"
    assert v["cloud"].finish == pytest.approx(0.1413, abs=5e-4)
    # here: 18 ms; with 17 ms left it cannot meet either; exactly its finish left is in time (">" in 7.2).
    finish = v["here"].finish
    assert by_option(cfg, ctx(cfg, "T1", time_left=0.017))["here"].masked == "cannot_meet"
    assert by_option(cfg, ctx(cfg, "T1", time_left=finish))["here"].masked is None


def test_a_soft_task_is_judged_against_time_left_plus_l_max(cfg: Config) -> None:
    # T4 cloud: 67.8 + 0 + 12 + 69.5 = 149.3 ms. 0 s left but L_max = 1 s: still allowed.
    assert by_option(cfg, ctx(cfg, "T4", time_left=0.0))["cloud"].masked is None
    assert by_option(cfg, ctx(cfg, "T4", time_left=-0.9))["cloud"].masked == "cannot_meet"


def test_full_here_uses_no_headroom_and_the_hard_reserve(cfg: Config) -> None:
    here = empty(cfg, "F-GPU")
    full_gpu = replace(here["gpu"], n_hard=32, u=2, remaining=(0.008, 0.008), b_hard=32 * 0.008)
    full_cpu = replace(here["cpu"], n_hard=16, u=2, remaining=(0.09, 0.09), b_hard=16 * 0.09)
    assert (
        by_option(cfg, ctx(cfg, "T1", time_left=10.0, here={"gpu": full_gpu, "cpu": full_cpu}))["here"].masked
        == "full"
    )
    # h = 0 for the anchor's own queue: 31 queued still leaves a place for a hard task.
    one_left = {"gpu": replace(full_gpu, n_hard=31), "cpu": full_cpu}
    assert by_option(cfg, ctx(cfg, "T1", time_left=10.0, here=one_left))["here"].slot == "gpu"
    # A soft task needs n < Q - r = 28 on the GPU and < 12 on the CPU.
    soft_full = {"gpu": replace(full_gpu, n_hard=28), "cpu": replace(full_cpu, n_hard=12)}
    assert by_option(cfg, ctx(cfg, "T4", time_left=10.0, here=soft_full))["here"].masked == "full"


def test_the_cloud_never_fills(cfg: Config) -> None:
    cloud = {q: replace(s, n_hard=s.capacity) for q, s in empty(cfg, "cloud").items()}
    assert by_option(cfg, ctx(cfg, "T4", cloud=cloud))["cloud"].masked is None


def test_q_star_compares_wait_plus_run_not_wait_alone(cfg: Config) -> None:
    # GPU busy 50 ms: 50 + 8 = 58 ms; the CPU is free but slow: 0 + 90 ms. q* = GPU (by wait alone it would be
    # the CPU).
    here = empty(cfg, "F-GPU")
    gpu_busy = {"gpu": replace(here["gpu"], u=2, remaining=(0.05, 0.05)), "cpu": here["cpu"]}
    assert by_option(cfg, ctx(cfg, "T1", time_left=1.0, here=gpu_busy))["here"].slot == "gpu"


def test_the_headroom_h_for_options_other_than_here(cfg: Config) -> None:
    # 7.2 (R2): a neighbour is masked "full" at n >= Q - h (hard), n >= Q - r - h (soft); h = 2, r = 4, Q =
    # 32.
    from pcnme.zone.masks import _has_place

    s = replace(empty(cfg, "F-GPU")["gpu"], n_hard=29)
    assert _has_place(s, "hard", headroom=2, reserve=4, never_fills=False)  # 29 < 30
    assert not _has_place(replace(s, n_hard=30), "hard", headroom=2, reserve=4, never_fills=False)
    assert _has_place(replace(s, n_hard=25), "soft", headroom=2, reserve=4, never_fills=False)  # 25 < 26
    assert not _has_place(replace(s, n_hard=26), "soft", headroom=2, reserve=4, never_fills=False)
    assert _has_place(replace(s, n_hard=31), "hard", headroom=0, reserve=4, never_fills=False)  # here: h = 0


def test_q_star_moves_to_the_cpu_when_the_gpu_would_finish_later(cfg: Config) -> None:
    # GPU slots busy for 100 ms each: Ŵ_gpu = 100 ms, F_gpu = 108 ms > F_cpu = 90 ms (free).
    here = empty(cfg, "F-GPU")
    busy = {"gpu": replace(here["gpu"], u=2, remaining=(0.1, 0.1)), "cpu": here["cpu"]}
    assert by_option(cfg, ctx(cfg, "T1", time_left=1.0, here=busy))["here"].slot == "cpu"


# --- Greedy (16.1) ------------------------------------------------------------------------------------------


def test_greedy_takes_the_earliest_finish(cfg: Config) -> None:
    assert Greedy(cfg).choose(ctx(cfg, "T1")) == "here"  # cloud masked
    assert Greedy(cfg).choose(ctx(cfg, "T4")) == "here"  # 35 ms here vs 149 ms in the cloud
    here = empty(cfg, "F-GPU")
    swamped = {
        "gpu": replace(here["gpu"], u=2, remaining=(0.5, 0.5)),
        "cpu": replace(here["cpu"], u=2, remaining=(0.5, 0.5)),
    }
    assert Greedy(cfg).choose(ctx(cfg, "T4", here=swamped)) == "cloud"  # here 0.5 + 0.025 + 0.01 > 0.149


def test_greedy_breaks_ties_by_energy_then_by_option_order() -> None:
    a = OptionView("here", None, "gpu", 0.0, 0.010, 0.0, 0.010, energy=0.5)
    b = OptionView("cloud", None, "gpu", 0.0, 0.010, 0.0, 0.010, energy=0.3)
    assert greedy_pick([a, b]) is b  # same finish, less energy
    c = replace(b, energy=0.5)
    assert greedy_pick([c, a]) is a  # same finish and energy: here before cloud
    assert greedy_pick([replace(a, masked="full"), replace(b, masked="empty")]) is None


def test_every_option_masked_returns_none(cfg: Config) -> None:
    assert Greedy(cfg).choose(ctx(cfg, "T1", time_left=0.001)) is None


# --- Random (16.1) ------------------------------------------------------------------------------------------


def test_random_is_uniform_over_the_unmasked_options_and_repeatable(cfg: Config) -> None:
    streams = RandomStreams.from_config(cfg, 2000, "training")
    r = Random(cfg, streams, "z1")
    picks = Counter(r.choose(ctx(cfg, "T4")) for _ in range(4000))
    assert set(picks) == {"here", "cloud"}
    assert stats.chisquare(list(picks.values())).pvalue > 1e-3
    again = Random(cfg, RandomStreams.from_config(cfg, 2000, "training"), "z1")
    first = Random(cfg, RandomStreams.from_config(cfg, 2000, "training"), "z1")
    assert [again.choose(ctx(cfg, "T4")) for _ in range(50)] == [
        first.choose(ctx(cfg, "T4")) for _ in range(50)
    ]


def test_random_never_picks_a_masked_option(cfg: Config) -> None:
    r = Random(cfg, RandomStreams.from_config(cfg, 2000, "training"), "z1")
    assert {r.choose(ctx(cfg, "T1")) for _ in range(200)} == {"here"}  # cloud cannot meet a T1


# --- in a running zone --------------------------------------------------------------------------------------


def test_greedy_runs_a_zone_end_to_end() -> None:
    cfg = fixed_delay_config()
    s = one_zone(cfg, Greedy(cfg))
    s.zone.upload(task_now(s, "T1"))
    s.clock.run_until(1.0)
    assert (s.closed[0].option, s.closed[0].outcome) == ("here", "on_time")
