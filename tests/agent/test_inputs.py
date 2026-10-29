"""The agent's 48 inputs (PLAN M4.1; final-design 11.1, 12.3, 12.4, 7.2, 10.2; QUESTIONS Q32-Q34).

Expected values are by hand from config/default.yaml: mean delays backhaul 3 ms, broker 1 ms, radio downlink
3 ms, WAN 62.5 ms; T1 150 KB in, 3 KB out; T2 9 000 KB; the largest deadline is T2's loose 3 s.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pcnme.agent.inputs import (
    INPUT_NAMES,
    N_INPUTS,
    Scales,
    build_inputs,
    next_zone_index,
)
from pcnme.car.car import Task
from pcnme.config import Config, load_config
from pcnme.energy import EnergyModel
from pcnme.ids import TaskId
from pcnme.node.state import Pending, SlotState
from pcnme.zone.decision import DecisionContext, Mobility, NeighbourView
from pcnme.zone.estimates import PathEstimates, ZeroWanWait
from pcnme.zone.masks import views

E_BH = 0.141e-9 * 1.5
E_WAN = 0.228e-9 * 1.5
E_DOWN = 4.2e-9
T1_IN = 150 * 8000
T1_OUT = 3 * 8000
E_RET_FOG = (2 * E_BH + E_DOWN) * T1_OUT
E_REF_T1 = 1.5 * 25 * 0.008 + E_RET_FOG  # 12.3: T1 on F-GPU's GPU slot, here
T_RET_FOG = 0.003 + 0.001 + 0.003 + 0.003  # Q16: backhaul, broker, backhaul, radio downlink
T_IN_NB_T1 = T1_IN / 1e9 + 0.003  # 4.3: own backhaul wait 0 + S / B_bh + δ̄_bh
POSITIONS = ("z2", "z3", "z4", None)


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


def slots(cfg: Config, hw: str, **over: dict[str, object]) -> dict[str, SlotState]:
    node = cfg.hardware[hw]
    out = {}
    for q in node.slot_types:
        base: dict[str, object] = dict(
            c=node.slots(q),
            capacity=node.queue[q],
            n_hard=0,
            n_soft=0,
            b_hard=0.0,
            b_soft=0.0,
            remaining=(0.0,) * node.slots(q),
            u=0,
        )
        base.update(over.get(q, {}))
        out[q] = SlotState(**base)  # type: ignore[arg-type]
    return out


def make_ctx(
    cfg: Config,
    kind: str = "T1",
    time_left: float | None = None,
    mode: str = "loose",
    here: dict[str, SlotState] | None = None,
    hw: str = "F-GPU",
    cloud: dict[str, SlotState] | None = None,
    neighbours: tuple[NeighbourView, ...] = (),
    mobility: Mobility | None = None,
    cloud_age: float | None = 0.05,
    cloud_stale: bool = False,
) -> DecisionContext:
    t = cfg.task_types[kind]
    task = Task(
        TaskId("c01", 0, 1),
        "c01",
        kind,
        t.cls,
        t.deadline[mode],  # type: ignore[index]
        t.l_max[mode],  # type: ignore[index]
        t.input_bits,
        t.result_bits,
        0.0,
    )
    left = task.deadline if time_left is None else time_left
    c = DecisionContext(
        task=task,
        now=0.0,
        d_a=left,
        here=here or slots(cfg, hw),
        cloud=cloud if cloud is not None else slots(cfg, "cloud"),
        backhaul_wait=0.0,
        estimates=PathEstimates(cfg, ZeroWanWait()),
        here_hw=hw,
        cloud_stale=cloud_stale,
        neighbours=neighbours,
        cloud_age=cloud_age,
        mobility=mobility,
    )
    return replace(c, views=tuple(views(cfg, EnergyModel(cfg), c)))


def inputs(cfg: Config, c: DecisionContext) -> dict[str, float]:
    x = build_inputs(cfg, EnergyModel(cfg), c, Scales.from_config(cfg), "z1", POSITIONS)
    assert len(x) == N_INPUTS
    return dict(zip(INPUT_NAMES, x, strict=True))


def option(x: dict[str, float], o: str) -> list[float]:
    return [x[f"{o}_{f}"] for f in ("present", "wait", "exec", "network", "energy", "age")]


# --- length and order (11.1) --------------------------------------------------------------------------------


def test_48_inputs_in_the_order_of_11_1() -> None:
    per = ("present", "wait", "exec", "network", "energy", "age")
    expected = (
        ["input_size", "ref_exec", "slack", "hard", "D_norm"]
        + [f"{o}_{f}" for o in ("here", "n1", "n2", "n3", "n4", "cloud") for f in per]
        + ["exit", "next_stays", "next_n1", "next_n2", "next_n3", "next_n4", "w"]
    )
    assert list(INPUT_NAMES) == expected and N_INPUTS == 48


# --- task (5) -----------------------------------------------------------------------------------------------


def test_the_task_inputs_by_hand(cfg: Config) -> None:
    x = inputs(cfg, make_ctx(cfg, "T1", time_left=0.08))
    assert x["input_size"] == pytest.approx(150 / 9000)  # ÷ the largest input, T2's 9 000 KB
    assert x["ref_exec"] == pytest.approx(0.008 / 0.1)  # F-GPU's GPU slot, its fastest, ÷ D
    assert x["slack"] == pytest.approx(0.8)
    assert x["hard"] == 1.0
    assert x["D_norm"] == pytest.approx(0.1 / 3.0)  # ÷ the largest deadline of any type: T2 loose, 3 s


def test_a_soft_task_and_its_d_norm_in_tight_mode(cfg: Config) -> None:
    x = inputs(cfg, make_ctx(cfg, "T4", mode="tight"))
    assert x["hard"] == 0.0
    assert x["D_norm"] == pytest.approx(0.5 / 3.0)  # the divisor is the same in every scenario
    assert x["ref_exec"] == pytest.approx(0.025 / 0.5)


def test_slack_is_capped_at_2_and_may_be_negative_for_a_soft_task(cfg: Config) -> None:
    assert inputs(cfg, make_ctx(cfg, "T1", time_left=0.5))["slack"] == 2.0
    assert inputs(cfg, make_ctx(cfg, "T4", time_left=-0.3))["slack"] == pytest.approx(-0.3)


# --- per option (36) ----------------------------------------------------------------------------------------


def test_an_empty_f_gpu_here_by_hand(cfg: Config) -> None:
    x = inputs(cfg, make_ctx(cfg, "T1"))
    present, wait, run, network, energy, age = option(x, "here")
    assert (present, wait, age) == (1.0, 0.0, 0.0)
    assert run == pytest.approx(0.08)  # q* = GPU: 8 ms ÷ 100 ms
    assert network == pytest.approx(T_RET_FOG / 0.1)  # here: only the return
    assert energy == pytest.approx(1.0 / cfg.derived.e_bar)  # Ê_here = E_ref exactly (12.3)


def test_the_cloud_by_hand(cfg: Config) -> None:
    x = inputs(cfg, make_ctx(cfg, "T1"))
    present, wait, run, network, energy, age = option(x, "cloud")
    t_in = T1_IN / cfg.derived.b_wan + 0.003 + 0.0625  # ŵ_WAN = 0 before any sample
    t_ret = 0.0625 + 0.001 + 0.003 + 0.003
    e = (E_BH + E_WAN) * T1_IN + 1.1 * 150 * 0.004 + (E_WAN + E_BH + E_DOWN) * T1_OUT
    assert (present, wait) == (1.0, 0.0)
    assert run == pytest.approx(0.04)
    assert network == pytest.approx((t_in + t_ret) / 0.1)
    assert energy == pytest.approx(e / E_REF_T1 / cfg.derived.e_bar)
    assert age == pytest.approx(0.05 / 0.3)  # ÷ 3 x the state period


def test_time_inputs_are_capped_at_2(cfg: Config) -> None:
    # Every slot busy for 0.5 s more with 1 s of hard backlog: water level 2 (W - 0.5) = 1, W = 1 s = 10 D on
    # both slot types; q* = GPU (1.008 s < 1.09 s).
    jam = dict(u=2, remaining=(0.5, 0.5), n_hard=4, b_hard=1.0)
    busy = {"gpu": jam, "cpu": jam}
    x = inputs(cfg, make_ctx(cfg, "T1", time_left=5.0, here=slots(cfg, "F-GPU", **busy)))
    assert x["here_wait"] == 2.0
    # T1 tight (D = 50 ms) to the cloud: (67.8 + 69.5) ms ÷ 50 ms = 2.75
    assert inputs(cfg, make_ctx(cfg, "T1", mode="tight"))["cloud_network"] == 2.0
    # T1 on a CPU slot only (an F-CPU zone): 90 ms ÷ 50 ms = 1.8, not capped
    assert inputs(cfg, make_ctx(cfg, "T1", mode="tight", hw="F-CPU"))["here_exec"] == pytest.approx(1.8)


def test_the_energy_ratio_is_capped_at_e_bar(cfg: Config) -> None:
    # T2 on the cloud: 1.1 x 150 W x 0.2 s = 33 J against E_ref(T2) = 1.5 x 25 x 0.72 = 27 J: ratio 1.22.
    x = inputs(cfg, make_ctx(cfg, "T2"))
    e = (E_BH + E_WAN) * 72e6 + 1.1 * 150 * 0.2 + (E_WAN + E_BH + E_DOWN) * 80_000
    e_ref = 1.5 * 25 * 0.72 + (2 * E_BH + E_DOWN) * 80_000
    assert x["cloud_energy"] == pytest.approx(e / e_ref / cfg.derived.e_bar)
    assert 0.0 < x["cloud_energy"] < 1.0
    big = replace(cfg, derived=replace(cfg.derived, e_bar=1.1))  # a cap below the ratio: clipped to 1
    assert inputs(big, make_ctx(big, "T2"))["cloud_energy"] == 1.0


def test_wait_counts_the_anchors_own_pending_sends(cfg: Config) -> None:
    # R1: two T1s sent to z2 and not yet answered fill its two free GPU slots: the third waits, so Ŵ > 0.
    n = NeighbourView("z2", "F-GPU", slots(cfg, "F-GPU"), {"gpu": (Pending("hard", 0.008),) * 3}, age=0.01)
    x = inputs(cfg, make_ctx(cfg, "T1", neighbours=(n,)))
    # water level over r = (0, 0) with backlog 3 x 8 ms: 2 W = 24 ms -> W = 12 ms
    assert x["n1_wait"] == pytest.approx(0.012 / 0.1)


# --- masked options still get their numbers (11.1, QUESTIONS Q33 stand-in) ---------------------------------


def test_an_empty_position_is_all_zeros(cfg: Config) -> None:
    x = inputs(cfg, make_ctx(cfg, "T1"))
    for o in ("n1", "n2", "n3", "n4"):
        assert option(x, o) == [0.0] * 6


def test_a_stale_neighbour_with_a_snapshot_gets_its_numbers_and_present_0(cfg: Config) -> None:
    n = NeighbourView("z2", "F-GPU", slots(cfg, "F-GPU"), stale=True, age=0.45)
    x = inputs(cfg, make_ctx(cfg, "T1", neighbours=(n,)))
    present, wait, run, network, energy, age = option(x, "n1")
    e = E_BH * T1_IN + 1.5 * 25 * 0.008 + E_RET_FOG
    assert (present, wait) == (0.0, 0.0)
    assert run == pytest.approx(0.08)
    assert network == pytest.approx((T_IN_NB_T1 + T_RET_FOG) / 0.1)
    assert energy == pytest.approx(e / E_REF_T1 / cfg.derived.e_bar)
    assert age == pytest.approx(1.5)  # 450 ms ÷ 300 ms


def test_an_offline_neighbour_gets_the_numbers_of_its_last_snapshot(cfg: Config) -> None:
    stale = NeighbourView("z2", "F-GPU", slots(cfg, "F-GPU"), stale=True, age=0.45)
    offline = replace(stale, offline=True, stale=False)
    a = option(inputs(cfg, make_ctx(cfg, "T1", neighbours=(stale,))), "n1")
    b = option(inputs(cfg, make_ctx(cfg, "T1", neighbours=(offline,))), "n1")
    assert a == b and a[0] == 0.0


def test_a_neighbour_with_no_snapshot_gets_only_its_path(cfg: Config) -> None:
    n = NeighbourView("z2", "F-GPU", None, stale=True, age=None)
    present, wait, run, network, energy, age = option(inputs(cfg, make_ctx(cfg, "T1", neighbours=(n,))), "n1")
    assert (present, wait, run, energy, age) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert network == pytest.approx((T_IN_NB_T1 + T_RET_FOG) / 0.1)


def test_a_stale_cloud_with_no_snapshot_gets_only_its_path(cfg: Config) -> None:
    c = make_ctx(cfg, "T1", cloud={}, cloud_stale=True, cloud_age=None)
    c = replace(c, cloud=None)
    c = replace(c, views=tuple(views(cfg, EnergyModel(cfg), c)))
    present, wait, run, network, energy, age = option(inputs(cfg, c), "cloud")
    assert (present, wait, run, energy, age) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert network > 1.0


def test_a_full_here_gets_the_numbers_of_its_best_slot_type_ignoring_room(cfg: Config) -> None:
    # T4 (soft) here: GPU 28 queued hard (soft admitted below Q - r = 28), CPU 12 queued (below 16 - 4).
    # Both GPU slots have 10 ms left: soft level 2 (W - 0.01) = 28 x 8 ms -> W = 0.122 s; CPU slots 50 ms left
    # with 1.08 s of backlog: W = 0.59 s. q* ignoring room: GPU (0.122 + 0.025 < 0.59 + 0.2).
    full = {
        "gpu": dict(u=2, remaining=(0.01, 0.01), n_hard=28, b_hard=28 * 0.008),
        "cpu": dict(u=2, remaining=(0.05, 0.05), n_hard=12, b_hard=12 * 0.09),
    }
    c = make_ctx(cfg, "T4", here=slots(cfg, "F-GPU", **full))
    assert c.views[0].masked == "full"
    present, wait, run, network, energy, _ = option(inputs(cfg, c), "here")
    assert present == 1.0  # full is not "absent"
    assert wait == pytest.approx(0.122)  # D = 1 s
    assert run == pytest.approx(0.025)
    assert energy == pytest.approx(1.0 / cfg.derived.e_bar)  # GPU here = E_ref(T4)
    assert network == pytest.approx(T_RET_FOG)


def test_cannot_run_gets_only_the_path(cfg: Config) -> None:
    # T2 is GPU only; an F-CPU zone has no GPU slot.
    c = make_ctx(cfg, "T2", hw="F-CPU")
    assert c.views[0].masked == "cannot_run"
    present, wait, run, network, energy, age = option(inputs(cfg, c), "here")
    assert (present, wait, run, energy, age) == (1.0, 0.0, 0.0, 0.0, 0.0)
    assert network == pytest.approx(T_RET_FOG / 3.0)


def test_cannot_meet_keeps_the_masks_own_numbers(cfg: Config) -> None:
    c = make_ctx(cfg, "T1", mode="tight", time_left=0.005)
    v = c.views[5]
    assert v.masked == "cannot_meet"
    x = option(inputs(cfg, c), "cloud")
    assert x[2] == pytest.approx(v.run / 0.05) and x[0] == 1.0


# --- mobility (6) and w (1) ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mobility", "expected"),
    [
        (None, [0.0, 0, 0, 0, 0, 0]),  # no GPS sample: unknown (Q34)
        (Mobility(math.inf, "z1"), [1.0, 1, 0, 0, 0, 0]),  # stopped in the anchor's zone: exit 1, stays
        (Mobility(0.05, "z2"), [0.5, 0, 1, 0, 0, 0]),  # 50 ms ÷ 100 ms; heading to position 1
        (Mobility(0.03, "z4"), [0.3, 0, 0, 0, 1, 0]),
        (Mobility(0.5, "z3"), [1.0, 0, 0, 1, 0, 0]),  # exit capped at 1
        (Mobility(0.02, "z9"), [0.2, 0, 0, 0, 0, 0]),  # a zone in none of the 5 positions (Q34)
    ],
)
def test_the_mobility_inputs(cfg: Config, mobility: Mobility | None, expected: list[float]) -> None:
    x = inputs(cfg, make_ctx(cfg, "T1", mobility=mobility))
    names = ("exit", "next_stays", "next_n1", "next_n2", "next_n3", "next_n4")
    assert [x[n] for n in names] == pytest.approx(expected)


def test_next_zone_index() -> None:
    assert next_zone_index("z1", "z1", POSITIONS) == 0
    assert next_zone_index("z3", "z1", POSITIONS) == 2
    assert next_zone_index(None, "z1", POSITIONS) is None
    assert next_zone_index("z5", "z1", POSITIONS) is None


def test_w_is_the_configs_stand_in(cfg: Config) -> None:
    assert inputs(cfg, make_ctx(cfg, "T1"))["w"] == 0.5  # QUESTIONS Q32


# --- every value in its range (PLAN M4 acceptance) ----------------------------------------------------------

ALL = ("here", "n1", "n2", "n3", "n4", "cloud")
UNIT = {"input_size", "D_norm", "exit", "w", *(f"{o}_energy" for o in ALL)}
FLAGS = {"hard", "next_stays", "next_n1", "next_n2", "next_n3", "next_n4"}
FLAGS |= {f"{o}_present" for o in ALL}


def check_ranges(x: dict[str, float]) -> None:
    for name, v in x.items():
        assert math.isfinite(v), name
        if name in FLAGS:
            assert v in (0.0, 1.0), (name, v)
        elif name in UNIT:
            assert 0.0 <= v <= 1.0, (name, v)
        elif name == "slack":
            assert -1.0 <= v <= 2.0, (name, v)  # a decision needs an open option: time left > -L_max = -D
        else:
            assert 0.0 <= v <= 2.0, (name, v)
    assert sum(x[f"next_{p}"] for p in ("stays", "n1", "n2", "n3", "n4")) <= 1.0


slot_st = st.fixed_dictionaries(
    {
        "u": st.integers(0, 2),
        "n_hard": st.integers(0, 40),
        "n_soft": st.integers(0, 40),
        "b": st.floats(0.0, 5.0),
        "r": st.floats(0.0, 2.0),
    }
)


@settings(max_examples=200, deadline=None)
@given(
    kind=st.sampled_from(["T1", "T3", "T4", "T2"]),
    mode=st.sampled_from(["loose", "tight"]),
    frac=st.floats(-1.0, 3.0),
    gpu=slot_st,
    cpu=slot_st,
    stale=st.booleans(),
    snap=st.booleans(),
    age=st.floats(0.0, 10.0),
    t_exit=st.one_of(st.just(math.inf), st.floats(0.0, 100.0)),
    target=st.sampled_from(["z1", "z2", "z3", "z4", "z5"]),
)
def test_every_input_is_in_its_range(
    cfg: Config,
    kind: str,
    mode: str,
    frac: float,
    gpu: dict[str, float],
    cpu: dict[str, float],
    stale: bool,
    snap: bool,
    age: float,
    t_exit: float,
    target: str,
) -> None:
    def state(s: dict[str, float]) -> dict[str, object]:
        u = int(s["u"])
        return dict(u=u, remaining=(s["r"],) * u + (0.0,) * (2 - u), n_hard=s["n_hard"], b_hard=s["b"])

    here = slots(cfg, "F-GPU", gpu=state(gpu), cpu=state(cpu))
    t = cfg.task_types[kind]
    left = frac * t.deadline[mode]  # type: ignore[index]
    if t.cls == "soft":
        left = max(left, -t.l_max[mode])  # type: ignore[index]
    n = NeighbourView("z2", "F-GPU", here if snap else None, stale=stale, age=age if snap else None)
    c = make_ctx(cfg, kind, left, mode, here, neighbours=(n,), mobility=Mobility(t_exit, target))
    check_ranges(inputs(cfg, c))


def test_ref_exec_is_capped_at_2_and_w_follows_the_config() -> None:
    # A T1 slower than 2 D on the reference class (a config no scenario uses: it would be a boulder).
    from tests.zone.harness import fixed_delay_config

    slow = fixed_delay_config(
        **{
            "time_table_placeholder__T1__F-GPU": {"gpu": 0.3, "cpu": 0.4},
            "reward__w": 0.8,
        }
    )
    x = inputs(slow, make_ctx(slow, "T1"))
    assert x["ref_exec"] == 2.0 and x["w"] == 0.8


def test_d_norm_divides_by_the_largest_deadline_over_both_modes() -> None:
    from tests.zone.harness import fixed_delay_config

    long_tight = fixed_delay_config(**{"task_types__T4__deadline__tight": 4.0})  # longer than T2's loose 3 s
    assert inputs(long_tight, make_ctx(long_tight, "T1"))["D_norm"] == pytest.approx(0.1 / 4.0)
