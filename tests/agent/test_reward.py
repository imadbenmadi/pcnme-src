"""Rewards, one per attempt (PLAN M4.2; final-design 11.4, 12.5, 13.2, 11.3.3; QUESTIONS Q32).

By hand with the config: B = 1, w3 = 1, C_drop = 5, w = 0.5 (Q32 stand-in), ē = 11.697. Scripted two zones
(every delay and run at its mean): a T1 made at 0.1 s is in at t_rx = 0.1 + 5.226 ms; to z2 the input takes
4.2 ms and the reply 3 ms; a result reaches the car 10 ms after its run ends (7 ms to the serving zone, 3 ms
downlink).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pcnme.agent.reward import RewardParams, c_time, delivered, dropped, e_hat, settle, wasted
from pcnme.car.car import Confirm, Task
from pcnme.config import Config, load_config
from pcnme.energy import EnergyModel
from pcnme.ids import TaskId
from pcnme.node.executor import Report
from pcnme.zone.zone import AttemptInfo, TaskRecord
from tests.zone.harness import Fixed, Seq, city_task
from tests.zone.test_failures import T_RX, z1_key
from tests.zone.test_placement import refuse, two

E_BH = 0.141e-9 * 1.5
FWD_NB_T1 = E_BH * 1.2e6  # 12.2: one backhaul crossing for a 150 KB frame


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


@pytest.fixture(scope="module")
def p(cfg: Config) -> RewardParams:
    return RewardParams.from_config(cfg)


# --- the 11.4 table by hand ---------------------------------------------------------------------------------


def test_the_constants_come_from_the_config(p: RewardParams, cfg: Config) -> None:
    assert (p.b, p.w3, p.c_drop, p.w) == (1.0, 1.0, 5.0, 0.5)
    assert p.e_bar == cfg.derived.e_bar == pytest.approx(11.697, abs=1e-3)


def test_c_time_on_time_and_late() -> None:
    assert c_time(0.05, 0.1, 1.0) == pytest.approx(0.5)
    assert c_time(1.5, 1.0, 1.0) == pytest.approx(1.5 + 0.5)  # L/D + w3 (L - D)/D


def test_delivered_on_time(p: RewardParams) -> None:
    assert delivered(p, 0.05, 0.1, 1.2) == pytest.approx(1 - 0.5 * 0.5 - 0.5 * 1.2)  # 0.15


def test_delivered_soft_and_late(p: RewardParams) -> None:
    assert delivered(p, 1.5, 1.0, 1.0) == pytest.approx(1 - 0.5 * 2.0 - 0.5 * 1.0)  # -0.5


def test_the_energy_ratio_is_capped_at_e_bar(p: RewardParams) -> None:
    assert delivered(p, 0.05, 0.1, 50.0) == pytest.approx(1 - 0.25 - 0.5 * p.e_bar)
    assert wasted(p, 0.0, 0.1, 50.0) == pytest.approx(-0.5 * p.e_bar)
    assert e_hat(100.0, 2.0, p.e_bar) == p.e_bar and e_hat(1.0, 2.0, p.e_bar) == 0.5


def test_wasted(p: RewardParams) -> None:
    assert wasted(p, 0.02, 0.1, 0.0007) == pytest.approx(-(0.5 * 0.2 + 0.5 * 0.0007))


def test_dropped(p: RewardParams) -> None:
    assert dropped(p) == pytest.approx(1 - 0.5 * 5 - 0.5 * p.e_bar)  # about -7.35


# --- N18: every drop reward is lower than every delivery reward ---------------------------------------------


@settings(max_examples=500, deadline=None)
@given(
    w=st.floats(1e-9, 1.0),
    kind=st.sampled_from(["T1", "T3", "T4", "T2"]),
    mode=st.sampled_from(["loose", "tight"]),
    frac=st.floats(1e-6, 1.0),
    energy=st.floats(0.0, 1e6),
)
def test_every_drop_is_worse_than_every_delivery(
    cfg: Config, w: float, kind: str, mode: str, frac: float, energy: float
) -> None:
    # 11.4, N18: for any w in (0, 1] and any energy. A delivered hard task has L <= D, a soft one
    # L <= D + L_max.
    # (w starts at 1e-9: below that the gap, w (C_drop - c_time), is under float resolution next to ē.)
    p = RewardParams(cfg.reward.b, cfg.reward.w3, cfg.reward.c_drop, w, cfg.derived.e_bar)
    t = cfg.task_types[kind]
    d = t.deadline[mode]  # type: ignore[index]
    longest = d if t.cls == "hard" else d + t.l_max[mode]  # type: ignore[index]
    ratio = energy / cfg.derived.e_ref[kind]
    assert dropped(p) < delivered(p, frac * longest, d, ratio)


# --- per attempt in a run (12.5, 11.3.3) --------------------------------------------------------------------


def settled(c: object, wait: float = 1.0) -> TaskRecord:
    c.clock.run_until(c.clock.now + wait)  # type: ignore[attr-defined]
    (rec,) = c.closed  # type: ignore[attr-defined]
    assert rec.closed_at is not None and c.clock.now >= rec.closed_at + 0.5  # type: ignore[attr-defined]
    return rec  # type: ignore[no-any-return]


def test_a_task_run_here_on_time(cfg: Config, p: RewardParams) -> None:
    # L = upload 5.226 + run 8 + return 10 = 23.226 ms; E = PUE p T_run + the fog return = E_ref(T1).
    c = two(Fixed("here"))
    c.zones["z1"].upload(city_task(c, "T1"))
    rec = settled(c)
    (r,) = rec.rewards.values()
    assert rec.confirm is not None and rec.confirm.latency == pytest.approx(0.023226, abs=1e-6)
    assert (r.kind, r.decided, r.t_waste) == ("delivered", True, None)
    assert r.energy == pytest.approx(cfg.derived.e_ref["T1"])
    assert r.reward == pytest.approx(1 - 0.5 * rec.confirm.latency / 0.1 - 0.5 * 1.0)


def test_a_refused_attempt_wastes_its_time_and_forwarding(cfg: Config, p: RewardParams) -> None:
    # 9.1: attempt 1 to z2 is refused 4.2 + 3 = 7.2 ms after its send; attempt 2 runs here.
    c = two(Seq("n1", "here"))
    refuse(c, "z2")
    c.zones["z1"].upload(city_task(c, "T1"))
    rec = settled(c)
    one, two_ = rec.rewards[1], rec.rewards[2]
    assert (one.kind, one.t_waste) == ("wasted", pytest.approx(0.0072))
    assert one.energy == pytest.approx(FWD_NB_T1)  # 12.5: refused = forwarding only
    assert one.reward == pytest.approx(-(0.5 * 0.072 + 0.5 * FWD_NB_T1 / cfg.derived.e_ref["T1"]))
    assert rec.confirm is not None and rec.confirm.latency == pytest.approx(0.030426, abs=1e-6)
    assert two_.kind == "delivered"
    assert two_.reward == pytest.approx(1 - 0.5 * rec.confirm.latency / 0.1 - 0.5)  # E = E_ref here


def test_a_timed_out_attempt_whose_copy_the_car_kept_is_the_delivered_one(cfg: Config) -> None:
    # 11.4: "the attempt delivered first is the one in the car's confirm". z2's accept reply is lost: attempt
    # 1 times out at t_rx + 18.6 ms and attempt 2 runs here; z2's result reaches the car first and wins.
    # Attempt 2 is cancelled at its first result (19.2 ms): wasted for 0.6 ms, with the execution it spent.
    c = two(Seq("n1", "here"))
    z1 = c.zones["z1"]
    z1.lose_reply = lambda key: key.attempt == 1
    z1.upload(city_task(c, "T1"))
    rec = settled(c)
    assert rec.confirm is not None and rec.confirm.attempt == 1
    one, two_ = rec.rewards[1], rec.rewards[2]
    assert one.kind == "delivered"
    assert one.energy == pytest.approx(FWD_NB_T1 + cfg.derived.e_ref["T1"])  # forwarding + run + return
    assert two_.kind == "wasted" and two_.t_waste == pytest.approx(0.0006, abs=1e-9)
    info = rec.attempts[2]
    assert info.report is not None and info.report.outcome == "cancelled"
    assert two_.energy == pytest.approx(info.report.energy) and two_.energy > 0  # aborted while running
    assert z1.executor.status(z1_key(rec, 2)) == "cancelled"


def test_a_hard_task_refused_twice_ends_on_the_drop_reward(p: RewardParams) -> None:
    c = two(Seq("n1", "n1"))
    refuse(c, "z2", "late")
    c.zones["z1"].upload(city_task(c, "T1"))
    rec = settled(c)
    assert rec.outcome == "drop_refused_late"
    assert rec.rewards[1].kind == "wasted"
    assert (rec.rewards[2].kind, rec.rewards[2].reward) == ("dropped", pytest.approx(dropped(p)))
    assert rec.rewards[2].energy == pytest.approx(FWD_NB_T1)  # logged, not in the reward


def test_the_soft_fallback_to_the_cloud_is_rewarded_but_not_decided() -> None:
    # 9.1: refused twice, the soft task goes to the cloud by rule (attempt 3): no decision made it.
    c = two(Seq("n1", "n1"))
    refuse(c, "z2")
    c.zones["z1"].upload(city_task(c, "T4"))
    rec = settled(c, 3.0)
    assert [r.decided for r in rec.rewards.values()] == [True, True, False]
    assert [r.kind for r in rec.rewards.values()] == ["wasted", "wasted", "delivered"]


def test_a_task_dropped_with_every_option_masked_logs_the_drop_reward(p: RewardParams) -> None:
    c = two(Fixed(None))
    c.zones["z1"].upload(city_task(c, "T1"))
    rec = settled(c)
    assert rec.outcome == "drop_masked" and rec.rewards == {} and rec.drop_reward == pytest.approx(dropped(p))


def test_rewards_are_final_at_close_plus_g_with_the_late_energy_report(p: RewardParams) -> None:
    # 11.3.3, 12.5: closing cancels the T4 running at z2 (6E.15); the abort's report (energy spent) reaches
    # z1 after the close, within G = 500 ms, and is in the attempt's energy. The drop reward ignores it.
    c = two(Fixed("n1"))
    z1 = c.zones["z1"]
    z1.upload(city_task(c, "T4"))
    c.clock.run_until(T_RX + 0.010)  # accepted and running at z2
    rec = next(iter(z1.records.values()))
    z1._close(rec, "drop_test")
    closed = c.clock.now
    c.clock.run_until(closed + 0.499)
    assert rec.rewards == {} and rec.attempts[1].report is not None  # the report is in, not yet final
    c.clock.run_until(closed + 0.501)
    (r,) = rec.rewards.values()
    report = rec.attempts[1].report
    assert report.outcome == "cancelled" and report.energy > 0
    assert r.energy == pytest.approx(FWD_NB_T1 + report.energy)
    assert (r.kind, r.reward) == ("dropped", pytest.approx(dropped(p)))


def test_a_report_after_close_plus_g_is_not_counted() -> None:
    c = two(Fixed("n1"), timing__reward_grace_G=0.001)
    z1 = c.zones["z1"]
    z1.upload(city_task(c, "T4"))
    c.clock.run_until(T_RX + 0.010)
    rec = next(iter(z1.records.values()))
    z1._close(rec, "drop_test")  # the abort's report needs about 10 ms to come back
    c.clock.run_until(c.clock.now + 0.5)
    assert rec.attempts[1].report is not None  # it came, after G
    assert rec.rewards[1].energy == pytest.approx(FWD_NB_T1)


# --- settle on a record (the rule for the last attempt, the confirm's anchor, republishes) ------------------


def record(cfg: Config, outcome: str, confirm: Confirm | None, republishes: int = 0) -> TaskRecord:
    t = cfg.task_types["T1"]
    task = Task(TaskId("c01", 0, 1), "c01", "T1", "hard", 0.1, 0.1, t.input_bits, t.result_bits, 0.0)
    rec = TaskRecord(task=task, anchor="z1", outcome=outcome, closed_at=0.09, republishes=republishes)
    for a, opt in ((1, "n1"), (2, "here")):
        key = z1_key(rec, a)
        rep = Report(key, "T1", "done", "gpu", 0.01 * a, 0.01 * a + 0.008, 0.008, 0.3)
        rec.attempts[a] = AttemptInfo(opt, "z2" if a == 1 else "z1", 0.01 * a, report=rep, t_end=0.05)  # type: ignore[arg-type]
    rec.confirm = confirm
    return rec


def test_the_kept_copy_pays_one_return_per_publish(cfg: Config, p: RewardParams) -> None:
    em = EnergyModel(cfg)
    c = Confirm(TaskId("c01", 0, 1), "z1", 2, 0.04)
    one = settle(record(cfg, "on_time", c), "z1", p, em)[2]
    three = settle(record(cfg, "on_time", c, republishes=2), "z1", p, em)[2]
    assert three.energy - one.energy == pytest.approx(2 * em.ret("fog", 3 * 8000))


def test_a_confirm_for_another_anchor_scores_this_anchors_attempts_as_not_kept(
    cfg: Config, p: RewardParams
) -> None:
    # 9.7.4: the car kept the other anchor's copy; here no ack came: a drop, the last attempt dropped.
    em = EnergyModel(cfg)
    r = settle(record(cfg, "drop_no_ack", Confirm(TaskId("c01", 0, 1), "z3", 1, 0.04)), "z1", p, em)
    assert (r[1].kind, r[2].kind) == ("wasted", "dropped")
    assert r[1].t_waste == pytest.approx(0.05 - 0.01)


def test_a_hard_result_after_d_is_a_drop_on_the_kept_attempt(cfg: Config, p: RewardParams) -> None:
    # Q15: a hard L > D counts as a drop; the kept copy is the last attempt, even when it is not the highest.
    em = EnergyModel(cfg)
    r = settle(record(cfg, "drop_late", Confirm(TaskId("c01", 0, 1), "z1", 1, 0.12)), "z1", p, em)
    assert (r[1].kind, r[2].kind) == ("dropped", "wasted")


def test_the_params_follow_the_config() -> None:
    from tests.zone.harness import fixed_delay_config

    p = RewardParams.from_config(fixed_delay_config(reward__w=0.8, reward__C_drop=6.0, reward__B=2.0))
    assert (p.w, p.c_drop, p.b) == (0.8, 6.0, 2.0)
    assert dropped(p) == pytest.approx(2.0 - 0.8 * 6.0 - 0.2 * p.e_bar)


def test_the_anchor_counts_its_republishes() -> None:
    # 9.5.4: the result came but no ack within T_ack (200 ms): the anchor republishes once (12.2: each
    # republish pays the return again).
    c = two(Fixed("here"))
    c.bus.faults.drop = lambda topic, key, sub: topic == "car/c01/ack"
    z1 = c.zones["z1"]
    z1.upload(city_task(c, "T1"))
    c.clock.run_until(c.clock.now + 0.25)
    (rec,) = z1.records.values()
    assert rec.republished and rec.republishes == 1
