"""Transitions (PLAN M4.3; final-design 11.3, 7.1.3, 7.3, 9.1; QUESTIONS Q36).

The scripted runs use two zones with every delay and run at its mean (tests/zone/harness.py), so every time is
by hand: a T1 made at t is in at t + 3 ms (radio delay) + 2.226 ms (150 KB at 539.1 Mbps from 300 m); run here
it takes 8 ms, its result reaches the car 10 ms later (7 ms through the broker to the serving zone, 3 ms
downlink), the car's confirm takes 3 ms up and the ack 7 ms back: the anchor closes it 28 ms after t_rx, and
its reward is final G = 500 ms later.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pcnme.agent.inputs import Decision
from pcnme.agent.reward import AttemptReward
from pcnme.agent.transitions import TransitionBuffer
from pcnme.car.car import Car, Task
from pcnme.ids import TaskId
from tests.zone.harness import City, Fixed, Seq, city, city_task, fixed_delay_config

UPLOAD = 0.003 + 1.2e6 / 539_116_139  # 5.226 ms: radio delay + 150 KB at R_car(300 m) = 539 116 139 bit/s
CLOSE = 0.028  # after t_rx, run here (see the module docstring)
G = 0.5


def decision(t: float, task: str, attempt: int = 1, anchor: str = "z1", action: int = 0) -> Decision:
    x = (t,) * 48  # a recognisable vector
    return Decision(
        t, anchor, task, attempt, "first", x, (None, "empty", "empty", "empty", "empty", None), action
    )


def rewards(**by_attempt: tuple[float, bool]) -> dict[int, AttemptReward]:
    return {
        int(a[1:]): AttemptReward(int(a[1:]), "wasted", r, 0.0, 0.0, d) for a, (r, d) in by_attempt.items()
    }


# --- the buffer ---------------------------------------------------------------------------------------------


def test_s_prime_is_the_anchors_next_decision_and_the_reward_comes_later() -> None:
    b = TransitionBuffer("z1")
    b.on_decision(decision(1.0, "A"))
    b.on_decision(replace(decision(1.25, "B"), masked=(None,) * 6))  # s' of A, for another task (Q36)
    assert b.replay == []  # A's reward is not final yet
    b.on_rewards("A", rewards(a1=(0.3, True)))
    (t,) = b.replay
    assert (t.task_id, t.attempt, t.r, t.dt) == ("A", 1, 0.3, pytest.approx(0.25))
    assert t.s == (1.0,) * 48 and t.s_next == (1.25,) * 48 and t.next_key == ("B", 1)
    assert t.valid == (True, False, False, False, False, True) and t.valid_next == (True,) * 6
    assert not t.truncated and list(b.pending) == [("B", 1)]


def test_a_final_reward_waits_for_s_prime() -> None:
    b = TransitionBuffer("z1")
    b.on_decision(decision(1.0, "A"))
    b.on_rewards("A", rewards(a1=(0.3, True)))
    assert b.replay == [] and ("A", 1) in b.pending
    b.on_decision(decision(2.0, "B"))
    (t,) = b.replay
    assert t.dt == pytest.approx(1.0)


def test_the_pending_buffer_is_keyed_by_task_and_attempt() -> None:
    # A re-decision of the same task is its own step (11.3.1); a rule's attempt (decided False) is not.
    b = TransitionBuffer("z1")
    b.on_decision(decision(1.0, "A", 1))
    b.on_decision(decision(1.01, "A", 2))
    b.on_decision(decision(1.02, "B", 1))
    b.on_rewards("A", rewards(a1=(-0.1, True), a2=(0.4, True), a3=(0.5, False)))
    assert [(t.attempt, t.r, t.next_key) for t in b.replay] == [(1, -0.1, ("A", 2)), (2, 0.4, ("B", 1))]
    with pytest.raises(ValueError, match="second decision"):
        b.on_decision(decision(1.03, "B", 1))
    with pytest.raises(ValueError, match="z2"):
        b.on_decision(decision(1.04, "C", anchor="z2"))


def test_the_episode_end_truncates_and_discards() -> None:
    # 11.3.4: no terminal state. The last decision has no s' (Q36); a reward not final yet: discarded.
    b = TransitionBuffer("z1")
    b.on_decision(decision(1.0, "A"))
    b.on_decision(decision(1.1, "B"))
    b.on_decision(decision(1.2, "C"))
    b.on_rewards("C", rewards(a1=(0.2, True)))
    b.end_episode()
    (t,) = b.replay
    assert (t.task_id, t.truncated, t.s_next, t.dt) == ("C", True, None, None)
    assert b.discarded == 2 and b.pending == {}


def test_a_transition_with_s_prime_still_pending_at_the_end_is_truncated_and_bootstraps() -> None:
    b = TransitionBuffer("z1")
    b.on_decision(decision(1.0, "A"))
    b.on_rewards("A", rewards(a1=(0.2, True)))
    b.on_decision(decision(1.1, "B"))  # A moves now, not truncated
    b.on_rewards("B", rewards(a1=(0.1, True)))
    b.end_episode()
    assert [(t.task_id, t.truncated) for t in b.replay] == [("A", False), ("B", True)]


# --- scripted runs with two zones (times by hand) -----------------------------------------------------------


def two(z1: object, z2: object = None) -> City:
    seq = {"z1": z1, "z2": z2 if z2 is not None else Fixed(None)}
    c = city(fixed_delay_config(), lambda z: seq[z], zone_ids=("z1", "z2"))
    c.clock.run_until(0.1)
    return c


def test_transition_timing_two_tasks_at_one_anchor_by_hand() -> None:
    # A made at 0.100 s: decided at t_rx = 0.105226; B made at 0.150 s: decided at 0.155226 = A's s',
    # Δt = 50 ms.
    # A closes at 0.105226 + 0.028 = 0.133226; its reward is final at 0.633226, when it moves to the replay.
    c = two(Seq("here", "here"))
    z1 = c.zones["z1"]
    a = city_task(c, "T1", seq=1)
    z1.upload(a)
    c.clock.run_until(0.15)
    b = city_task(c, "T1", seq=2)
    z1.upload(b)
    t_rx_a, t_rx_b = 0.1 + UPLOAD, 0.15 + UPLOAD
    c.clock.run_until(t_rx_a + CLOSE + G - 1e-6)
    d_a, d_b = z1.decisions
    assert (d_a.time, d_b.time) == (pytest.approx(t_rx_a, abs=1e-8), pytest.approx(t_rx_b, abs=1e-8))
    assert z1.transitions.replay == []
    c.clock.run_until(t_rx_a + CLOSE + G + 1e-6)
    (t,) = z1.transitions.replay
    rec = next(r for r in c.closed if r.task.task_id == a.task_id)
    assert rec.closed_at == pytest.approx(t_rx_a + CLOSE, abs=1e-8)
    assert t.task_id == a.task_id and t.dt == pytest.approx(0.05, abs=1e-12)
    assert t.s == d_a.inputs and t.s_next == d_b.inputs and t.a == 0
    assert t.r == rec.rewards[1].reward == pytest.approx(1 - 0.5 * 0.023226 / 0.1 - 0.5, abs=1e-5)
    c.clock.run_until(3.0)
    c.zones["z1"].transitions.end_episode()  # B: the last decision, truncated with no s'
    assert [(x.task_id, x.truncated, x.s_next) for x in z1.transitions.replay][1] == (b.task_id, True, None)


def test_a_re_decision_is_the_next_step_of_the_same_anchor() -> None:
    # 9.1: z2's accept reply is lost; at t_rx + 18.6 ms the anchor re-decides (attempt 2, here): Δt = 18.6 ms.
    c = two(Seq("n1", "here"))
    z1 = c.zones["z1"]
    z1.lose_reply = lambda key: key.attempt == 1
    z1.upload(city_task(c, "T1"))
    c.clock.run_until(3.0)
    z1.transitions.end_episode()
    one, two_ = z1.transitions.replay
    assert (one.attempt, one.a, one.next_key, one.truncated) == (1, 1, (one.task_id, 2), False)
    assert one.dt == pytest.approx(0.0186, abs=1e-9)
    (rec,) = c.closed
    assert (one.r, two_.r) == (rec.rewards[1].reward, rec.rewards[2].reward)
    assert two_.truncated and two_.s_next is None


def test_s_prime_never_comes_from_another_anchor() -> None:
    # 11.3.2: the same anchor's next decision. z2 decides for its own car at 0.125 s, between z1's two.
    c = two(Seq("here", "here"), Seq("here"))
    z1, z2 = c.zones["z1"], c.zones["z2"]
    car2 = Car("c02", c.cfg, c.cfg.scenarios["S2"], z2._streams, c.clock, (0.0, 0.0), z2.upload, fleet=200)
    z2.add_car(car2, 300.0)
    z1.upload(city_task(c, "T1", seq=1))
    c.clock.run_until(0.12)
    t = c.cfg.task_types["T1"]
    other = Task(TaskId("c02", 0, 1), "c02", "T1", "hard", 0.1, 0.1, t.input_bits, t.result_bits, c.clock.now)
    car2._tasks[other.task_id] = other
    z2.upload(other)
    c.clock.run_until(0.15)
    z1.upload(city_task(c, "T1", seq=2))
    c.clock.run_until(3.0)
    (d2,) = z2.decisions
    first = z1.transitions.replay[0]
    assert 0.125 < d2.time < 0.13 and first.dt == pytest.approx(0.05, abs=1e-12)
    assert first.next_key == (TaskId("c01", 0, 2), 1)


def test_boulders_all_masked_drops_and_the_soft_fallback_make_no_transition() -> None:
    # 7.1.3, 7.3, 9.1: no decision, no transition. Tight T2 is a boulder (18); Fixed(None) masks everything.
    c = city(fixed_delay_config(), lambda z: Fixed(None), zone_ids=("z1",), mode="tight", scenario="S4")
    c.clock.run_until(0.1)
    z1 = c.zones["z1"]
    z1.upload(city_task(c, "T2", seq=1, mode="tight"))
    z1.upload(city_task(c, "T1", seq=2, mode="tight"))
    c.clock.run_until(5.0)
    z1.transitions.end_episode()
    assert len(c.closed) == 2 and z1.transitions.replay == [] and z1.transitions.discarded == 0


def test_the_soft_fallback_attempt_has_no_transition() -> None:
    from tests.zone.test_placement import refuse

    c = two(Seq("n1", "n1"))
    refuse(c, "z2")
    z1 = c.zones["z1"]
    z1.upload(city_task(c, "T4"))
    c.clock.run_until(3.0)
    z1.transitions.end_episode()
    assert [t.attempt for t in z1.transitions.replay] == [1, 2]  # attempt 3 went to the cloud by rule
