"""The 16.3 metrics computed from existing logs (src/pcnme/metrics/extra.py; PLAN "Missing metrics")."""

from __future__ import annotations

from typing import Any

import pytest

from pcnme.metrics.extra import (
    decisions_until_stable,
    front_hypervolumes,
    hypervolume_2d,
    predictor_errors,
    zone_change_delivery,
)

CENTRES = {"z1": (0.0, 0.0), "z2": (200.0, 0.0), "z3": (0.0, 200.0)}
R = 120.0


def _track(
    car: str, xs: list[float], zones: list[str], v: float = 10.0, dt: float = 1.0
) -> dict[str, list[Any]]:
    n = len(xs)
    return {"time_s": [i * dt for i in range(n)], "car_id": [car] * n, "x_m": xs, "y_m": [0.0] * n,
            "speed_mps": [v] * n, "heading_x": [1.0] * n, "heading_y": [0.0] * n, "zone": zones}  # fmt: skip


def test_a_car_driving_straight_east_is_predicted_into_z2_and_its_exit_time_is_exact() -> None:
    """10.2: going straight east from z1 the path enters z2's circle next; T_exit = (R - x) / v on the axis.
    The zone column switches when the car crosses z1's circle (x = 120 m)."""
    xs = [float(x) for x in range(0, 200, 10)]  # 10 m/s, one row per second
    zones = ["z1" if x < 120 else "z2" for x in xs]
    got = predictor_errors(_track("c1", xs, zones), CENTRES, R)
    assert got["acc1"] == 1.0 and got["predictions"] == 12  # samples in z1 (x = 0..110)
    assert got["t_exit_mae_s"] == pytest.approx(0.0, abs=1e-9)  # exits at x = 120 exactly on a whole second


def test_a_wrong_prediction_counts_against_acc1_and_a_late_exit_shows_as_error() -> None:
    xs = [float(x) for x in range(0, 200, 10)]
    zones = ["z1" if x < 140 else "z3" for x in xs]  # it really went to z3 (say it turned), and 2 s late
    got = predictor_errors(_track("c1", xs, zones), CENTRES, R)
    assert got["acc1"] == 0.0
    # x = 0..110: predicted exit at 120 m, real at 140 m: 2 s each; x = 120, 130 are already outside the
    # circle (predicted 0 s), real exits 2 s and 1 s later
    assert got["t_exit_mae_s"] == pytest.approx((12 * 2 + 2 + 1) / 14)


def test_a_stopped_car_or_a_car_in_a_gap_makes_no_prediction() -> None:
    track = _track("c1", [0.0, 0.0, 0.0], ["z1", "z1", "z1"], v=0.0)
    assert predictor_errors(track, CENTRES, R)["predictions"] == 0
    gap = _track("c2", [500.0, 510.0], ["", ""])
    assert predictor_errors(gap, CENTRES, R)["predictions"] == 0


def test_delivery_after_a_zone_change_counts_only_tasks_whose_car_changed_zone() -> None:
    pos = _track("c1", [0.0, 50.0, 130.0, 150.0], ["z1", "z1", "z2", "z2"])
    tasks = {"task_id": ["c1.0.1", "c1.0.2", "c1.0.3"], "time_s": [0.0, 0.5, 2.0],
             "outcome": ["on_time", "drop_late", "on_time"], "latency_s": [2.5, None, 0.2],
             "deadline_s": [3.0, 2.0, 1.0]}  # fmt: skip
    got = zone_change_delivery(tasks, pos)
    # task 1: z1 at 0 s, z2 at 2.5 s (changed, on time); task 2: z1 at 0.5, z2 at 2.5 (changed, dropped);
    # task 3: z2 -> z2 (no change)
    assert got == {"tasks": 2, "delivered": 0.5, "on_time": 0.5}


def test_decisions_until_stable_is_where_the_rolling_on_time_share_stays_at_the_level() -> None:
    outcomes = ["drop_late"] * 10 + ["on_time"] * 30
    tasks = {"time_s": [float(i) for i in range(40)], "outcome": outcomes}
    decisions = [i + 0.5 for i in range(40)]  # one decision just after each task
    # window 10, level 0.9: the window ending at task 18 (9..18) holds 9 on time -> first good window
    assert decisions_until_stable(tasks, decisions, window=10, level=0.9) == 18
    worse = {"time_s": tasks["time_s"], "outcome": ["on_time"] * 35 + ["drop_late"] * 5}
    assert decisions_until_stable(worse, decisions, window=10, level=0.9) is None  # not stable at the end
    assert decisions_until_stable({"time_s": [0.0], "outcome": ["on_time"]}, [0.1], window=10) is None


def test_the_hypervolume_of_a_front_is_the_area_it_dominates_up_to_the_reference() -> None:
    assert hypervolume_2d([(1.0, 3.0), (2.0, 1.0)], (4.0, 4.0)) == pytest.approx(1 * 1 + 2 * 3)
    assert hypervolume_2d([(1.0, 3.0), (2.0, 1.0), (3.0, 2.0)], (4.0, 4.0)) == pytest.approx(7.0)  # dominated
    assert hypervolume_2d([(5.0, 5.0)], (4.0, 4.0)) == 0.0  # beyond the reference


def test_front_hypervolumes_share_one_reference_per_scenario_and_compare_with_greedy() -> None:
    windows = [
        {"scenario": "S1", "index": "0", "F": [[1.0, 3.0], [2.0, 1.0]], "greedy": [2.0, 3.0, 0]},
        {"scenario": "S1", "index": "1", "F": [[1.5, 2.0]], "greedy": [3.0, 2.5, 0]},
    ]
    rows = front_hypervolumes(windows)
    assert [r["window"] for r in rows] == [0, 1]
    assert rows[0]["ref_f1"] == pytest.approx(3.3) and rows[0]["ref_f2"] == pytest.approx(3.3)
    assert rows[0]["hv"] > rows[0]["hv_greedy"] and rows[0]["hv_ratio"] > 1.0
