"""The mobility predictor (PLAN M3.9; final-design 10.2, C3, C8; QUESTIONS Q27). Hand calculations."""

from __future__ import annotations

import math

import pytest

from pcnme.mobility.predictor import candidates, exit_time, next_zone

C = {"z1": (300.0, 300.0), "z2": (900.0, 300.0), "z3": (300.0, 900.0), "z4": (900.0, 900.0)}
R = 350.0


def test_exit_time_from_the_centre_is_radius_over_speed() -> None:
    # b = 0, |p - c| = 0: T = sqrt(v^2 R^2) / v^2 = R / v = 350 / 10 = 35 s.
    assert exit_time((300.0, 300.0), (10.0, 0.0), C["z1"], R) == pytest.approx(35.0)


def test_exit_time_off_centre_by_hand() -> None:
    # p - c = (100, 0), v = (0, 10): b = 0, T = sqrt(100 (350^2 - 100^2)) / 100 = sqrt(112500) / 10 = 33.54 s.
    assert exit_time((400.0, 300.0), (0.0, 10.0), C["z1"], R) == pytest.approx(math.sqrt(112500) / 10)
    # heading out along x from x = 400: b = 100 x 10 = 1000; T = (-1000 + sqrt(1e6 + 100 (122500 - 10000)))
    # / 100 = (-1000 + 3500) / 100 = 25 s (it has 250 m left).
    assert exit_time((400.0, 300.0), (10.0, 0.0), C["z1"], R) == pytest.approx(25.0)
    # heading back in: b = -1000: (1000 + 3500) / 100 = 45 s (450 m to the far edge).
    assert exit_time((400.0, 300.0), (-10.0, 0.0), C["z1"], R) == pytest.approx(45.0)


def test_exit_time_is_infinite_when_stopped() -> None:
    # C8: speed near 0 -> T_exit = infinity: running here is safe.
    assert exit_time((400.0, 300.0), (0.0, 0.0), C["z1"], R) == math.inf
    assert next_zone((400.0, 300.0), (0.0, 0.0), "z1", C, R) is None


def test_next_zone_going_straight() -> None:
    # East from z1's centre: leaves z1 at x = 650, already inside z2 (overlap 550-650): z2.
    assert next_zone((300.0, 300.0), (10.0, 0.0), "z1", C, R) == "z2"
    assert next_zone((300.0, 300.0), (0.0, 10.0), "z1", C, R) == "z3"
    # West from z2 into z1; north from z2 into z4.
    assert next_zone((900.0, 300.0), (-10.0, 0.0), "z2", C, R) == "z1"
    assert next_zone((900.0, 300.0), (0.0, 10.0), "z2", C, R) == "z4"


def test_next_zone_into_a_gap_is_the_nearest_other_centre_to_the_exit_point() -> None:
    # Q27: west from z1's centre leaves at x = -50 (off the map, no circle there). The nearest other centre
    # to (-50, 300): z3 at 694.6 m (z2 950 m, z4 1171.1 m).
    assert next_zone((300.0, 300.0), (-10.0, 0.0), "z1", C, R) == "z3"


def test_at_an_intersection_the_roads_on_offer_give_the_candidates_straight_on_first() -> None:
    # At (400, 200) heading east, with roads east, north and south on offer (a T: no U-turn here): east
    # leaves into z2; north leaves z1 at y = 300 + 335.4 into z3; south leaves at y = -35.4: the nearest
    # other centre to (400, -35) is z2.
    roads = [(1, 0), (0, 1), (0, -1)]
    got = candidates((400.0, 200.0), 10.0, roads, (1, 0), "z1", C, R)
    assert got[0] == "z2" and set(got) == {"z2", "z3"}


def test_a_circle_the_exit_point_is_already_in_comes_before_one_entered_later() -> None:
    # East from (0, 0) (radius 350) leaves at (350, 0), already inside "a" (250 m from its centre); "c" at
    # (1000, 0) is entered only at x = 650. The next zone is "a" (entered first, at the exit).
    centres = {"z0": (0.0, 0.0), "a": (500.0, 200.0), "c": (1000.0, 0.0)}
    assert next_zone((0.0, 0.0), (10.0, 0.0), "z0", centres, R) == "a"


def test_straight_on_comes_first_heading_north() -> None:
    # At (400, 200) heading north with east and south also on offer: straight on (north) -> z3, then z2.
    got = candidates((400.0, 200.0), 10.0, [(1, 0), (0, 1), (0, -1)], (0, 1), "z1", C, R)
    assert got == ["z3", "z2"]
