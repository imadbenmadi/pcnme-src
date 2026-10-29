"""The knee point and the weight w on synthetic fronts with known answers (PLAN M5.3, M5 acceptance;
final-design 13.3)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pcnme.expert.knee import knee, w_star, weights


def test_a_convex_front_by_hand() -> None:
    # (0, 1), (0.2, 0.2), (0.4, 0.1), (1, 0): already normalised (R = 1). Sums 1, 0.4, 0.5, 1: the knee is
    # (0.2, 0.2).
    # Neighbours (0, 1) and (0.4, 0.1): s = (1 - 0.1) / (0.4 - 0) = 2.25, w̃ = 2.25 / 3.25; R1 = R2: w = w̃.
    k = knee([(0.4, 0.1), (1.0, 0.0), (0.0, 1.0), (0.2, 0.2)])
    assert k is not None and k.point == (0.2, 0.2) and k.index == 1
    assert k.slope == pytest.approx(2.25) and k.w_tilde == pytest.approx(2.25 / 3.25)
    assert k.w == pytest.approx(2.25 / 3.25) and k.ranges == (1.0, 1.0)


def test_the_weight_goes_back_through_the_ranges() -> None:
    # The same shape with f1 x 10 (R1 = 10, R2 = 1): w̃ = 0.5 at the middle knee (s = 1), and
    # w = (0.5 / 10) / (0.5 / 10 + 0.5 / 1) = 1 / 11: a unit of f1 is worth a tenth of a unit of f2.
    k = knee([(0.0, 1.0), (1.0, 0.3), (10.0, 0.0)])
    assert k is not None and k.point == (1.0, 0.3)
    assert k.slope == pytest.approx(1.0) and k.w_tilde == pytest.approx(0.5)
    assert k.w == pytest.approx(1 / 11)


def test_shifting_the_front_changes_nothing() -> None:
    # Normalisation removes the minima.
    a = knee([(0.0, 1.0), (0.2, 0.2), (0.4, 0.1), (1.0, 0.0)])
    b = knee([(3.0, 6.0), (3.2, 5.2), (3.4, 5.1), (4.0, 5.0)])
    assert a is not None and b is not None and (a.index, a.w) == (b.index, pytest.approx(b.w))


def test_a_knee_at_an_end_uses_its_one_neighbour() -> None:
    # (0, 0.2) has the smallest sum (0.2): s from it to (0.5, 0.1): (0.2 - 0.1) / 0.5 on normalised axes.
    pts = [(0.0, 0.2), (0.5, 0.1), (1.0, 0.0)]  # R1 = 1, R2 = 0.2
    k = knee(pts)
    assert k is not None and k.index == 0
    # normalised: (0, 1), (0.5, 0.5), (1, 0): sums 1, 1, 1: a tie, the smaller f̃1 wins
    assert k.slope == pytest.approx((1.0 - 0.5) / 0.5)
    assert k.w == pytest.approx((0.5 / 1.0) / (0.5 / 1.0 + 0.5 / 0.2))


def test_two_points_and_duplicates() -> None:
    k = knee([(1.0, 0.0), (0.0, 1.0), (0.0, 1.0)])
    assert k is not None and k.index == 0 and k.slope == pytest.approx(1.0) and k.w == pytest.approx(0.5)
    assert knee([(0.3, 0.3), (0.3, 0.3)]) is None and knee([]) is None
    assert weights([(0.3, 0.3)]) == {}


def test_every_point_gets_its_own_weight() -> None:
    w = weights([(0.0, 1.0), (0.2, 0.2), (0.4, 0.1), (1.0, 0.0)])
    assert list(w) == [(0.0, 1.0), (0.2, 0.2), (0.4, 0.1), (1.0, 0.0)]
    assert w[(0.0, 1.0)] == pytest.approx(4 / 5)  # one-sided: s = (1 - 0.2) / 0.2 = 4
    assert w[(0.2, 0.2)] == pytest.approx(2.25 / 3.25)
    assert w[(1.0, 0.0)] == pytest.approx(1 / 7)  # s = (0.1 - 0) / (1 - 0.4) = 1/6, w̃ = 1/7
    vals = list(w.values())
    assert vals == sorted(vals, reverse=True)  # the slope flattens along a convex front


def test_w_star_is_the_median_knee_weight() -> None:
    fronts = [[(0.0, 1.0), (1.0, 0.3), (10.0, 0.0)], [(0.0, 1.0), (0.2, 0.2), (0.4, 0.1), (1.0, 0.0)]]
    ks = [knee(f) for f in fronts] + [None, knee([(0.0, 1.0), (1.0, 0.0)])]
    assert w_star(ks) == pytest.approx(0.5)  # median of 1/11, 0.692, 0.5
    with pytest.raises(ValueError):
        w_star([None])


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.tuples(*[st.integers(0, 10_000).map(lambda v: v / 1000)] * 2), min_size=2, max_size=30).map(
        lambda ps: [p for p in ps if not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in ps)]
    )
)
def test_on_any_front_w_is_in_0_1_and_the_knee_has_the_smallest_sum(front: list[tuple[float, float]]) -> None:
    k = knee(front)
    if len(set(front)) < 2:
        assert k is None
        return
    assert k is not None and 0.0 < k.w < 1.0 and 0.0 < k.w_tilde < 1.0
    lo1, hi1 = min(p[0] for p in front), max(p[0] for p in front)
    lo2, hi2 = min(p[1] for p in front), max(p[1] for p in front)

    def total(p: tuple[float, float]) -> float:
        return (p[0] - lo1) / (hi1 - lo1) + (p[1] - lo2) / (hi2 - lo2)

    assert total(k.point) <= min(total(p) for p in front) + 1e-12


def test_per_point_weights_go_back_through_the_ranges_too() -> None:
    w = weights([(0.0, 1.0), (1.0, 0.3), (10.0, 0.0)])  # R1 = 10, R2 = 1
    assert w[(1.0, 0.3)] == pytest.approx(1 / 11)


def test_a_set_with_no_range_on_one_objective_is_not_a_front() -> None:
    with pytest.raises(ValueError, match="range"):
        knee([(0.0, 1.0), (0.0, 2.0)])
