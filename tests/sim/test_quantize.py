"""Whole nanoseconds and whole bit/s (sim/quantize.py): a last-bit difference between platforms vanishes."""

from __future__ import annotations

import math

import pytest

from pcnme.sim.quantize import to_ns, to_whole_bps


@pytest.mark.parametrize("t", [0.008, 0.0123456789012, 0.72, 1.5e-3 + 1e-13, 3.0, 61.999999999])
def test_to_ns_is_within_half_a_nanosecond_and_idempotent(t: float) -> None:
    q = to_ns(t)
    assert abs(q - t) <= 0.5e-9 + 1e-18
    assert to_ns(q) == q
    assert round(q * 1e9) == pytest.approx(q * 1e9, abs=1e-6)  # a whole number of ns


@pytest.mark.parametrize("t", [0.008, 0.0123456789012, 0.72, 0.0447213595499958])
def test_one_ulp_either_way_gives_the_same_value(t: float) -> None:
    # What a platform's libm may change: the last bit.
    assert to_ns(math.nextafter(t, 0.0)) == to_ns(t) == to_ns(math.nextafter(t, 1.0))


def test_to_ns_rounds_to_the_nearest_nanosecond() -> None:
    assert to_ns(1.4e-9) == 1e-9
    assert to_ns(1.6e-9) == 2e-9
    assert to_ns(0.0123456789012) == 0.012345679


def test_to_whole_bps() -> None:
    r = 539_123_456.789
    assert to_whole_bps(r) == 539_123_457.0
    assert to_whole_bps(math.nextafter(r, 0.0)) == to_whole_bps(r)
    assert isinstance(to_whole_bps(r), float)
