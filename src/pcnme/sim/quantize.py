"""Whole nanoseconds and whole bits per second: one seed gives one run on every OS (CLAUDE.md, Determinism).

log, exp, log1p, ndtri, sin and cos come from the platform's maths library and may differ in the last bit
between Windows and Linux; one bit can flip a scheduling comparison and change the whole event log. Every
value drawn or computed with them is rounded once, where it is made: run times, link delays and arrival
gaps to whole nanoseconds, radio rates to whole bit/s. A last-bit difference then changes nothing unless the
value lies within a few ulps of a rounding boundary (about one draw in 10^8). Engineering choice: no design
quantity is near 1 ns or 1 bit/s, and every later sum is plain IEEE arithmetic, the same everywhere.
"""

from __future__ import annotations

_NS_PER_S = 1e9


def to_ns(seconds: float) -> float:
    """`seconds` rounded to a whole number of nanoseconds."""
    return round(seconds * _NS_PER_S) / _NS_PER_S


def to_whole_bps(rate: float) -> float:
    """`rate` (bit/s) rounded to a whole number of bits per second."""
    return float(round(rate))
