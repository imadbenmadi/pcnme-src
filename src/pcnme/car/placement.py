"""Where a car is while M2 has no mobility. TEST-ONLY STAND-IN (QUESTIONS Q13): M3.8 replaces it.

The design places cars by mobility (VFC grid or LuST traces, 10, 16.2). Until then each car stays at one
point, uniform over its zone's coverage disk, drawn from the car's own mobility stream.
"""

from __future__ import annotations

import math

from pcnme.sim.rng import RandomStreams


def standin_uniform_disk(
    streams: RandomStreams, car_id: str, centre: tuple[float, float], radius: float
) -> tuple[float, float]:
    """STAND-IN (Q13): one point uniform over the disk (r = R sqrt(u) gives a uniform density)."""
    g = streams.generator("mobility", car_id)
    r = radius * math.sqrt(float(g.random()))
    angle = 2 * math.pi * float(g.random())
    return centre[0] + r * math.cos(angle), centre[1] + r * math.sin(angle)
