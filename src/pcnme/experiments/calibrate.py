"""Map calibration runs (final-design 16.2, 18; QUESTIONS Q29, Q30; tag C).

A short positions-only run of the fleet on the map (no tasks): every GPS period each car's position is
sampled; a sample is covered when the car is in some active zone's circle, and attached to the zone the
attach rule (with hysteresis, C13) picks. From it:
- the hotspot share h: set so that z1 holds hotspot_share of the covered samples (Q30);
- coverage = the share of covered samples at that h. Task creations are Poisson, the same rate for every
  car, so this is the share of creations with coverage, and lambda_car is scaled by 1 / coverage (Q29).
Run once per map and mobility by tools/calibrate_map.py; the results are config values (cars.calibrated).
"""

from __future__ import annotations

from pcnme.config import Config
from pcnme.experiments.city import choose_zone
from pcnme.mobility.grid import RoadGrid, Walker
from pcnme.sim.clock import EventClock
from pcnme.sim.rng import RandomStreams


def measure(
    cfg: Config, mobility: str, h: float, seconds: float, seed: int, fleet: int
) -> tuple[float, float]:
    """(coverage, z1's share of the covered samples) for hotspot share h."""
    clock = EventClock()
    streams = RandomStreams.from_config(cfg, seed, "training")
    lay = cfg.layout
    centres = {z: lay.zone_positions_m[z] for z in lay.zones if z not in lay.joining_zones}
    grid = RoadGrid.from_area(lay.area_m, cfg.cars.grid.block_m)
    hot = grid.within(lay.zone_positions_m["z1"], lay.zone_radius_m)
    speeds = cfg.cars.speed_low_mps if mobility == "low" else cfg.cars.speed_high_mps
    n_hot = round(h * fleet)
    walkers = [
        Walker(
            clock,
            grid,
            streams.generator("mobility", f"c{i:03d}"),
            speeds,
            cfg.cars.grid,
            hot if i < n_hot else None,
        )
        for i in range(fleet)
    ]
    serving: list[str | None] = [None] * fleet
    covered = z1 = total = 0
    t, period = 0.0, cfg.cars.gps_period
    while t <= seconds:
        clock.run_until(t)
        for i, w in enumerate(walkers):
            serving[i] = choose_zone(
                w.position(t), serving[i], centres, lay.zone_radius_m, lay.handover_hysteresis_m
            )
            total += 1
            if serving[i] is not None:
                covered += 1
                z1 += serving[i] == "z1"
        t = round(t + period, 9)
    return covered / total, z1 / max(1, covered)


def calibrate(
    cfg: Config, mobility: str, seconds: float, seed: int, fleet: int, steps: int = 12
) -> dict[str, float]:
    """Q30: the hotspot share h by bisection so that z1 holds hotspot_share of the covered samples (z1 =
    1.6 x the mean at 40 % over 4 zones); Q29: the coverage share at that h (hotspot cars are always
    covered, so it depends on h)."""
    target = cfg.cars.hotspot_share
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        mid = (lo + hi) / 2
        _, share = measure(cfg, mobility, mid, seconds, seed, fleet)
        lo, hi = (mid, hi) if share < target else (lo, mid)
    h = round(hi * fleet) / fleet  # whole cars
    coverage, share = measure(cfg, mobility, h, seconds, seed, fleet)
    n = len([z for z in cfg.layout.zones if z not in cfg.layout.joining_zones])
    return {"coverage": coverage, "hotspot_h": h, "z1_share": share, "z1_over_mean": share * n}
