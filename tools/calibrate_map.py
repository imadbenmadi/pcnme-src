"""Calibrate the dev map once (final-design 16.2, 18; QUESTIONS Q29, Q30; tag C).

Prints, per mobility, the share of task creations with coverage (and z1's share), from a positions-only run
of the smallest fleet. The values go into config/default.yaml under cars.calibrated.
Usage: python tools/calibrate_map.py
"""

from __future__ import annotations

from pathlib import Path

from pcnme.config import load_config
from pcnme.experiments.calibrate import calibrate

cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
cal = cfg.cars.calibration
for mobility in ("low", "high"):
    r = calibrate(cfg, mobility, cal.seconds, cal.seed, cfg.cars.count[0])
    print(mobility, {k: round(v, 4) for k, v in r.items()})
