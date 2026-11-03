"""LuST trace generation (PLAN LuST): a few seconds of real SUMO traffic through to a replayable trace file.
Needs the LuST download (data/maps/lust/, data/DATA.md) and the `maps` extra; skipped without them."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcnme.maps import map_config, trace_source

pytestmark = pytest.mark.slow


def test_a_short_sumo_run_gives_a_trace_inside_the_crop(tmp_path: Path) -> None:
    pytest.importorskip("sumo")
    cfg = map_config("lust")
    scenario = trace_source(cfg)
    if not (scenario / "lust.net.xml").exists():
        pytest.skip(f"no LuST scenario at {scenario}")
    import numpy as np

    from pcnme.mobility.lust import make_trace

    out = tmp_path / "peak.npz"
    stats = make_trace(cfg, scenario, "peak", out, tmp_path / "work", trace_s=10, warmup_s=60)
    assert stats["vehicles"] > 0 and stats["edges"] > 0
    d = np.load(out)
    side = cfg.map.crop_m
    assert d["x"].min() >= 0 and d["x"].max() <= side and d["y"].min() >= 0 and d["y"].max() <= side
    assert np.all(np.diff(d["vehicle"]) >= 0)  # sorted by vehicle, then time
