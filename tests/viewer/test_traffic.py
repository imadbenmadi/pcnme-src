"""A trace map's traffic preview and the design status (src/pcnme/viewer/sources/traffic.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pcnme.paths import MapPaths
from pcnme.viewer.sources.base import SourceInfo
from pcnme.viewer.sources.replay import ParquetSource, frames
from pcnme.viewer.sources.traffic import PREVIEW_S, MapSource, status


@pytest.fixture()
def lust(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MapSource:
    """LuST with one tiny recording (two vehicles) and its roads cached: no SUMO needed."""
    monkeypatch.setenv("PCNME_DATA_DIR", str(tmp_path))
    traces = MapPaths("lust").traces
    traces.mkdir(parents=True)
    t = np.array([900.0, 900.2, 900.4, 900.2, 900.4])
    np.savez_compressed(traces / "peak.npz", t=t, vehicle=np.array([0, 0, 0, 1, 1]),
                        x=np.array([10.0, 12.0, 14.0, 500.0, 501.0], np.float32),
                        y=np.array([20.0, 20.0, 20.0, 600.0, 602.0], np.float32))  # fmt: skip
    (traces / "roads.json").write_text(json.dumps([[[0, 20], [1200, 20]]]), encoding="utf-8")
    info = SourceInfo("lust", "replay", "LuST", map="lust")
    return MapSource(info, ParquetSource(info, "lust", [MapPaths("lust").runs]))


def test_each_recording_is_a_traffic_run_with_roads_and_no_tasks(lust: MapSource) -> None:
    runs = lust.runs()
    assert [(r.run_id, r.scenario, r.mode) for r in runs] == [("traffic-peak", "peak", "traces")]
    rp = lust.replay("traffic-peak")
    assert rp["meta"]["mode"] == "traces" and rp["meta"]["fleet"] == 2 and "No model" in rp["meta"]["note"]
    assert rp["layout"]["roads"] == [[[0, 20], [1200, 20]]] and rp["layout"]["north_up"]
    assert rp["tasks"]["id"] == [] and rp["attempts"]["task"] == []
    xs = rp["cars"]["x"]
    assert len(xs) == 2 and all(len(x) == frames(PREVIEW_S) for x in xs)
    assert xs[0][:3] == [10.0, 12.0, 14.0] and xs[1][:3] == [None, 500.0, 501.0]  # the times start at 0
    with pytest.raises(KeyError):
        lust.replay("traffic-noon")


def test_the_status_lists_the_whole_design_and_what_exists_of_it(lust: MapSource) -> None:
    st = status("lust", lust.runs())
    assert st["scenarios"] == ["S1", "S2", "S3", "S4", "S5", "S6"] and "pcnme" in st["methods"]
    assert len(st["seeds"]) == 20 and st["seeds"][0] == 3000
    assert st["cells"]["S3/pcnme"] == {"state": "not trained", "seeds": []}
    assert st["cells"]["S3/greedy"]["state"] == "trained"  # the rules need no training: only an evaluation
