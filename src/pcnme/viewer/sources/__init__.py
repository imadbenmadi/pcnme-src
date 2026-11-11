"""Viewer data sources (viewer/README.md): the interface in base.py, one module per kind."""

from __future__ import annotations

from pcnme.viewer.sources.base import Placeholder, Registry, Source, SourceInfo

LUST_NOTES = {
    "title": "LuST (Luxembourg)",
    "summary": "A 1.2 km crop of central Luxembourg, with car traces from the LuST SUMO scenario at three "
    "times of day. The traces are not made yet, so there is nothing to show.",
    "steps": [
        ["Crop, SUMO traces, trace replayer", "built, smoke-tested", True],
        ["Full traces and coverage (train.py --maps lust --until traces)", "next", False],
        ["Training and evaluation on LuST", "planned", False],
    ],
}


def map_source(map_name: str, label: str) -> Source:
    """A map's runs (and a trace map's traffic previews); a placeholder while there is nothing to show."""
    from pcnme.viewer.sources.replay import parquet_source
    from pcnme.viewer.sources.traffic import MapSource

    src = MapSource(SourceInfo(map_name, "replay", label, map=map_name), parquet_source(map_name, label))
    if src.runs():
        return src
    return Placeholder(
        SourceInfo(map_name, "replay", label, status="development", map=map_name, notes=LUST_NOTES)
    )


def default_registry() -> Registry:
    """What `view.py` shows: the VFC grid and LuST, each with whatever runs and traces exist; the live testbed
    (InfluxDB, when a live run pushes to it); the training of the learning methods."""
    from pcnme.viewer.sources.live import live_source
    from pcnme.viewer.sources.training import training_source

    return Registry([map_source("vfc", "VFC grid"), map_source("lust", "LuST (Luxembourg)"), live_source(),
                     training_source()])  # fmt: skip
