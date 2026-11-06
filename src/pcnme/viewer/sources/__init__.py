"""Viewer data sources (viewer/README.md): the interface in base.py, one module per kind."""

from __future__ import annotations

from pcnme.viewer.sources.base import Placeholder, Registry, SourceInfo


def default_registry() -> Registry:
    """What `view.py` shows: the VFC grid's runs; LuST listed in development (no runs yet)."""
    from pcnme.viewer.sources.replay import map_source

    lust = SourceInfo(
        "lust",
        "replay",
        "LuST (Luxembourg)",
        status="development",
        map="lust",
        notes={
            "title": "LuST (Luxembourg)",
            "summary": "A 1.2 km crop of central Luxembourg, with car traces from the LuST SUMO scenario at "
            "three times of day. No model has been trained or evaluated on this map yet, so there is nothing "
            "to replay.",
            "steps": [
                ["Crop, SUMO traces, trace replayer", "built, smoke-tested", True],
                ["Full traces and coverage", "next", False],
                ["Training and evaluation on LuST", "planned", False],
            ],
        },
    )
    return Registry([map_source("vfc", "VFC grid"), Placeholder(lust)])
