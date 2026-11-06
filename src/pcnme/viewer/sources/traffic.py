"""A trace map's traffic preview (viewer/README.md): the SUMO traces before any model is trained on the map.

One run per time of day (data/<map>/traces/<time>.npz): the vehicles SUMO recorded inside the crop, the
crop's roads from the scenario's network file, the zone layout. No tasks, no fog activity: nothing is trained
yet, and the page says so. The map's evaluated runs, when there are some, come from a ParquetSource next to it
(MapSource joins both).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from pcnme.paths import MapPaths
from pcnme.viewer.sources.base import RunInfo, SourceInfo
from pcnme.viewer.sources.replay import STEP_S, ParquetSource, _layout, frames

PREVIEW_S = 120.0  # seconds of traffic per preview, from the start of the recording (after SUMO's warm-up)
PREFIX = "traffic-"
NOTE = ("Traffic only: SUMO's vehicles in the crop. No model is trained on this map yet, so there are no "
        "tasks and the fog nodes are idle.")  # fmt: skip
TASK_KEYS = ("id", "car", "t", "close", "anchor", "type", "cls", "outcome", "attempts", "deadline",
             "latency", "energy", "upload", "forward", "wait", "run", "back")  # fmt: skip
ATTEMPT_KEYS = ("task", "attempt", "executor", "option", "slot", "result", "send", "arrive", "start",
                "finish", "car", "energy")  # fmt: skip


def roads(map_name: str) -> list[list[list[float]]]:
    """The crop's roads as polylines in crop coordinates (m), cached in data/<map>/traces/roads.json."""
    paths = MapPaths(map_name)
    cache = paths.traces / "roads.json"
    if cache.exists():
        out: list[list[list[float]]] = json.loads(cache.read_text(encoding="utf-8"))
        return out
    from pcnme.maps import map_config, trace_source
    from pcnme.mobility.lust import crop_and_edges, sumolib

    cfg = map_config(map_name)
    scenario = trace_source(cfg)
    crop, edge_ids = crop_and_edges(scenario, cfg)
    net = sumolib().net.readNet(str(scenario / "lust.net.xml"))
    lines = []
    for e in edge_ids:
        shape = net.getEdge(e).getShape()
        lines.append([[round(x - crop.x0, 1), round(y - crop.y0, 1)] for x, y in shape])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(lines, separators=(",", ":")), encoding="utf-8")
    return lines


def preview_runs(map_name: str) -> list[RunInfo]:
    folder = MapPaths(map_name).traces
    files = sorted(folder.glob("*.npz")) if folder.exists() else []
    return [RunInfo(f"{PREFIX}{f.stem}", f.stem, "traffic", 0, "traces") for f in files]


def preview(map_name: str, time_of_day: str) -> dict[str, Any]:
    """One time of day as a replay without tasks: every recorded vehicle in the first PREVIEW_S seconds."""
    d = np.load(MapPaths(map_name).traces / f"{time_of_day}.npz")
    t, v, x, y = d["t"], d["vehicle"], d["x"], d["y"]
    t0 = float(t.min())
    keep = (t >= t0) & (t <= t0 + PREVIEW_S)
    t, v, x, y = t[keep] - t0, v[keep], x[keep], y[keep]
    n = frames(PREVIEW_S)
    ids = sorted(set(int(k) for k in v))
    index = {k: i for i, k in enumerate(ids)}
    xs: list[list[float | None]] = [[None] * n for _ in ids]
    ys: list[list[float | None]] = [[None] * n for _ in ids]
    for tt, vv, xx, yy in zip(t.tolist(), v.tolist(), x.tolist(), y.tolist(), strict=True):
        f = round(tt / STEP_S)
        if f < n and abs(tt / STEP_S - f) < 1e-3:
            xs[index[vv]][f], ys[index[vv]][f] = round(xx, 1), round(yy, 1)
    layout = _layout(map_name, _scenario_at(map_name, time_of_day))
    layout["roads"] = roads(map_name)
    layout["north_up"] = True
    zero = [0.0] * n
    meta = {
        "run_id": f"{PREFIX}{time_of_day}",
        "map": map_name,
        "scenario": time_of_day,
        "method": "traffic",
        "seed": 0,
        "mode": "traces",
        "seconds": PREVIEW_S,
        "end": PREVIEW_S,
        "fleet": len(ids),
        "reward_w": None,
        "note": NOTE,
    }
    cars = {
        "ids": [f"v{k}" for k in ids],
        "step": STEP_S,
        "x": xs,
        "y": ys,
        "speed": [zero] * len(ids),
        "zone": [[""] * n] * len(ids),
    }
    tasks: dict[str, list[Any]] = {k: [] for k in TASK_KEYS}
    attempts: dict[str, list[Any]] = {k: [] for k in ATTEMPT_KEYS}
    return {"meta": meta, "layout": layout, "cars": cars, "zones": {}, "pipes": {}, "tasks": tasks,
            "attempts": attempts}  # fmt: skip


def _scenario_at(map_name: str, time_of_day: str) -> str:
    """A scenario replayed at this time of day (its fog layout is drawn; Q58)."""
    from pcnme.maps import map_config

    cfg = map_config(map_name)
    return next((s for s, t in cfg.map.scenario_time.items() if t == time_of_day), next(iter(cfg.scenarios)))


class MapSource:
    """A map's evaluated runs (ParquetSource) and, for a trace map, its traffic previews."""

    def __init__(self, info: SourceInfo, runs: ParquetSource) -> None:
        self.info = info
        self._runs = runs
        self.map_name = runs.map_name

    def runs(self) -> list[RunInfo]:
        return self._runs.runs() + preview_runs(self.map_name)

    def replay(self, run_id: str) -> dict[str, Any]:
        if run_id.startswith(PREFIX):
            time_of_day = run_id[len(PREFIX) :]
            if not (MapPaths(self.map_name).traces / f"{time_of_day}.npz").exists():
                raise KeyError(run_id)
            return preview(self.map_name, time_of_day)
        return self._runs.replay(run_id)

    def status(self) -> dict[str, Any]:
        return status(self.map_name, self.runs())


def status(map_name: str, runs: list[RunInfo]) -> dict[str, Any]:
    """What the design asks for and where each scenario x method stands: evaluated (with its seeds), trained
    but not evaluated, or not trained; plus a note per scenario when one applies (a revision under way)."""
    from pcnme.maps import map_config
    from pcnme.pipeline.evaluate import METHODS
    from pcnme.pipeline.revision import label_of

    cfg = map_config(map_name)
    paths = MapPaths(map_name)
    first = cfg.fast_mode.seed_split["evaluation"].start
    seeds: dict[tuple[str, str], list[int]] = {}
    for r in runs:
        if r.mode == "fast":
            seeds.setdefault((r.scenario, r.method), []).append(r.seed)
    out: dict[str, Any] = {
        "scenarios": list(cfg.scenarios),
        "methods": list(METHODS),
        "seeds": list(range(first, first + cfg.fast_mode.eval_seeds)),
        "cells": {},
        "notes": {},
    }
    for s in cfg.scenarios:
        for m in METHODS:
            trained = m in ("greedy", "random") or (paths.train(s, m) / "trained.pt").exists()
            have = sorted(seeds.get((s, m), []))
            state = "evaluated" if have else ("trained" if trained else "not trained")
            out["cells"][f"{s}/{m}"] = {"state": state, "seeds": have}
        label = label_of(paths, s)
        if label and (paths.results / "archive" / f"{s}-{label}").exists():
            out["notes"][s] = f"{s} was rerun; its earlier version is shown as '{s} {label} (archived)'"
        elif label:
            out["notes"][s] = (
                f"{s} is being rerun (revision '{label}'): the runs shown are still the earlier version "
                "until the new evaluation"
            )
    return out
