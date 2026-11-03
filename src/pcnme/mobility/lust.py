"""LuST traces (PLAN LuST; final-design 16.2, 18; Codeca et al. 2015): SUMO runs of the whole LuST scenario,
recorded only inside a square crop around central Luxembourg, one run per time of day (maps.lust in the
config), written as the replayer's trace files (mobility/traces.py).

Per time of day T: SUMO from T - warmup_s (vehicles that depart earlier are skipped, so the network fills
during the warm-up) to T + trace_s, step fcd_period_s; FCD recorded from T, only on the crop's edges; the FCD
XML is streamed into data/<map>/traces/<time of day>.npz in the crop's coordinates (0..crop_m), keeping the
samples inside the crop. SUMO comes from the eclipse-sumo package (the `maps` extra).
"""

from __future__ import annotations

import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pcnme.config import Config

CONFIG_FILE = "dua.static.sumocfg"  # LuST's static traffic lights and DUA routes (its README's default)


def sumo_home() -> Path:
    import sumo  # the eclipse-sumo package

    return Path(sumo.SUMO_HOME)


def sumolib() -> Any:
    tools = str(sumo_home() / "tools")
    if tools not in sys.path:
        sys.path.append(tools)
    import sumolib as lib

    return lib


def sumo_binary() -> Path:
    exe = "sumo.exe" if sys.platform == "win32" else "sumo"
    return sumo_home() / "bin" / exe


@dataclass(frozen=True)
class Crop:
    """The crop in SUMO's network coordinates: lower-left corner and side, m."""

    x0: float
    y0: float
    side: float

    def inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x >= self.x0) & (x <= self.x0 + self.side) & (y >= self.y0) & (y <= self.y0 + self.side)


def lonlat_to_xy(net: Any, lon: float, lat: float) -> tuple[float, float]:
    """Linear between the network's geographic and projected boundaries (its location element): within
    tens of metres over LuST's 13 km, enough to place a crop centre, and no pyproj needed."""
    lon0, lat0, lon1, lat1 = (float(v) for v in net._location["origBoundary"].split(","))
    (x0, y0), (x1, y1) = net.getBBoxXY()
    return x0 + (lon - lon0) / (lon1 - lon0) * (x1 - x0), y0 + (lat - lat0) / (lat1 - lat0) * (y1 - y0)


def crop_and_edges(scenario: Path, cfg: Config) -> tuple[Crop, list[str]]:
    """The crop around the map's centre and the network edges with any part inside it."""
    net = sumolib().net.readNet(str(scenario / "lust.net.xml"))
    cx, cy = lonlat_to_xy(net, *cfg.map.center_lonlat)
    side = cfg.map.crop_m
    crop = Crop(cx - side / 2, cy - side / 2, side)
    edges = []
    for e in net.getEdges():
        xs, ys = np.array([p[0] for p in e.getShape()]), np.array([p[1] for p in e.getShape()])
        if crop.inside(xs, ys).any():
            edges.append(e.getID())
    return crop, edges


def run_sumo(
    scenario: Path,
    crop_edges: Path,
    fcd: Path,
    begin: float,
    record: float,
    end: float,
    step: float,
    log: Path,
) -> None:
    crop_edges, fcd, log = crop_edges.resolve(), fcd.resolve(), log.resolve()  # SUMO runs in the scenario
    cmd = [
        str(sumo_binary()),
        "-c", str(scenario / CONFIG_FILE),
        "--begin", f"{begin:g}", "--end", f"{end:g}", "--step-length", f"{step:g}",
        "--fcd-output", str(fcd), "--device.fcd.begin", f"{record:g}",
        "--fcd-output.filter-edges.input-file", str(crop_edges),
        "--summary-output", str(fcd.with_suffix(".summary.xml")),
        "--tripinfo-output", str(fcd.with_suffix(".tripinfo.xml")),
        "--log", str(log), "--no-step-log", "true", "--verbose", "false",
        "--ignore-route-errors", "true",
    ]  # fmt: skip
    subprocess.run(cmd, check=True, cwd=scenario, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def fcd_to_trace(fcd: Path, crop: Crop, record: float, out: Path) -> dict[str, float]:
    """Stream the FCD XML into the trace file: crop coordinates, samples inside the crop, sorted by vehicle
    then time. Returns the trace's numbers (vehicles, samples, mean and peak vehicles at once)."""
    ids: dict[str, int] = {}
    t_l: list[float] = []
    v_l: list[int] = []
    x_l: list[float] = []
    y_l: list[float] = []
    per_step: list[int] = []
    for _, el in ET.iterparse(fcd, events=("end",)):
        if el.tag != "timestep":
            continue
        t = float(el.get("time", "0")) - record
        n = 0
        for v in el:
            x, y = float(v.get("x", "nan")), float(v.get("y", "nan"))
            if not crop.inside(np.array(x), np.array(y)):
                continue
            t_l.append(t)
            v_l.append(ids.setdefault(v.get("id", ""), len(ids)))
            x_l.append(x - crop.x0)
            y_l.append(y - crop.y0)
            n += 1
        per_step.append(n)
        el.clear()
    t_a, v_a = np.array(t_l, np.float32), np.array(v_l, np.int32)
    order = np.lexsort((t_a, v_a))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        t=t_a[order],
        vehicle=v_a[order],
        x=np.array(x_l, np.float32)[order],
        y=np.array(y_l, np.float32)[order],
    )
    return {
        "vehicles": len(ids),
        "samples": len(t_l),
        "mean_at_once": float(np.mean(per_step)) if per_step else 0.0,
        "peak_at_once": float(max(per_step, default=0)),
    }


def make_trace(
    cfg: Config,
    scenario: Path,
    time_of_day: str,
    out: Path,
    work: Path,
    *,
    trace_s: float | None = None,
    warmup_s: float | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> dict[str, float]:
    """One time of day's trace file. `trace_s` / `warmup_s` shorten the run (the smoke test)."""
    say = on_stage or (lambda s: None)
    m = cfg.map
    t0 = m.time_of_day_s[time_of_day]
    warm = m.warmup_s if warmup_s is None else warmup_s
    length = m.trace_s if trace_s is None else trace_s
    work.mkdir(parents=True, exist_ok=True)
    say("crop: reading the LuST network")
    crop, edges = crop_and_edges(scenario, cfg)
    sel = work / "crop-edges.txt"
    sel.write_text("".join(f"edge:{e}\n" for e in edges), encoding="utf-8")
    fcd = work / f"{time_of_day}.fcd.xml"
    say(f"SUMO: {warm + length:.0f} s of the whole city at {m.fcd_period_s:g} s steps")
    started = time.monotonic()
    run_sumo(scenario, sel, fcd, t0 - warm, t0, t0 + length, m.fcd_period_s, work / f"{time_of_day}.sumo.log")
    sumo_s = time.monotonic() - started
    say("FCD -> trace file")
    stats = fcd_to_trace(fcd, crop, t0, out)
    fcd.unlink()  # the trace file holds what the replayer needs; the XML is large
    return {
        **stats,
        "crop_x0": crop.x0,
        "crop_y0": crop.y0,
        "edges": len(edges),
        "sumo_wall_s": sumo_s,
        "simulated_s": warm + length,
    }
