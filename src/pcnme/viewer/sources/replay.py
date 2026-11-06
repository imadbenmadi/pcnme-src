"""The replay source: finished runs read from their logs (final-design 16.3, METRICS.md; viewer/README.md).

`ParquetSource` lists and loads the fast-mode and testbed run directories of one map
(data/<map>/results/runs/<run_id>/, data/<map>/testbed/results/<run_id>/).

A replay is plain JSON for the page: the city (grid, zones, hardware), the cars' positions on a fixed time
step, the zones' slots and queues, the WAN and backhaul pipes, and every task and attempt (columnar arrays).
Nothing is computed that the logs do not hold, except each task's close time for the running statistics.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

from pcnme.viewer.sources.base import RunInfo, SourceInfo

STEP_S = 0.2  # the replay's time step for positions, zones and pipes (the logs hold 0.1 s)


def parse_run_id(run_id: str) -> RunInfo | None:
    """<scenario>-<method>-<seed>-<mode>; methods may hold hyphens (d3qn-greedy)."""
    parts = run_id.split("-")
    if len(parts) < 4 or not parts[-2].isdigit():
        return None
    return RunInfo(run_id, parts[0], "-".join(parts[1:-2]), int(parts[-2]), parts[-1])


class ParquetSource:
    """The run directories under `roots` (each holds run.json and the Parquet tables)."""

    def __init__(self, info: SourceInfo, map_name: str, roots: Iterable[Path]) -> None:
        self.info = info
        self.map_name = map_name
        self._roots = [r for r in roots if r.exists()]

    def _dir(self, run_id: str) -> Path:
        for r in self._roots:
            d = r / run_id
            if (d / "run.json").exists():
                return d
        raise KeyError(run_id)

    def runs(self) -> list[RunInfo]:
        """Every run once: a run id in two roots is the first root's (as `replay` reads it)."""
        out, seen = [], set()
        for r in self._roots:
            for d in sorted(r.iterdir()):
                info = parse_run_id(d.name)
                if info is not None and info.run_id not in seen and (d / "tasks.parquet").exists():
                    seen.add(info.run_id)
                    out.append(info)
        return out

    def replay(self, run_id: str) -> dict[str, Any]:
        return _replay(str(self._dir(run_id)), self.map_name)


def _table(d: Path, name: str, columns: list[str]) -> dict[str, list[Any]]:
    import pyarrow.parquet as pq

    f = d / f"{name}.parquet"
    if not f.exists():
        return {c: [] for c in columns}
    out: dict[str, list[Any]] = pq.read_table(f, columns=columns).to_pydict()
    return out


def _r(x: float | None, digits: int = 4) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, digits)


def _frame(t: float) -> int:
    return round(t / STEP_S)


def frames(end: float) -> int:
    """The number of replay frames up to `end` (0, STEP_S, ..., the last step at or before end)."""
    return math.floor(end / STEP_S + 1e-9) + 1


@lru_cache(maxsize=6)
def _replay(run_dir: str, map_name: str) -> dict[str, Any]:
    d = Path(run_dir)
    info = json.loads((d / "run.json").read_text(encoding="utf-8"))
    pos = _table(d, "positions", ["time_s", "car_id", "x_m", "y_m", "speed_mps", "zone"])
    end = max(pos["time_s"], default=float(info.get("seconds", 60.0)))
    n = frames(end)
    cars = sorted(set(pos["car_id"]))
    index = {c: i for i, c in enumerate(cars)}
    xs = [[None] * n for _ in cars]
    ys = [[None] * n for _ in cars]
    sp = [[0.0] * n for _ in cars]
    zn = [[""] * n for _ in cars]
    for t, c, x, y, v, z in zip(
        *(pos[k] for k in ("time_s", "car_id", "x_m", "y_m", "speed_mps", "zone")), strict=True
    ):
        if abs(t / STEP_S - round(t / STEP_S)) > 1e-6:
            continue  # keep the frames on the replay's step
        i, f = index[c], _frame(t)
        if f >= n:
            continue
        xs[i][f], ys[i][f], sp[i][f], zn[i][f] = round(x, 1), round(y, 1), round(v, 1), z or ""
    return {
        "meta": _meta(info, d, map_name, end),
        "layout": _layout(map_name, info["scenario"]),
        "cars": {"ids": cars, "step": STEP_S, "x": xs, "y": ys, "speed": sp, "zone": zn},
        "zones": _zones(d, n),
        "pipes": _pipes(d, n),
        **_tasks(d, index),
    }


def _meta(info: dict[str, Any], d: Path, map_name: str, end: float) -> dict[str, Any]:
    return {
        "run_id": d.name,
        "map": map_name,
        "scenario": info["scenario"],
        "method": info.get("method") or (r.method if (r := parse_run_id(d.name)) else ""),
        "seed": info["seed"],
        "mode": "testbed" if d.name.endswith("-testbed") else "fast",
        "seconds": info.get("seconds", 60.0),
        "end": round(end, 3),
        "fleet": info.get("fleet"),
        "reward_w": info.get("reward_w"),
    }


def _layout(map_name: str, scenario: str) -> dict[str, Any]:
    from pcnme.maps import map_config

    cfg = map_config(map_name)
    lay, sc = cfg.layout, cfg.scenarios[scenario]
    zones: list[dict[str, Any]] = []
    for z, (x, y) in lay.zone_positions_m.items():
        hw = cfg.hardware[sc.fog[z]]
        zones.append({"id": z, "x": x, "y": y, "r": lay.zone_radius_m, "active": z not in lay.joining_zones,
                      "hw": hw.name, "gpu": hw.c_gpu, "cpu": hw.c_cpu,
                      "qgpu": hw.queue.get("gpu", 0), "qcpu": hw.queue.get("cpu", 0)})  # fmt: skip
    ids: list[str] = [str(z["id"]) for z in zones]
    edges = [
        [a, b]
        for i, a in enumerate(ids)
        for b in ids[i + 1 :]
        if math.dist(lay.zone_positions_m[a], lay.zone_positions_m[b]) <= lay.d_nb_m
    ]
    cloud = cfg.hardware["cloud"]
    return {
        "area": list(lay.area_m),
        "block": cfg.cars.grid.block_m,
        "zones": zones,
        "edges": edges,
        "cloud": {"gpu": cloud.c_gpu, "cpu": cloud.c_cpu},
        "deadlines": sc.deadlines,
        "rho_fog": sc.rho_fog,
    }


def _series(n: int) -> list[float]:
    return [0.0] * n


def _zones(d: Path, n: int) -> dict[str, Any]:
    t = _table(d, "zones", ["time_s", "zone", "slot_type", "rho", "queue_hard", "queue_soft", "busy"])
    out: dict[str, dict[str, dict[str, list[float]]]] = {}
    for time_s, z, slot, rho, qh, qs, busy in zip(*t.values(), strict=True):
        f = _frame(time_s)
        if f >= n or abs(time_s / STEP_S - f) > 1e-6:
            continue
        s = out.setdefault(z, {}).setdefault(
            slot, {"busy": _series(n), "queue": _series(n), "rho": _series(n)}
        )
        s["busy"][f], s["queue"][f], s["rho"][f] = busy, qh + qs, round(rho, 3)
    return out


def _pipes(d: Path, n: int) -> dict[str, list[float]]:
    t = _table(d, "pipes", ["time_s", "pipe", "wait_s"])
    out: dict[str, list[float]] = {}
    for time_s, pipe, wait in zip(*t.values(), strict=True):
        f = _frame(time_s)
        if f < n and abs(time_s / STEP_S - f) <= 1e-6:
            out.setdefault(pipe, _series(n))[f] = round(wait, 4)
    return out


# replay column <- log column (METRICS.md); times are rounded to 0.1 ms for the page
TASK_KEEP = {"anchor": "anchor", "type": "type", "cls": "cls", "outcome": "outcome", "attempts": "attempts"}
TASK_TIMES = {"deadline": "deadline_s", "latency": "latency_s", "energy": "energy_j", "upload": "upload_s",
              "forward": "forward_s", "wait": "wait_s", "run": "run_s", "back": "return_s"}  # fmt: skip
ATTEMPT_KEEP = {"attempt": "attempt", "executor": "executor", "option": "option", "slot": "slot_type",
                "result": "result"}  # fmt: skip
ATTEMPT_TIMES = {"send": "t_send_s", "arrive": "t_at_executor_s", "start": "t_start_s",
                 "finish": "t_finish_s", "car": "t_result_at_car_s", "energy": "energy_j"}  # fmt: skip


def _tasks(d: Path, cars: dict[str, int]) -> dict[str, Any]:
    """Tasks sorted by creation time (the page bisects on it); attempts point at a task by its index."""
    t = _table(d, "tasks", ["time_s", "task_id", *TASK_KEEP.values(), *TASK_TIMES.values()])
    a = _table(d, "attempts", ["task_id", *ATTEMPT_KEEP.values(), *ATTEMPT_TIMES.values()])
    order = sorted(range(len(t["time_s"])), key=lambda i: t["time_s"][i])
    pos = {t["task_id"][i]: k for k, i in enumerate(order)}
    tasks: dict[str, list[Any]] = {k: [] for k in ("id", "car", "t", "close", *TASK_KEEP, *TASK_TIMES)}
    for i in order:
        tid, created, lat = t["task_id"][i], t["time_s"][i], t["latency_s"][i]
        tasks["id"].append(tid)
        tasks["car"].append(cars.get(tid.split(".")[0], -1))
        tasks["t"].append(_r(created))
        tasks["close"].append(_r(created + (lat if lat is not None else t["deadline_s"][i])))
        for k, col in TASK_KEEP.items():
            tasks[k].append(t[col][i])
        for k, col in TASK_TIMES.items():
            tasks[k].append(_r(t[col][i]))
    attempts: dict[str, list[Any]] = {k: [] for k in ("task", *ATTEMPT_KEEP, *ATTEMPT_TIMES)}
    rows = sorted(range(len(a["task_id"])), key=lambda i: (pos.get(a["task_id"][i], -1), a["attempt"][i]))
    for i in rows:
        if a["task_id"][i] not in pos:
            continue
        attempts["task"].append(pos[a["task_id"][i]])
        for k, col in ATTEMPT_KEEP.items():
            attempts[k].append(a[col][i])
        for k, col in ATTEMPT_TIMES.items():
            attempts[k].append(_r(a[col][i]))
    return {"tasks": tasks, "attempts": attempts}


def parquet_source(map_name: str, label: str) -> ParquetSource:
    """A map's evaluation runs and testbed runs (data/<map>/results/runs/, data/<map>/testbed/results/)."""
    from pcnme.paths import MapPaths

    p = MapPaths(map_name)
    info = SourceInfo(map_name, "replay", label, map=map_name)
    return ParquetSource(info, map_name, [p.runs, p.testbed / "results"])
