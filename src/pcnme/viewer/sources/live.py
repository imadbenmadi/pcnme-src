"""The live-testbed source (PLAN V3; viewer/README.md): a testbed run while it goes, read from the InfluxDB
the zones push to (testbed/live.py, started by run_live.py).

The page polls `delta(run, since)` about once a second and appends what is new: the tasks closed (each with
its attempts, the same columns as a replay), the zones' slots and the agents' learning, and the cars'
positions. Times are seconds since the run's first point, on the containers' shared wall clock (final-design
5.4: display only). A poll reads back a few seconds before its cursor, because each zone posts its points once
a second and a later post can carry earlier times; the page keeps tasks by id and samples by time, so overlaps
are harmless.

Standard library only: InfluxDB's HTTP query API with Flux, its CSV answer parsed here.
"""

from __future__ import annotations

import csv
import io
import json
import math
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pcnme.viewer.sources.base import SourceInfo
from pcnme.viewer.sources.replay import ATTEMPT_KEEP, ATTEMPT_TIMES, TASK_KEEP, TASK_TIMES

OVERLAP_NS = 3_000_000_000  # a poll reads 3 s before its cursor (zones post once a second)
HELP = ("No live run yet. Start one (Docker Desktop running, nothing else on the testbed):\n"
        "uv run python run_live.py --scenario S2 --method pcnme --minutes 10")  # fmt: skip


def read_env(path: Path) -> dict[str, str]:
    """docker/.env: KEY=VALUE lines (the InfluxDB org, bucket and token)."""
    out: dict[str, str] = {}
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if s and not s.startswith("#") and "=" in s:
                k, v = s.split("=", 1)
                out[k.strip()] = v.strip().strip('"')
    return out


def parse_csv(text: str) -> list[dict[str, str]]:
    """InfluxDB's annotated-CSV answer without annotations: tables separated by blank lines, each with its
    own header row (",result,table,..."); one dict per record."""
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for rec in csv.reader(io.StringIO(text)):
        if not rec or all(not c for c in rec):
            header = None
            continue
        if header is None:
            header = rec
            continue
        rows.append(dict(zip(header, rec, strict=False)))
    return rows


class Influx:
    def __init__(self, url: str, org: str, bucket: str, token: str, timeout: float = 4.0) -> None:
        self.url, self.org, self.bucket, self.token, self.timeout = (
            url.rstrip("/"),
            org,
            bucket,
            token,
            timeout,
        )

    @classmethod
    def from_env_file(cls, path: Path, url: str = "http://127.0.0.1:8086") -> Influx:
        e = read_env(path)
        return cls(
            url, e.get("INFLUX_ORG", "pcnme"), e.get("INFLUX_BUCKET", "pcnme"), e.get("INFLUX_TOKEN", "")
        )

    def query(self, flux: str) -> list[dict[str, str]]:
        body = json.dumps({"query": flux, "dialect": {"annotations": [], "header": True}}).encode()
        req = urllib.request.Request(
            f"{self.url}/api/v2/query?org={urllib.parse.quote(self.org)}",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Token {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/csv",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:  # the testbed's own InfluxDB
            return parse_csv(r.read().decode("utf-8"))


def _flux_str(s: str) -> str:
    return json.dumps(s)  # a Flux string literal: double quotes, backslash escapes


def _ns(rfc3339: str) -> int:
    """InfluxDB's RFC 3339 time (nanoseconds, UTC 'Z') -> ns since the epoch, exactly."""
    import datetime as dt

    date, _, frac = rfc3339.rstrip("Z").partition(".")
    base = dt.datetime.fromisoformat(date).replace(tzinfo=dt.UTC)
    return int(base.timestamp()) * 1_000_000_000 + int((frac + "000000000")[:9])


def _num(v: str | None) -> float | None:
    if v in (None, ""):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def _rel(x: Any, t0: float) -> float | None:
    return round(x - t0, 4) if isinstance(x, int | float) and math.isfinite(x) else None


def _clean(x: Any) -> Any:
    return None if isinstance(x, float) and not math.isfinite(x) else x


def task_object(row: Mapping[str, Any], t0: float, offsets: dict[str, float] | None = None) -> dict[str, Any]:
    """A live_task point's JSON row (testbed/live.py: METRICS.md columns, wall seconds) -> the replay's task
    columns (sources/replay.py) as one object with its attempts, times relative to the run's start.

    The creation time is the car's clock, the attempts the anchor's (replay.anchor_offsets): it is moved onto
    the anchor's clock by its first send - upload, or by the anchor's last such offset (`offsets`, kept per
    anchor) for a task dropped before any attempt. Display only (final-design 5.4)."""
    t = row["task"]
    sends = [a["t_send_s"] for a in row.get("attempts", []) if isinstance(a.get("t_send_s"), int | float)]
    raw = t.get("time_s")
    if isinstance(raw, int | float) and offsets is not None:
        anchor = str(t.get("anchor", ""))
        if sends:
            offsets[anchor] = min(sends) - (t.get("upload_s") or 0.0) - raw
        raw = raw + offsets.get(anchor, 0.0)
    created = _rel(raw, t0)
    lat, dl = t.get("latency_s"), t.get("deadline_s")
    out: dict[str, Any] = {"id": t["task_id"], "car": str(t["task_id"]).split(".")[0], "t": created}
    out["close"] = None if created is None else round(created + (lat if lat is not None else dl or 0.0), 4)
    for k, col in TASK_KEEP.items():
        out[k] = _clean(t.get(col))
    for k, col in TASK_TIMES.items():
        out[k] = _clean(t.get(col))
    atts = []
    for a in sorted(row.get("attempts", []), key=lambda a: a.get("attempt", 0)):
        o = {k: _clean(a.get(col)) for k, col in ATTEMPT_KEEP.items()}
        for k, col in ATTEMPT_TIMES.items():
            o[k] = _clean(a.get(col)) if k == "energy" else _rel(a.get(col), t0)
        atts.append(o)
    out["att"] = atts
    return out


class LiveSource:
    """The live testbed runs in InfluxDB (testbed/live.py measurements)."""

    def __init__(self, info: SourceInfo, influx: Influx, map_name: str = "vfc") -> None:
        self.info = info
        self.influx = influx
        self.map_name = map_name
        self._t0: dict[str, int] = {}
        self._offsets: dict[str, dict[str, float]] = {}  # per run and anchor: anchor clock - car clock

    def _zone_query(self, run: str, start: str, tail: str) -> str:
        return (f'from(bucket: {_flux_str(self.influx.bucket)}) |> range(start: {start})\n'
                f'  |> filter(fn: (r) => r._measurement == "live_zone" and r.run == {_flux_str(run)} '
                f'and r._field == "records")\n{tail}')  # fmt: skip

    def runs(self) -> dict[str, Any]:
        """The live runs of the last 30 days, newest first; `online` False when InfluxDB does not answer."""
        try:
            last = self.influx.query(
                f"from(bucket: {_flux_str(self.influx.bucket)}) |> range(start: -30d)\n"
                '  |> filter(fn: (r) => r._measurement == "live_zone" and r._field == "records")\n'
                '  |> group(columns: ["run", "scenario", "method", "seed"]) |> last()'
            )
        except OSError as e:
            return {"online": False, "error": str(e), "help": HELP, "runs": []}
        runs = sorted(
            ({"run": r["run"], "scenario": r.get("scenario", ""), "method": r.get("method", ""),
              "seed": r.get("seed", ""), "last": r["_time"]} for r in last if r.get("run")),
            key=lambda r: r["last"], reverse=True,
        )  # fmt: skip
        return {"online": True, "help": HELP, "runs": runs}

    def t0(self, run: str) -> int:
        """The run's first point (ns): time 0 of the page."""
        if run not in self._t0:
            first = self.influx.query(self._zone_query(run, "-30d", "  |> group() |> first()"))
            if not first:
                raise KeyError(run)
            self._t0[run] = _ns(first[0]["_time"])
        return self._t0[run]

    def delta(self, run: str, since_ns: int = 0) -> dict[str, Any]:
        """What is new in `run` since `since_ns` (0: from the start), and the new cursor."""
        t0n = self.t0(run)
        t0 = t0n / 1e9
        start = max(t0n - 10_000_000_000, since_ns - OVERLAP_NS) if since_ns else t0n - 10_000_000_000
        rng = f"time(v: {start})"
        bucket, r = _flux_str(self.influx.bucket), _flux_str(run)
        base = f"from(bucket: {bucket}) |> range(start: {rng})\n"
        tasks = self.influx.query(
            base + f'  |> filter(fn: (r) => r._measurement == "live_task" and r.run == {r})\n'
            '  |> filter(fn: (r) => r._field == "row")\n'
            '  |> group() |> keep(columns: ["_time", "_value"])'
        )
        zones = self.influx.query(
            base + f'  |> filter(fn: (r) => r._measurement == "live_zone" and r.run == {r})\n'
            "  |> toFloat()\n"  # integer and float fields in one row (pivot needs one type)
            '  |> group(columns: ["zone", "scenario", "method", "seed"])\n'
            '  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")'
        )
        cars = self.influx.query(
            base + f'  |> filter(fn: (r) => r._measurement == "live_car" and r.run == {r})\n'
            "  |> toFloat()\n"
            '  |> group(columns: ["car", "zone"])\n'
            '  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")'
        )
        cursor = since_ns
        out_tasks = []
        for row in tasks:
            cursor = max(cursor, _ns(row["_time"]))
            try:
                offsets = self._offsets.setdefault(run, {})
                out_tasks.append(task_object(json.loads(row["_value"]), t0, offsets))
            except (ValueError, KeyError, TypeError):
                continue  # a malformed point is skipped, never the poll
        skip = {"result", "table", "_start", "_stop", "_time", "_measurement", "zone", "run", "scenario",
                "method", "seed", "car", ""}  # fmt: skip
        out_zones = []
        for row in zones:
            ns = _ns(row["_time"])
            cursor = max(cursor, ns)
            fields = {k: _num(v) for k, v in row.items() if k not in skip}
            out_zones.append([row.get("zone", ""), round(ns / 1e9 - t0, 3), fields])
        out_cars = []
        for row in cars:
            ns = _ns(row["_time"])
            cursor = max(cursor, ns)
            x, y = _num(row.get("x")), _num(row.get("y"))
            if x is not None and y is not None:
                out_cars.append([row.get("car", ""), row.get("zone", ""), round(ns / 1e9 - t0, 3),
                                 round(x, 1), round(y, 1), round(_num(row.get("v")) or 0.0, 1)])  # fmt: skip
        out: dict[str, Any] = {"run": run, "t0": t0, "cursor": cursor, "tasks": out_tasks, "zones": out_zones,
                               "cars": out_cars}  # fmt: skip
        if not since_ns:
            out["meta"], out["layout"] = self._meta(run, zones)
        return out

    def _meta(self, run: str, zones: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
        from pcnme.viewer.sources.replay import _layout

        tags = zones[0] if zones else {}
        scenario = tags.get("scenario") or run.split("-")[0]
        method = tags.get("method") or ""
        meta = {"run_id": run, "map": self.map_name, "scenario": scenario, "method": method,
                "seed": tags.get("seed", ""), "mode": "live", "seconds": None, "end": 0,
                "fleet": None}  # fmt: skip
        return meta, _layout(self.map_name, scenario)


def live_source() -> LiveSource:
    root = Path(__file__).resolve().parents[4]
    return LiveSource(SourceInfo("live", "live-testbed", "Live testbed", map="vfc"),
                      Influx.from_env_file(root / "docker" / ".env"))  # fmt: skip
