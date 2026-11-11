"""Live metrics from the testbed to InfluxDB, for Grafana and the viewer's live mode (PLAN R2b, V3;
final-design 15.4: metrics are test equipment, never on the decision path; reported numbers still come from
the Parquet files).

Off unless PCNME_LIVE=1 (the measured batches run without it). A zone keeps the points in memory and a
background thread posts them every `period` seconds to InfluxDB's write API (line protocol). A failed post is
counted and dropped: the zone never waits for its metrics.

Times are the containers' shared wall clock (final-design 5.4: allowed for analysis and display, never for a
decision): a zone converts its own clock's times with `wall = clock time + offset`.

Measurements (tags: run, scenario, method, seed, zone):
    live_task   one point per task its anchor closes: tags outcome, type, cls; fields latency_s, deadline_s,
                energy_j, on_time, and `row`: the task's METRICS.md row and its attempts as JSON, times in
                wall seconds (the viewer draws them)
    live_zone   one point per zone and `sample_s`: busy and queued per slot type, open records, served cars,
                and the agent's online learning (updates, new samples, last loss, epsilon) when it learns
    live_car    one point per served car and `sample_s`: tag car; fields x, y, v (its last GPS fix)
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

Post = Callable[[str, bytes, Mapping[str, str]], int]  # url, body, headers -> HTTP status

# METRICS.md columns that are times on the zone's clock (the others are durations)
TASK_TIMES = ("time_s",)
ATTEMPT_TIMES = ("time_s", "t_send_s", "t_at_executor_s", "t_start_s", "t_finish_s", "t_result_at_car_s")


def _escape(v: object) -> str:
    return str(v).replace(" ", r"\ ").replace(",", r"\,").replace("=", r"\=")


def _string(v: str) -> str:
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def line(measurement: str, tags: Mapping[str, object], fields: Mapping[str, object], ns: int) -> str:
    """One InfluxDB line protocol line; None fields are left out."""
    t = ",".join(f"{k}={_escape(v)}" for k, v in tags.items() if v not in (None, ""))
    f = []
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, bool):
            f.append(f"{k}={'true' if v else 'false'}")
        elif isinstance(v, int):
            f.append(f"{k}={v}i")
        elif isinstance(v, float):
            if math.isfinite(v):
                f.append(f"{k}={v!r}")
        else:
            f.append(f"{k}={_string(str(v))}")
    return f"{measurement},{t} {','.join(f)} {ns}"


def _post(url: str, body: bytes, headers: Mapping[str, str]) -> int:
    req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    with urllib.request.urlopen(req, timeout=2) as r:  # the testbed's own InfluxDB
        return int(r.status)


class LivePusher:
    def __init__(
        self,
        url: str,
        org: str,
        bucket: str,
        token: str,
        tags: Mapping[str, object],
        period: float = 1.0,
        post: Post = _post,
        start: bool = True,
    ) -> None:
        query = urllib.parse.urlencode({"org": org, "bucket": bucket, "precision": "ns"})
        self.url = f"{url.rstrip('/')}/api/v2/write?{query}"
        self.headers = {"Authorization": f"Token {token}", "Content-Type": "text/plain; charset=utf-8"}
        self.tags = dict(tags)
        self.period = period
        self._post = post
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._producers: list[Callable[[], list[str]]] = []
        self.sent = self.failed = 0
        self._stop = threading.Event()
        if start:
            threading.Thread(target=self._run, name="live-metrics", daemon=True).start()

    @classmethod
    def from_env(cls, tags: Mapping[str, object]) -> LivePusher | None:
        if os.environ.get("PCNME_LIVE", "") not in ("1", "true", "yes"):
            return None
        return cls(
            os.environ.get("INFLUX_URL", "http://10.40.0.3:8086"),
            os.environ.get("INFLUX_ORG", "pcnme"),
            os.environ.get("INFLUX_BUCKET", "pcnme"),
            os.environ.get("INFLUX_TOKEN", ""),
            tags,
        )

    def add(
        self,
        measurement: str,
        tags: Mapping[str, object],
        fields: Mapping[str, object],
        ns: int | None = None,
    ) -> None:
        text = line(measurement, {**self.tags, **tags}, fields, time.time_ns() if ns is None else ns)
        with self._lock:
            self._lines.append(text)

    def task(self, outcome: str, task_type: str, cls: str, latency: float | None, deadline: float) -> None:
        self.add("live_task", {"outcome": outcome, "type": task_type, "cls": cls},
                 {"latency_s": latency, "deadline_s": deadline, "on_time": outcome == "on_time"})  # fmt: skip

    def producer(self, fn: Callable[[], list[str]]) -> None:
        """`fn` runs on the metrics thread before each post and returns more lines (work kept off the zone's
        loop, for example turning closed records into rows)."""
        self._producers.append(fn)

    def flush(self) -> None:
        extra = [x for fn in self._producers for x in fn()]
        with self._lock:
            lines, self._lines = self._lines + extra, []
        if not lines:
            return
        try:
            ok = self._post(self.url, "\n".join(lines).encode(), self.headers) in (200, 204)
        except OSError:
            ok = False
        if ok:
            self.sent += len(lines)
        else:
            self.failed += len(lines)  # dropped: the zone never waits for its metrics

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            try:
                self.flush()
            except Exception:
                self.failed += 1

    def close(self) -> None:
        self._stop.set()
        self.flush()


def zone_fields(zone: Any) -> dict[str, Any]:
    """A zone's live state: busy and queued per slot type, open records, served cars (zone.py names)."""
    out: dict[str, Any] = {"records": len(zone.records), "cars": len(zone._cars)}
    for slot, s in zone.executor.snapshot().items():
        out[f"busy_{slot}"] = int(s.u)
        out[f"queued_{slot}"] = int(s.n_hard + s.n_soft)
    return out


def agent_fields(placement: Any) -> dict[str, Any]:
    """The online learning of a zone's agent (agent/online.py OnlineAgent): nothing for a rule."""
    if not hasattr(placement, "updates"):
        return {}
    out: dict[str, Any] = {
        "updates": int(placement.updates),
        "samples": int(getattr(placement, "new_samples", 0)),
    }
    losses = getattr(placement, "losses", None)
    if losses:
        out["loss"] = float(losses[-1][4])  # (update, td, n_step, margin, total, mean_q)
        out["mean_q"] = float(losses[-1][5])
    eps = getattr(placement, "current_epsilon", None)
    if callable(eps):
        out["epsilon"] = float(eps())
    return out


def shifted(row: Mapping[str, Any], keys: tuple[str, ...], offset: float) -> dict[str, Any]:
    """A row with its clock times moved to wall seconds."""
    return {k: (v + offset if k in keys and isinstance(v, int | float) else v) for k, v in row.items()}


class LiveZone:
    """One zone's live points: its closed tasks (as rows, built on the metrics thread), and every `sample_s`
    its slots, cars and agent. Hooks: closed(record) from on_close, fix(...) from /gps, start() once."""

    def __init__(self, pusher: LivePusher, zone: Any, cfg: Any, clock: Any, sample_s: float = 0.2) -> None:
        self.pusher, self.zone, self.cfg, self.clock, self.sample_s = pusher, zone, cfg, clock, sample_s
        self.offset = time.time() - clock.now  # wall = clock time + offset (display only, 5.4)
        self._records: list[Any] = []
        self._fixes: dict[str, tuple[float, float, float]] = {}
        self._lock = threading.Lock()
        pusher.producer(self._task_lines)

    def start(self) -> None:
        self.clock.schedule_in(self.sample_s, self._sample)

    def closed(self, record: Any) -> None:
        with self._lock:
            self._records.append(record)

    def fix(self, car_id: str, x: float, y: float, vx: float, vy: float) -> None:
        self._fixes[car_id] = (x, y, (vx * vx + vy * vy) ** 0.5)

    def _sample(self) -> None:
        z = self.zone
        ns = time.time_ns()
        self.pusher.add("live_zone", {}, {**zone_fields(z), **agent_fields(z._placement)}, ns)
        for car_id in list(z._cars):
            f = self._fixes.get(car_id)
            if f is not None:
                self.pusher.add("live_car", {"car": car_id}, {"x": f[0], "y": f[1], "v": f[2]}, ns)
        self.clock.schedule_in(self.sample_s, self._sample)

    def _task_lines(self) -> list[str]:
        with self._lock:
            records, self._records = self._records, []
        return [self.task_line(r) for r in records]

    def task_line(self, record: Any) -> str:
        from pcnme.experiments.city import record_rows
        from pcnme.metrics.runlog import RunLog

        log = RunLog("", "", 0, "testbed")
        record_rows(log, self.cfg, [record])
        common = ("run_id", "scenario", "method", "seed", "mode")
        task = {k: v for k, v in log.rows["tasks"][0].items() if k not in common}
        attempts = [{k: v for k, v in a.items() if k not in common} for a in log.rows["attempts"]]
        row = {"task": shifted(task, TASK_TIMES, self.offset),
               "attempts": [shifted(a, ATTEMPT_TIMES, self.offset) for a in attempts]}  # fmt: skip
        tags = {"zone": task["anchor"], "outcome": task["outcome"], "type": task["type"], "cls": task["cls"]}
        fields = {
            "latency_s": task["latency_s"],
            "deadline_s": task["deadline_s"],
            "energy_j": task["energy_j"],
            "on_time": task["outcome"] == "on_time",
            "row": json.dumps(row, default=str),
        }
        return line("live_task", {**self.pusher.tags, **tags}, fields, time.time_ns())
