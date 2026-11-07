"""Live metrics from the testbed to InfluxDB, for Grafana (PLAN R2b; final-design 15.4: metrics are test
equipment, never on the decision path; reported numbers still come from the Parquet files).

Off unless PCNME_LIVE=1 (the measured batches run without it). A zone keeps the points in memory and a
background thread posts them every `period` seconds to InfluxDB's write API (line protocol, wall-clock time,
so Grafana's "last 5 minutes" shows the run as it goes). A failed post is counted and dropped: the zone never
waits for the metrics.

Measurements (tags: run_id, scenario, method, seed, zone):
    live_task   one point per task its anchor closes: tags outcome, type, cls;
                fields latency_s, deadline_s, on_time
    live_zone   one point per zone and period: fields busy and queued per slot type, records, cars, closed
"""

from __future__ import annotations

import os
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

Post = Callable[[str, bytes, Mapping[str, str]], int]  # url, body, headers -> HTTP status


def _escape(v: object) -> str:
    return str(v).replace(" ", r"\ ").replace(",", r"\,").replace("=", r"\=")


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
            f.append(f"{k}={v!r}")
        else:
            f.append(f'{k}="{v}"')
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

    def add(self, measurement: str, tags: Mapping[str, object], fields: Mapping[str, object]) -> None:
        text = line(measurement, {**self.tags, **tags}, fields, time.time_ns())
        with self._lock:
            self._lines.append(text)

    def task(self, outcome: str, task_type: str, cls: str, latency: float | None, deadline: float) -> None:
        self.add("live_task", {"outcome": outcome, "type": task_type, "cls": cls},
                 {"latency_s": latency, "deadline_s": deadline, "on_time": outcome == "on_time"})  # fmt: skip

    def flush(self) -> None:
        with self._lock:
            lines, self._lines = self._lines, []
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
            self.flush()

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
