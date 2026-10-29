"""The run's logs (final-design 16.3; docs/METRICS.md; PLAN M3.10; QUESTIONS Q28).

Rows are collected during a run and written once at the end: Parquet files under results/<run_id>/ (the
source of every reported number) and the same rows as InfluxDB line protocol (points.lp, for Grafana only).
Tables: tasks, attempts, zones, pipes (METRICS.md) and, for a map replay, positions and events (Q28). Every
row carries the common tags run_id, scenario, method, seed, mode. Times are simulated seconds; the Influx
points are stamped 2026-01-01T00:00:00Z + the simulated time (METRICS.md).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EPOCH_NS = 1_767_225_600_000_000_000  # 2026-01-01T00:00:00Z in ns
TABLES = ("tasks", "attempts", "zones", "pipes", "positions", "events")
MEASUREMENT = {
    "tasks": "task",
    "attempts": "attempt",
    "zones": "zone",
    "pipes": "pipe",
    "positions": "position",
    "events": "event",
}
TAGS = {
    "tasks": ("anchor", "type", "cls", "outcome"),
    "attempts": ("anchor", "executor", "slot_type", "result", "option"),
    "zones": ("zone", "slot_type"),
    "pipes": ("pipe",),
    "positions": ("car_id", "zone"),
    "events": ("car_id", "kind"),
}


@dataclass
class RunLog:
    scenario: str
    method: str
    seed: int
    mode: str = "fast"
    rows: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: {t: [] for t in TABLES})

    @property
    def run_id(self) -> str:
        """METRICS.md: <scenario>-<method>-<seed>-<mode>."""
        return f"{self.scenario}-{self.method}-{self.seed}-{self.mode}"

    def add(self, table: str, time_s: float, **values: Any) -> None:
        tags = {
            "run_id": self.run_id,
            "scenario": self.scenario,
            "method": self.method,
            "seed": self.seed,
            "mode": self.mode,
        }
        self.rows[table].append({**tags, "time_s": time_s, **values})

    def write(self, out_dir: Path, run_info: Mapping[str, Any]) -> Path:
        """results/<run_id>/: one Parquet file per table, run.json, points.lp."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        d = out_dir / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        for table, rows in self.rows.items():
            if rows:
                pq.write_table(pa.Table.from_pylist(rows), d / f"{table}.parquet")
        (d / "run.json").write_text(
            json.dumps({"run_id": self.run_id, **run_info}, indent=1, default=str), encoding="utf-8"
        )
        with (d / "points.lp").open("w", encoding="utf-8", newline="\n") as f:
            for table, rows in self.rows.items():
                for row in rows:
                    f.write(line_protocol(MEASUREMENT[table], row, TAGS[table]) + "\n")
        return d


def _escape(v: object) -> str:
    return str(v).replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")


def line_protocol(measurement: str, row: Mapping[str, Any], tag_keys: tuple[str, ...]) -> str:
    """One InfluxDB line: measurement,tags fields time (ns since 2026-01-01 + the simulated time)."""
    common = ("run_id", "scenario", "method", "seed", "mode")
    tags = ",".join(f"{k}={_escape(row[k])}" for k in (*common, *tag_keys) if row.get(k) not in (None, ""))
    fields = []
    for k, v in row.items():
        if k in common or k in tag_keys or k == "time_s" or v is None:
            continue
        if isinstance(v, bool):
            fields.append(f"{k}={'true' if v else 'false'}")
        elif isinstance(v, int):
            fields.append(f"{k}={v}i")
        elif isinstance(v, float):
            if math.isfinite(v):
                fields.append(f"{k}={v!r}")
        else:
            fields.append(f'{k}="{v}"')
    t_ns = EPOCH_NS + round(float(row["time_s"]) * 1e9)
    return f"{measurement},{tags} {','.join(fields) or 'n=0i'} {t_ns}"
