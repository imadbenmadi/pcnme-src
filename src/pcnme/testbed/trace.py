"""Per-hop timestamps of sampled tasks on the testbed, for analysis only (final-design 5.4: the containers
share the host's clock, which analysis may use and decisions never do; PLAN Q1).

Switched on with PCNME_TRACE=<n> (every n-th task of each car, by sequence number). Each process records
(wall time, task, attempt, event) in memory and writes /results/trace-<process>.jsonl at its end; the runner
moves the files into the run's folder and tools/testbed_hops.py turns them into per-hop durations. A zone
records its executor's start and finish too, converted from its own clock to wall time.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

Hook = Callable[[Any, int, str, "float | None"], None]


class Tracer:
    def __init__(self, process: str, every: int, clock_now: Callable[[], float] | None = None) -> None:
        self.process = process
        self.every = every
        self._now = clock_now
        self.rows: list[tuple[float, str, int, str]] = []

    @classmethod
    def from_env(cls, process: str, clock_now: Callable[[], float] | None = None) -> Tracer | None:
        every = int(os.environ.get("PCNME_TRACE", "0") or 0)
        return cls(process, every, clock_now) if every > 0 else None

    def sampled(self, task_id: Any) -> bool:
        try:
            return int(task_id[2]) % self.every == 0  # TaskId = (car, epoch, seq)
        except (TypeError, ValueError, IndexError):
            return False

    def mark(self, task_id: Any, attempt: int, event: str, at: float | None = None) -> None:
        """`at`: a time on this process's own clock (converted to wall time); None: now."""
        if not self.sampled(task_id):
            return
        wall = time.time()
        if at is not None and self._now is not None:
            wall -= self._now() - at
        self.rows.append((wall, ".".join(str(p) for p in task_id), int(attempt), event))

    def write(self, folder: Path) -> Path:
        out = folder / f"trace-{self.process}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for wall, task, attempt, event in self.rows:
                f.write(json.dumps({"t": wall, "task": task, "attempt": attempt, "event": event,
                                    "at": self.process}) + "\n")  # fmt: skip
        return out


# --- analysis ---------------------------------------------------------------------------------------------

HOPS = (  # (name, from event, to event); "serving" events are the ones of the zone that pushed to the car
    ("car: created -> upload sent", "created", "upload sent"),
    ("upload: sent -> intake at the anchor", "upload sent", "intake (upload complete)"),
    ("anchor: intake -> decision sent", "intake (upload complete)", "sent"),
    ("forward and queue: sent -> run start", "sent", "exec start"),
    ("run", "exec start", "exec finish"),
    ("executor: finish -> publish", "exec finish", "publish called"),
    ("MQTT: publish -> result at the serving zone", "publish called", "result at serving"),
    ("serving zone: result -> push", "result at serving", "push to the car"),
    ("HTTP: push -> result at the car", "push to the car", "result at the car"),
    ("total: created -> result at the car", "created", "result at the car"),
)


def hop_durations(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Per hop, its duration (s) for every traced attempt that has both ends; the attempt that reached the
    car first per task. Rows: the trace files' lines (t, task, attempt, event, at)."""
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    out: dict[str, list[float]] = {name: [] for name, _, _ in HOPS}
    for events in by_task.values():
        pushes = sorted((e for e in events if e["event"] == "push to the car"), key=lambda e: e["t"])
        if not pushes:
            continue
        push = pushes[0]
        attempt, serving = push["attempt"], push["at"]
        t: dict[str, float] = {}
        for e in events:
            ev = e["event"]
            if ev.startswith("sent (") and e["attempt"] == attempt:
                t["sent"] = e["t"]
            elif ev.startswith("result at z") and e["attempt"] == attempt and e["at"] == serving:
                t["result at serving"] = e["t"]
            elif (
                ev in ("exec start", "exec finish", "publish called") and e["attempt"] == attempt
            ) or ev in (
                "created",
                "upload sent",
                "intake (upload complete)",
                "result at the car",
            ):
                t.setdefault(ev, e["t"])
        t["push to the car"] = push["t"]
        for name, a, b in HOPS:
            if a in t and b in t:
                out[name].append(t[b] - t[a])
    return out
