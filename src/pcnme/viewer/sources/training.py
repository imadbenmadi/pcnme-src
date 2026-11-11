"""The live-training source (PLAN V4; viewer/README.md): the learning methods' training, finished or running,
read from what train.py writes per episode (data/<map>/train/<scenario>-<method>/: episodes.csv, train.log,
training.json; pipeline/train.py).

`runs()` lists every trained or training scenario x method with its state; `curve(run)` gives its episodes
(on-time share, mean ê, loss, updates, decisions used of the budget) and the log's last lines. The page polls
a running one every few seconds, so a new episode appears as train.py writes it. Nothing is recomputed.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pcnme.viewer.sources.base import SourceInfo

RUNNING_S = 900  # written to in the last 15 minutes and not "done": still training
NUMERIC = (
    "episode",
    "seed",
    "on_time",
    "mean_e_hat",
    "tasks",
    "updates",
    "mean_loss",
    "decisions_used",
    "wall_s",
)


def _float(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _tail(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 16_384))
        lines = f.read().decode("utf-8", "replace").splitlines()
    return lines[-n:]


class TrainingSource:
    def __init__(self, info: SourceInfo, roots: Sequence[tuple[str, Path]], now: Any = time.time) -> None:
        self.info = info
        self._roots = list(roots)  # (map, data/<map>/train)
        self._now = now

    def _dirs(self) -> list[tuple[str, Path]]:
        out = []
        for m, root in self._roots:
            if root.exists():
                out += [(m, d) for d in sorted(root.iterdir()) if (d / "episodes.csv").exists()]
        return out

    def _state(self, d: Path) -> str:
        files = [f for f in (d / "episodes.csv", d / "train.log", d / "checkpoint.pt") if f.exists()]
        newest = max((f.stat().st_mtime for f in files), default=0.0)
        tail = _tail(d / "train.log", 1)
        done = bool(tail) and " done:" in tail[-1]
        return "running" if not done and self._now() - newest < RUNNING_S else "done" if done else "stopped"

    def runs(self) -> list[dict[str, Any]]:
        out = []
        for m, d in self._dirs():
            scenario, _, method = d.name.partition("-")
            with (d / "episodes.csv").open(encoding="utf-8", newline="") as f:
                n = sum(1 for _ in csv.DictReader(f))
            out.append({"run": f"{m}/{d.name}", "map": m, "scenario": scenario, "method": method,
                        "episodes": n, "state": self._state(d),
                        "updated": round((d / "episodes.csv").stat().st_mtime)})  # fmt: skip
        return out

    def _dir(self, run: str) -> Path:
        for m, d in self._dirs():
            if f"{m}/{d.name}" == run:
                return d
        raise KeyError(run)

    def curve(self, run: str) -> dict[str, Any]:
        d = self._dir(run)
        with (d / "episodes.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        cols: dict[str, list[float | None]] = {k: [_float(r.get(k, "")) for r in rows] for k in NUMERIC}
        budget = None
        info = d / "training.json"
        if info.exists():
            try:
                budget = json.loads(info.read_text(encoding="utf-8")).get("budget")
            except (ValueError, OSError):
                budget = None  # being rewritten by train.py: the next poll reads it
        scenario, _, method = d.name.partition("-")
        return {"run": run, "scenario": scenario, "method": method, "state": self._state(d), "budget": budget,
                "episodes": cols, "log": _tail(d / "train.log", 30)}  # fmt: skip


def training_source(maps: Sequence[str] = ("vfc", "lust")) -> TrainingSource:
    from pcnme.paths import MapPaths

    roots = [(m, MapPaths(m).root / "train") for m in maps]
    return TrainingSource(SourceInfo("training", "live-training", "Training"), roots)
