"""Rerun some scenarios of a finished map without touching the others (QUESTIONS Q60: S6, loose deadlines).

`train.py --scenarios S6` on a map whose full run has finished is a revision of that run:

- the plan's settings (windows, population, generations, budget, ...) must equal the finished run's, so the
  rerun scenario is trained exactly like the others;
- the scenario's old outputs move to data/<map>/_archive/<S>-<label>-<time>/: its NSGA-II windows,
  demonstrations and trained methods, and its rows of nsga/fronts.csv and demos/demos.csv;
- w* stays the finished run's (w_star.json is not rewritten): the other scenarios' models stay valid (13.3);
- the manifest keeps the finished run's identity and adds a record under "revisions": the scenarios, the new
  config hash, the commit, the plan, the label, the archive folder, start and end. Resuming an interrupted
  revision checks that record (same config and commit), as a full run checks its identity.

`evaluate.py --scenarios S6` then moves the scenario's rows of another config out of results/summary.csv and
its old run folders into results/archive/<S>-<label>/, and the report shows them as "<S> <label> (archived)".
"""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pcnme.paths import MapPaths
from pcnme.pipeline import runinfo

IGNORED = ("scenarios", "workers", "checkpoint_every_s")  # plan fields a revision may change


def key(scenarios: Iterable[str]) -> str:
    return "+".join(sorted(scenarios))


def applies(paths: MapPaths, scenarios: Iterable[str], all_scenarios: Iterable[str]) -> bool:
    """A strict subset of the map's scenarios, on a map whose full run has finished."""
    if not paths.manifest.exists() or not set(scenarios) < set(all_scenarios):
        return False
    return "finished" in json.loads(paths.manifest.read_text(encoding="utf-8"))


def start(paths: MapPaths, identity: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Begin or resume the revision of identity["plan"]["scenarios"]; returns its record."""
    identity = json.loads(json.dumps(identity, default=str))
    m = json.loads(paths.manifest.read_text(encoding="utf-8"))
    base, plan = m["identity"]["plan"], identity["plan"]
    diff = {k: (base.get(k), v) for k, v in plan.items() if k not in IGNORED and base.get(k) != v}
    if diff:
        lines = "\n".join(
            f"  {k}: the finished run has {a!r}, this run has {b!r}" for k, (a, b) in diff.items()
        )
        raise runinfo.Mismatch(
            "a revision trains with the finished run's settings (its scenarios compare with the others):\n"
            + lines
        )
    k = key(plan["scenarios"])
    revisions = m.setdefault("revisions", {})
    rec = revisions.get(k)
    if rec is not None and rec["identity"] == identity:
        rec.setdefault("resumed", []).append(runinfo.now())
    elif rec is not None and "finished" not in rec:
        lines = "\n".join(
            f"  {f}: the revision has {rec['identity'].get(f)!r}, this run has {v!r}"
            for f, v in identity.items()
            if rec["identity"].get(f) != v
        )
        raise runinfo.Mismatch(f"an unfinished revision of {k} differs:\n{lines}\nRestore it to resume.")
    else:
        if rec is not None:
            m.setdefault("revision_history", []).append(rec)
        dest = archive(paths, plan["scenarios"], label)
        rec = {"identity": identity, "label": label, "archived_to": str(dest), "started": runinfo.now()}
        revisions[k] = rec
    runinfo.write(paths.manifest, m)
    return dict(rec)


def finish(paths: MapPaths, scenarios: Iterable[str], hours: float) -> None:
    m = json.loads(paths.manifest.read_text(encoding="utf-8"))
    m["revisions"][key(scenarios)].update(finished=runinfo.now(), hours=hours)
    runinfo.write(paths.manifest, m)


def kept_w_star(paths: MapPaths) -> float:
    return float(json.loads(paths.w_star.read_text(encoding="utf-8"))["w_star"])


def archive(paths: MapPaths, scenarios: Iterable[str], label: str) -> Path:
    """Move the scenarios' training outputs aside (nothing is deleted); returns the folder."""
    scenarios = list(scenarios)
    dest = paths.root / "_archive" / f"{key(scenarios)}-{label}-{runinfo.now()[:19].replace(':', '')}"
    for s in scenarios:
        for pattern, sub in ((f"{s}-w*.json", "nsga"), (f"*-{s}-w*.parquet", "demos")):
            folder = paths.root / sub
            for f in sorted(folder.glob(pattern)) if folder.exists() else []:
                _move(f, dest / sub / f.name)
        train = paths.root / "train"
        for d in sorted(train.glob(f"{s}-*")) if train.exists() else []:
            _move(d, dest / "train" / d.name)
        for table in (paths.nsga / "fronts.csv", paths.demos / "demos.csv"):
            _split_rows(table, dest / table.parent.name / table.name, lambda r, s=s: r.get("scenario") == s)
    return dest


def _move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _split_rows(table: Path, aside: Path, take: Any) -> int:
    """Move the rows `take` selects from `table` to `aside` (appended); returns how many moved."""
    if not table.exists():
        return 0
    with table.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = list(reader.fieldnames or []), list(reader)
    moved = [r for r in rows if take(r)]
    if not moved:
        return 0
    kept = [r for r in rows if not take(r)]
    aside.parent.mkdir(parents=True, exist_ok=True)
    new = not aside.exists()
    with aside.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(moved)
    with table.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
    return len(moved)


# --- evaluation ---------------------------------------------------------------------------------------------


def label_of(paths: MapPaths, scenario: str) -> str | None:
    """The label of the latest revision that reran `scenario` (None: never revised)."""
    if not paths.manifest.exists():
        return None
    m = json.loads(paths.manifest.read_text(encoding="utf-8"))
    for rec in m.get("revisions", {}).values():
        if scenario in rec["identity"]["plan"]["scenarios"]:
            return str(rec["label"])
    return None


def archive_results(results: Path, scenario: str, label: str, config_hash: str) -> int:
    """Move a revised scenario's evaluation rows of another config, and their run folders, to
    results/archive/<S>-<label>/; returns how many rows moved."""
    dest = results / "archive" / f"{scenario}-{label}"
    summary = results / "summary.csv"
    if not summary.exists():
        return 0
    with summary.open(encoding="utf-8", newline="") as f:
        old = [r for r in csv.DictReader(f) if r["scenario"] == scenario and r["config_hash"] != config_hash]
    for r in old:
        run = results / "runs" / r["run_dir"]
        if run.exists():
            _move(run, dest / "runs" / run.name)
    return _split_rows(
        summary, dest / "summary.csv", lambda r: r["scenario"] == scenario and r["config_hash"] != config_hash
    )


def archived(results: Path) -> list[tuple[Path, str]]:
    """Every archived scenario of a map's results: (its folder, the name the report gives it)."""
    folder = results / "archive"
    out = []
    for d in sorted(folder.iterdir()) if folder.exists() else []:
        if (d / "summary.csv").exists():
            scenario, _, label = d.name.partition("-")
            out.append((d, f"{scenario} {label} (archived)"))
    return out
