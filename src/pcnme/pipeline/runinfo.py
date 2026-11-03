"""What a run is (data/<map>/manifest.json): the config hash, the commit, the seeds, the budgets and the start
and end times. Outputs of different configs or commits are never mixed: a run that finds a manifest that
differs refuses to resume and says why (train.py --fresh archives the old outputs instead)."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pcnme.paths import REPO


def commit() -> str:
    """The checked-out commit; "-dirty" when tracked source files differ (data/ and docs/ excluded)."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                "src",
                "config",
                "train.py",
                "evaluate.py",
                "run_testbed.py",
                "pyproject.toml",
                "uv.lock",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head[:12] + ("-dirty" if dirty else "")


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


class Mismatch(RuntimeError):
    """Existing outputs came from another config or commit."""


def check_or_start(path: Path, identity: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """Load the manifest and check it names the same run; start one when there is none. `identity`: what must
    match to resume (config hash, commit, budgets, seeds); `extra`: recorded only."""
    identity = json.loads(json.dumps(identity, default=str))  # as it reads back (tuples become lists)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        diff = {
            k: (old.get("identity", {}).get(k), v)
            for k, v in identity.items()
            if old.get("identity", {}).get(k) != v
        }
        if diff:
            lines = "\n".join(f"  {k}: outputs have {a!r}, this run has {b!r}" for k, (a, b) in diff.items())
            raise Mismatch(
                f"{path.parent} holds outputs of another run:\n{lines}\n"
                "Refusing to mix them. Run with --fresh to archive them and start over, or restore that "
                "config / commit to resume."
            )
        old.setdefault("resumed", []).append(now())
        old.update({k: v for k, v in extra.items() if k not in ("started",)})
        write(path, old)
        return old
    data = {"identity": dict(identity), "started": now(), **extra}
    write(path, data)
    return data


def write(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def update(path: Path, **values: Any) -> None:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.update(values)
    write(path, data)
