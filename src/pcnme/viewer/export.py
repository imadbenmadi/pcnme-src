"""Export the viewer with its runs to a folder that runs anywhere with Python (no repo, no packages).

    uv run python -m pcnme.viewer.export                    # seed 3000 -> ../pcnme-history-viewer
    uv run python -m pcnme.viewer.export --seeds 3000 3001 --out D:/viewer
    uv run python -m pcnme.viewer.export --seeds all        # every seed (about 20 times larger)

The folder holds the page (static/), every chosen replay pre-built and gzipped (data/), and serve.py
(standalone.py: standard library only). On the other machine: python serve.py.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from pcnme.paths import REPO
from pcnme.viewer.sources import default_registry
from pcnme.viewer.sources.base import Registry, ReplaySource

HERE = Path(__file__).parent
DEFAULT_OUT = REPO.parent / "pcnme-history-viewer"

README = """# PCNME history viewer (exported {date})

Replays of the PCNME runs: every scenario and method, seeds {seeds}, and the testbed runs.

    python serve.py          (any Python 3.9+, nothing to install; Windows: double-click start.bat)

The browser opens http://127.0.0.1:8765. Try #vfc/S3/pcnme/3000/fast/greedy for PCNME next to Greedy.
Keys: Space play/pause, arrow keys step, [ ] speed, H hide the bar, F full screen, Esc close the panel.

Made with `uv run python -m pcnme.viewer.export` in the pcnme-src repo; export again to update it.
"""


def export(out: Path, seeds: set[int] | None, registry: Registry | None = None, log: bool = True) -> int:
    """Write the folder; returns the number of runs. seeds=None: every seed (testbed runs always go)."""
    out.mkdir(parents=True, exist_ok=True)
    if (out / "static").exists():
        shutil.rmtree(out / "static")
    shutil.copytree(HERE / "static", out / "static")
    shutil.copy2(HERE / "standalone.py", out / "serve.py")
    (out / "start.bat").write_bytes(b"@echo off\r\npython serve.py\r\npause\r\n")
    registry = registry or default_registry()
    sources = [s for s in registry.describe() if s["kind"] == "replay"]  # live kinds need the repo
    data = out / "data"
    (data / "runs").mkdir(parents=True, exist_ok=True)
    n = 0
    for s in sources:
        if s["status"] != "ready":
            continue
        src = registry.replay_source(s["id"])
        assert isinstance(src, ReplaySource)
        runs = [r for r in src.runs() if seeds is None or r.seed in seeds or r.mode in ("testbed", "traces")]
        (data / "status").mkdir(parents=True, exist_ok=True)
        (data / "status" / f"{s['id']}.json").write_text(json.dumps(src.status()), encoding="utf-8")
        (data / "runs" / f"{s['id']}.json").write_text(
            json.dumps([asdict(r) for r in runs]), encoding="utf-8"
        )
        folder = data / "replay" / s["id"]
        folder.mkdir(parents=True, exist_ok=True)
        for k, r in enumerate(runs, 1):
            f = folder / f"{r.run_id}.json.gz"
            if not f.exists():  # a replay never changes once a run is written
                body = json.dumps(src.replay(r.run_id), separators=(",", ":"), allow_nan=False).encode()
                f.write_bytes(gzip.compress(body, 6))
            n += 1
            if log:
                print(f"\r  {s['id']}: {k}/{len(runs)} {r.run_id:<40}", end="", flush=True)
        if log:
            print()
    (data / "sources.json").write_text(json.dumps(sources), encoding="utf-8")
    label = "all" if seeds is None else ", ".join(map(str, sorted(seeds)))
    (out / "README.md").write_text(
        README.format(date=time.strftime("%d %b %Y"), seeds=label), encoding="utf-8"
    )
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seeds", nargs="+", default=["3000"], help="evaluation seeds to export, or 'all'")
    a = ap.parse_args(argv)
    seeds = None if a.seeds == ["all"] else {int(x) for x in a.seeds}
    n = export(a.out, seeds)
    size = sum(f.stat().st_size for f in a.out.rglob("*") if f.is_file()) / 1e6
    print(f"{n} runs exported to {a.out} ({size:.0f} MB). Copy the folder, then: python serve.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
