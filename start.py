"""Start the PCNME website (the viewer) with one command. See READ.md.

uv run python start.py            the website: VFC, LuST, testbed runs, training (no live run)
uv run python start.py --live     the same, plus a live testbed run (Docker Desktop must be running)

Opens http://127.0.0.1:8765. Ctrl+C stops it (and the live run); `uv run python stop_all.py` cleans up.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def commands(a: argparse.Namespace) -> list[list[str]]:
    """The processes to start: the live run first (when asked), then the viewer."""
    out = []
    if a.live:
        out.append([sys.executable, "run_live.py", "--scenario", a.scenario, "--method", a.method,
                    "--minutes", str(a.minutes)])  # fmt: skip
    viewer = [sys.executable, "view.py", "--port", str(a.port)]
    if a.no_browser:
        viewer.append("--no-browser")
    return [*out, viewer]


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="also start a live testbed run (VFC, Docker)")
    ap.add_argument("--scenario", default="S2")
    ap.add_argument("--method", default="pcnme")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    cmds = commands(parse(argv))
    *background, viewer = cmds
    procs = [subprocess.Popen(c) for c in background]
    try:
        return subprocess.call(viewer)
    except KeyboardInterrupt:
        return 0
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
