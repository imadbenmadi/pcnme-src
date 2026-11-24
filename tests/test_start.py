"""start.py: the website alone by default, the live run too with --live."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("start", ROOT / "start.py")
assert _spec and _spec.loader
start = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(start)


def test_normal_starts_only_the_viewer() -> None:
    cmds = start.commands(start.parse([]))
    assert len(cmds) == 1 and cmds[0][1:] == ["view.py", "--port", "8765"]


def test_live_starts_the_live_run_then_the_viewer() -> None:
    cmds = start.commands(start.parse(["--live", "--scenario", "S3", "--minutes", "5", "--no-browser"]))
    assert [c[1] for c in cmds] == ["run_live.py", "view.py"]
    assert cmds[0][2:] == ["--scenario", "S3", "--method", "pcnme", "--minutes", "5.0"]
    assert cmds[1][-1] == "--no-browser"
