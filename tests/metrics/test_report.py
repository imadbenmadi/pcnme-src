"""`pcnme report` (PLAN M7 acceptance: a hand-checked interval from synthetic data; final-design 16.3)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow")

from pcnme.experiments.cli import main
from pcnme.metrics.report import report, run_dirs
from pcnme.metrics.stats import front_distance, window_objectives, window_points
from tests.metrics.test_stats import task, write_run

OBJECTIVE = {"w3": 2.0, "c_drop": 5.0, "e_bar": 3.0, "e_ref": {"T1": 2.0}}


def five_seeds(root: Path) -> list[Path]:
    """On-time shares 0.8, 0.9, 0.7, 0.85, 0.75 (20 tasks each): mean 0.8 ± 0.098162 (hand-checked)."""
    out = []
    for seed, share in zip(range(3000, 3005), (0.8, 0.9, 0.7, 0.85, 0.75), strict=True):
        k = round(share * 20)
        rows = [task(f"t{i}", "on_time" if i < k else "drop_late", 0.1 if i < k else None) for i in range(20)]
        zones = [{"zone": z, "rho": r, "time_s": 0.0} for z, r in (("z1", 0.6), ("z2", 0.3))]
        out.append(write_run(root, seed, rows, zones=zones))
    return out


def test_the_report_reproduces_a_hand_checked_interval(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    five_seeds(tmp_path / "runs")
    (tmp_path / "runs" / "S2-greedy-train").mkdir()  # a training directory: skipped
    files = report([tmp_path / "runs"], tmp_path / "rep", 10)
    names = sorted(f.name for f in files)
    assert names == ["S2-load.png", "S2.png", "report.csv", "report.md"]
    with (tmp_path / "rep" / "report.csv").open(encoding="utf-8") as f:
        rows = {r["metric"]: r for r in csv.DictReader(f) if r["method"] == "greedy"}
    on_time = rows["on_time"]
    assert float(on_time["mean"]) == pytest.approx(0.8)
    assert float(on_time["half"]) == pytest.approx(0.098162, abs=1e-6)
    assert float(on_time["ci_low"]) == pytest.approx(0.701838, abs=1e-6)
    assert on_time["n"] == "5" and on_time["seeds"] == "3000 3001 3002 3003 3004"
    md = (tmp_path / "rep" / "report.md").read_text(encoding="utf-8")
    assert "| greedy | 5 | 80.0 ± 9.8 |" in md
    assert all(f.stat().st_size > 5_000 for f in files if f.suffix == ".png")


def test_run_dirs_finds_runs_and_refuses_none(tmp_path: Path) -> None:
    runs = five_seeds(tmp_path / "runs")
    assert run_dirs([tmp_path / "runs"]) == runs
    assert run_dirs([runs[1], runs[1], runs[0]]) == runs[:2]
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no run directories"):
        run_dirs([tmp_path / "empty"])


# --- the knee-reference windows ----------------------------------------------------------------------------


def wtask(anchor: str, tid: str, outcome: str, latency: float | None, energy: float) -> dict[str, Any]:
    return {**task(tid, outcome, latency, energy), "anchor": anchor, "type": "T1", "deadline_s": 0.1}


def test_window_objectives_by_hand() -> None:
    rows = [
        wtask("z1", "a", "on_time", 0.05, 1.0),  # c 0.5, ê 0.5
        wtask("z1", "b", "late", 0.15, 8.0),  # c 1.5 + 2 x 0.5 = 2.5, ê min(4, 3) = 3
        wtask("z2", "c", "drop_late", None, 0.0),  # c 5, ê 3
        wtask("z2", "a", "on_time", 0.05, 1.0),  # another anchor's task a: not in the window
    ]
    keys = {("z1", "a"), ("z1", "b"), ("z2", "c"), ("z3", "x")}
    f1, f2, n = window_objectives(rows, keys, OBJECTIVE)
    assert n == 3 and f1 == pytest.approx(8 / 3) and f2 == pytest.approx(6.5 / 3)
    assert math.isnan(window_objectives([], keys, OBJECTIVE)[0])


def test_the_distance_to_the_front_is_on_normalised_axes() -> None:
    front = [[0.0, 1.0], [1.0, 0.0]]  # ranges 1 and 1
    assert front_distance((0.0, 1.0), front) == 0.0
    assert front_distance((1.0, 1.0), front) == pytest.approx(1.0)
    wide = [[0.0, 10.0], [2.0, 0.0]]  # ranges 2 and 10: (2, 10) is 1 range-unit from (0, 10)
    assert front_distance((2.0, 10.0), wide) == pytest.approx(1.0)
    assert front_distance((5.0, 5.0), [[5.0, 5.0]]) == 0.0  # one point: raw axes


def test_every_run_on_a_knee_window_is_put_on_it(tmp_path: Path) -> None:
    rows = [wtask("z1", "a", "on_time", 0.05, 1.0), wtask("z1", "b", "late", 0.15, 8.0)]
    knee = write_run(tmp_path, 3000, rows, method="knee-reference")
    other = write_run(tmp_path, 3000, [wtask("z1", "a", "drop_late", None, 0.0)], method="greedy")
    elsewhere = write_run(tmp_path, 3001, rows, method="random")
    for d in (knee, other, elsewhere):
        info = json.loads((d / "run.json").read_text(encoding="utf-8"))
        (d / "run.json").write_text(json.dumps({**info, "objective": OBJECTIVE}), encoding="utf-8")
    window = {"tasks": [["z1", "a"], ["z1", "b"]], "front": [[1.5, 1.75], [3.0, 1.0]]}
    (knee / "window.json").write_text(json.dumps(window), encoding="utf-8")
    pts = window_points([knee, other, elsewhere])
    found = [(p.method, p.seed, p.tasks) for p in pts]
    assert found == [("greedy", 3000, 1), ("knee-reference", 3000, 2)]
    g, k = pts
    assert (k.f1, k.f2, k.distance) == pytest.approx((1.5, 1.75, 0.0))
    assert (g.f1, g.f2) == (5.0, 3.0)
    # ranges 1.5 and 0.75; nearest (1.5, 1.75): (3.5 / 1.5, 1.25 / 0.75); to (3, 1): (2 / 1.5, 2 / 0.75)
    assert g.distance == pytest.approx(math.hypot(3.5 / 1.5, 1.25 / 0.75))


# --- the command line and a real run ------------------------------------------------------------------------


def test_pcnme_report_on_runner_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for m in ("greedy", "random"):
        args = ["run", "--scenario", "S1", "--method", m, "--seeds", "3000-3001", "--seconds", "1"]
        assert main([*args, "--cars", "10", "--out", str(tmp_path / "res")]) == 0
    capsys.readouterr()
    assert main(["report", str(tmp_path / "res"), "--out", str(tmp_path / "rep"), "--no-plots"]) == 0
    assert str(tmp_path / "rep" / "report.md") in capsys.readouterr().out
    with (tmp_path / "rep" / "report.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {(r["scenario"], r["method"]) for r in rows} == {("S1", "greedy"), ("S1", "random")}
    on_time = [r for r in rows if r["metric"] == "on_time"]
    assert all(r["n"] == "2" and 0 < float(r["mean"]) <= 1 for r in on_time)
    assert main(["report", str(tmp_path / "missing"), "--out", str(tmp_path / "rep2")]) == 2
