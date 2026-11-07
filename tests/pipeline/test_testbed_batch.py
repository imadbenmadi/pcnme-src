"""run_testbed.py's batch (src/pcnme/pipeline/testbed.py): what is left to run, and the fast-mode vs testbed
gap table (PLAN M9.4; final-design 16.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcnme.pipeline.testbed import GAP_METRICS, gap_rows, gap_table, plan_runs, read_done


def row(
    scenario: str, method: str, seed: int, mode: str, on_time: float, commit: str = "c1"
) -> dict[str, str]:
    r = {"scenario": scenario, "method": method, "seed": str(seed), "mode": mode, "commit": commit,
         "config_hash": "h1", "setup": "40 cars, 60 s"}  # fmt: skip
    r.update({k: "0.5" for k in GAP_METRICS})
    r["on_time"] = str(on_time)
    return r


def test_the_plan_is_seed_major_and_skips_what_is_done() -> None:
    done = {("S1", "greedy", 3000, "testbed"), ("S1", "greedy", 3000, "fast")}
    todo = plan_runs(["S1", "S2"], ["greedy", "pcnme"], [3000, 3001], done)
    assert todo[0] == ("S1", "pcnme", 3000)  # S1 greedy 3000 is done in both modes
    assert [t[2] for t in todo] == [3000] * 3 + [3001] * 4  # every scenario and method of a seed first
    assert ("S1", "greedy", 3000) not in todo
    half = {("S1", "greedy", 3000, "testbed")}  # its fast twin is missing: run it again (both modes)
    assert ("S1", "greedy", 3000) in plan_runs(["S1"], ["greedy"], [3000], half)


def test_done_rows_of_another_commit_are_refused(tmp_path: Path) -> None:
    import csv

    f = tmp_path / "summary.csv"
    rows = [row("S1", "greedy", 3000, "testbed", 0.9)]
    with f.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ident = {"commit": "c1", "config_hash": "h1", "setup": "40 cars, 60 s"}
    assert read_done(f, ident) == {("S1", "greedy", 3000, "testbed")}
    with pytest.raises(RuntimeError, match="another config or commit"):
        read_done(f, {**ident, "commit": "c2"})
    with pytest.raises(RuntimeError, match="setup"):
        read_done(f, {**ident, "setup": "40 cars, 20 s"})  # a shorter test batch never mixes in


def test_the_gap_is_testbed_minus_fast_mode_over_the_seeds_both_ran() -> None:
    rows = [
        row("S2", "pcnme", 3000, "fast", 0.99),
        row("S2", "pcnme", 3000, "testbed", 0.89),
        row("S2", "pcnme", 3001, "fast", 0.97),
        row("S2", "pcnme", 3001, "testbed", 0.91),
        row("S2", "pcnme", 3002, "fast", 0.95),
    ]  # fmt: skip  (seed 3002 has no testbed run: left out)
    g = gap_rows(rows)
    assert len(g) == 1 and g[0]["seeds"] == 2
    assert g[0]["on_time_fast"] == pytest.approx(0.98) and g[0]["on_time_testbed"] == pytest.approx(0.90)
    assert g[0]["on_time_gap"] == pytest.approx(-0.08)
    md = gap_table(g)
    assert "| S2 | PCNME | 2 | 98.0 | 90.0 | -8.0 |" in md
