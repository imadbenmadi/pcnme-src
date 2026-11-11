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


def test_done_rows_of_another_config_or_setup_are_refused_another_commit_is_not(tmp_path: Path) -> None:
    import csv

    f = tmp_path / "summary.csv"
    rows = [row("S1", "greedy", 3000, "testbed", 0.9)]
    with f.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ident = {"commit": "c1", "config_hash": "h1", "setup": "40 cars, 60 s"}
    assert read_done(f, ident) == {("S1", "greedy", 3000, "testbed")}
    assert read_done(f, {**ident, "commit": "c2"}) == {("S1", "greedy", 3000, "testbed")}  # a testbed fix
    with pytest.raises(RuntimeError, match="another config"):
        read_done(f, {**ident, "config_hash": "h2"})
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


def _summary(folder: Path, rows: list[dict[str, str]]) -> None:
    import csv

    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "summary.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_archive_moves_the_chosen_rows_and_their_runs_with_a_label(tmp_path: Path) -> None:
    from pcnme.pipeline.testbed import _read, archive

    rows = [
        {**row("S2", "pcnme", 3000, "testbed", 0.5), "run_dir": "S2-pcnme-3000-testbed"},
        {**row("S2", "pcnme", 3000, "fast", 0.9), "run_dir": "S2-pcnme-3000-fast"},
        {**row("S2", "greedy", 3000, "testbed", 0.9), "run_dir": "S2-greedy-3000-testbed"},
    ]
    _summary(tmp_path, rows)
    for r in rows:
        (tmp_path / "results" / r["run_dir"]).mkdir(parents=True)
    n = archive(tmp_path, lambda r: r["method"] == "pcnme", "frozen (bug)", "frozen-bug")
    assert n == 2
    assert [r["method"] for r in _read(tmp_path / "summary.csv")] == ["greedy"]
    kept = _read(tmp_path / "archive" / "frozen-bug" / "summary.csv")
    assert {r["label"] for r in kept} == {"frozen (bug)"} and len(kept) == 2
    assert (tmp_path / "archive" / "frozen-bug" / "results" / "S2-pcnme-3000-testbed").is_dir()
    assert not (tmp_path / "results" / "S2-pcnme-3000-testbed").exists()


def _run_dir(tmp_path: Path, zones: list[str], counts: dict[str, dict[str, int]], log: str = "") -> Path:
    import json

    d = tmp_path / "S2-pcnme-3000-testbed"
    (d / "zones").mkdir(parents=True)
    for z in zones:
        (d / "zones" / f"{z}.json").write_text("{}", encoding="utf-8")
    for z, c in counts.items():
        (d / f"counts-{z}.json").write_text(json.dumps(c), encoding="utf-8")
    (d / "containers.log").write_text(log, encoding="utf-8")
    return d


def test_health_finds_a_crashed_zone_a_learner_that_does_not_learn_and_lost_acks(tmp_path: Path) -> None:
    """The pre-flight check (Q62): refuse a long run unless every zone wrote its rows, every learning zone
    made online updates through a live trainer, and few tasks ended without an ack."""
    from pcnme.pipeline.testbed import health

    good = {"agent updates": 120, "trainer alive": 1, "no ack by the end (drop)": 0}
    zones = ["z1", "z2", "z3", "z4"]
    ok = _run_dir(tmp_path / "a", zones, dict.fromkeys(zones, good))
    assert health(ok, "pcnme", 0.02) == []
    assert health(ok, "greedy", 0.02) == []
    crashed = _run_dir(tmp_path / "b", zones[1:], {z: good for z in zones[1:]},
                       "Traceback (most recent call last):\nInvalidStateError: CANCELLED")  # fmt: skip
    p = health(crashed, "pcnme", 0.02)
    assert any("crashed" in x for x in p) and any("InvalidStateError" in x for x in p)
    frozen = _run_dir(tmp_path / "c", zones, {**dict.fromkeys(zones, good), "z3": {"agent updates": 0}})
    assert health(frozen, "pcnme", 0.02) == ["z3: no online updates (agent not connected to its trainer?)"]
    assert health(frozen, "greedy", 0.02) == []  # a rule does not learn
