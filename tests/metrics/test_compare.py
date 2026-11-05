"""Comparisons across scenarios and maps from summary.csv rows (PLAN T3c)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pcnme.metrics.compare import across_scenarios, paired, paired_rows, transfer_rows, write_all


def test_the_paired_t_test_matches_a_hand_computed_case() -> None:
    # differences 1, 2, 3: mean 2, sd 1, t = 2 / (1 / sqrt 3) = 3.464, df 2, two-sided p = 0.0742
    r = paired([2.0, 4.0, 6.0], [1.0, 2.0, 3.0])
    assert r.n == 3 and r.diff == pytest.approx(2.0)
    assert r.t == pytest.approx(math.sqrt(12.0))
    assert r.p == pytest.approx(0.0742, abs=1e-4)


def test_identical_differences_are_certain_and_no_difference_is_none() -> None:
    assert paired([2.0, 3.0], [1.0, 2.0]).p == 0.0
    assert paired([1.0, 2.0], [1.0, 2.0]).p == 1.0
    assert math.isnan(paired([1.0], [0.0]).p)  # one seed: no test


def _rows() -> list[dict[str, str]]:
    out = []
    for sc in ("S1", "S2"):
        for me, base in (("pcnme", 0.9), ("greedy", 0.8)):
            for seed in (3000, 3001, 3002):
                v = base + 0.01 * (seed - 3000)
                out.append({"scenario": sc, "method": me, "seed": str(seed), "on_time": str(v),
                            "latency_p99_s": str(1 - v), "mean_e_hat": str(v),
                            "energy_per_task_j": "1.0"})  # fmt: skip
    return out


def test_pcnme_is_paired_with_each_baseline_on_the_same_seeds_only() -> None:
    rows = _rows()
    rows.append({**rows[-1], "method": "greedy", "seed": "3999"})  # a seed pcnme lacks: not paired
    out = paired_rows(rows, "pcnme", ("on_time",))
    assert [(o["scenario"], o["baseline"]) for o in out] == [("S1", "greedy"), ("S2", "greedy")]
    assert out[0]["n"] == 3 and out[0]["diff"] == pytest.approx(0.1) and out[0]["p"] < 1e-6


def test_across_scenarios_gives_each_method_one_mean_per_scenario() -> None:
    m = across_scenarios(_rows(), "on_time")
    assert list(m) == ["greedy", "pcnme"]
    assert m["pcnme"]["S1"] == pytest.approx(0.91)


def test_transfer_rows_pair_the_home_and_the_other_map() -> None:
    home, other = _rows(), [{**r, "on_time": str(float(r["on_time"]) - 0.1)} for r in _rows()]
    out = transfer_rows(home, other, ("on_time",), "vfc", "lust")
    assert len(out) == 4
    assert out[0]["vfc"] == pytest.approx(out[0]["lust"] + 0.1)


def test_write_all_makes_the_tables_and_figures(tmp_path: Path) -> None:
    files = write_all(_rows(), tmp_path)
    names = {f.name for f in files}
    assert {"paired.csv", "paired.md", "across-on_time.png", "across-latency_p99_s.png"} <= names
    assert all(f.exists() for f in files)
