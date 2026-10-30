"""Metrics and confidence intervals from Parquet (PLAN M7 acceptance; final-design 16.3; QUESTIONS Q46).

Synthetic run directories with hand-picked rows, so every number is checked by hand.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from pcnme.metrics.stats import (  # noqa: E402
    classify,
    early_on_time,
    interval,
    jain,
    run_metrics,
    summarise,
    t_cdf,
    t_quantile,
    worst_share,
    zone_loads,
)


def task(
    tid: str, outcome: str, latency: float | None, energy: float = 1.0, wasted: float = 0.0
) -> dict[str, Any]:
    return {
        "task_id": tid,
        "outcome": outcome,
        "latency_s": latency,
        "energy_j": energy,
        "energy_wasted_j": wasted,
        "method": "greedy",
    }


def write_run(
    root: Path,
    seed: int,
    tasks: list[dict[str, Any]],
    *,
    method: str = "greedy",
    events: list[dict[str, Any]] | None = None,
    attempts: list[dict[str, Any]] | None = None,
    zones: list[dict[str, Any]] | None = None,
    decisions: list[dict[str, Any]] | None = None,
) -> Path:
    d = root / f"S2-{method}-{seed}-fast"
    d.mkdir(parents=True)
    info = {"scenario": "S2", "method": method, "seed": seed, "seconds": 10.0, "idle_w": 2.0}
    (d / "run.json").write_text(json.dumps(info), encoding="utf-8")
    tables = {"tasks": tasks, "events": events, "attempts": attempts, "zones": zones, "decisions": decisions}
    for name, rows in tables.items():
        if rows:
            pq.write_table(pa.Table.from_pylist(rows), d / f"{name}.parquet")
    return d


# --- Student t (16.3) ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("df", "t"), [(1, 12.7062), (2, 4.3027), (4, 2.7764), (9, 2.2622), (19, 2.0930), (100, 1.9840)]
)
def test_t_quantiles_match_the_tables(df: int, t: float) -> None:
    assert t_quantile(0.975, df) == pytest.approx(t, abs=1e-4)  # 16.3: t = 2.78 (4 df), 2.09 (19 df)
    assert t_quantile(0.025, df) == pytest.approx(-t, abs=1e-4)
    assert t_quantile(0.5, df) == pytest.approx(0.0, abs=1e-9)
    q = t_quantile(0.975, df)
    assert t_cdf(-q, df) == pytest.approx(0.025, abs=1e-9) and t_cdf(q, df) == pytest.approx(0.975, abs=1e-9)
    with pytest.raises(ValueError):
        t_quantile(1.0, df)


def test_a_hand_checked_interval() -> None:
    # mean 0.8; deviations 0, .1, -.1, .05, -.05: Σd² = 0.025, s² = 0.025/4, s = 0.0790569, s/√5 = 0.0353553,
    # half = 2.776445 × 0.0353553 = 0.098162
    ci = interval([0.8, 0.9, 0.7, 0.85, 0.75])
    assert ci.n == 5 and ci.mean == pytest.approx(0.8)
    assert ci.half == pytest.approx(0.098162, abs=1e-6)
    assert (ci.low, ci.high) == pytest.approx((0.701838, 0.898162), abs=1e-6)
    one = interval([0.4])
    assert one.mean == 0.4 and math.isnan(one.half) and one.n == 1
    assert interval([math.nan, 1.0, 3.0]).n == 2  # a run without the metric is left out


def test_jain_and_the_worst_share() -> None:
    assert jain([1, 1, 1, 1]) == 1.0
    assert jain([1, 0, 0, 0]) == 0.25
    assert jain([0.2, 0.6]) == pytest.approx(0.64 / (2 * 0.40))
    assert worst_share([float(x) for x in range(1, 201)]) == 199.5  # ceil(2) largest: 199, 200
    assert worst_share([3.0, 1.0]) == 3.0
    assert worst_share([float(x) for x in range(1, 151)]) == 149.5  # ceil(1.5) = 2: 149, 150


def test_outcomes_map_to_the_four_shares() -> None:
    assert [classify(o) for o in ("on_time", "late", "drop_lost", "drop_no_ack")] == [
        "on_time",
        "late",
        "failed",
        "failed",
    ]
    assert {classify(o) for o in ("drop_intake", "drop_masked", "drop_refused_full", "drop_late")} == {
        "dropped"
    }
    with pytest.raises(ValueError):
        classify("delivered")


# --- one run ------------------------------------------------------------------------------------------------


def test_the_metrics_of_one_run_by_hand(tmp_path: Path) -> None:
    tasks = [
        task("a", "on_time", 0.05, 2.0),
        task("b", "on_time", 0.07, 2.0),
        task("c", "late", 0.30, 3.0, wasted=1.0),
        task("d", "drop_lost", None, 1.0, wasted=1.0),
        task("e", "drop_refused_full", None, 0.0),
    ]
    events = [{"kind": "car_drop"}, {"kind": "uncovered"}, {"kind": "uncovered"}, {"kind": "handover"}]
    attempts = [{"result": r} for r in ("delivered", "refused_full", "refused_late", "lost", "delivered")]
    zones = [
        {"zone": "z1", "rho": 0.5, "time_s": 0.0},
        {"zone": "z1", "rho": 0.7, "time_s": 9.9},
        {"zone": "z2", "rho": 0.2, "time_s": 0.0},
        {"zone": "z2", "rho": 0.2, "time_s": 5.0},
        {"zone": "z2", "rho": 0.0, "time_s": 10.0},  # the drain after the 10 s episode: left out
    ]
    decisions = [
        {"time_s": 1.0, "anchor": "z1", "task_id": "b", "kind": "first"},
        {"time_s": 0.5, "anchor": "z2", "task_id": "a", "kind": "first"},
        {"time_s": 0.7, "anchor": "z1", "task_id": "d", "kind": "redecision"},
        {"time_s": 2.0, "anchor": "z1", "task_id": "c", "kind": "first"},
    ]
    r = run_metrics(
        write_run(tmp_path, 3000, tasks, events=events, attempts=attempts, zones=zones, decisions=decisions),
        2,
    )
    m = r.metrics
    assert (r.scenario, r.method, r.seed) == ("S2", "greedy", 3000)
    covered = 6  # 5 tasks held + 1 car drop
    assert m["on_time"] == pytest.approx(2 / covered) and m["late"] == pytest.approx(1 / covered)
    assert m["failed"] == pytest.approx(1 / covered) and m["dropped"] == pytest.approx(2 / covered)
    assert m["on_time"] + m["late"] + m["failed"] + m["dropped"] == pytest.approx(1.0)
    assert m["uncovered"] == pytest.approx(2 / 8)
    assert m["latency_mean_s"] == pytest.approx(0.14)
    assert m["latency_p99_s"] == pytest.approx(0.07 + 0.98 * (0.30 - 0.07))  # linear between the top two
    assert m["latency_worst1_s"] == pytest.approx(0.30)
    assert m["energy_dynamic_j"] == 8.0 and m["energy_per_task_j"] == pytest.approx(8 / 6)
    assert m["energy_total_j"] == 8.0 + 2.0 * 10.0
    assert m["wasted_share"] == pytest.approx(2 / 8)
    assert m["refusal_rate"] == pytest.approx(2 / 5)
    assert m["jain_load"] == pytest.approx(jain([0.6, 0.2]))
    assert m["early_on_time"] == 1.0  # the first 2 first decisions: a, b (both on time)


def test_zone_loads_and_early_on_time() -> None:
    rows = [{"zone": "z2", "rho": 1.0, "time_s": 0.0}, {"zone": "z1", "rho": 0.0, "time_s": 0.1}]
    rows += [{"zone": "z1", "rho": 0.5, "time_s": 0.2}, {"zone": "z1", "rho": 1.0, "time_s": 0.3}]
    assert zone_loads(rows) == {"z1": 0.5, "z2": 1.0}
    assert zone_loads(rows, 0.3) == {"z1": 0.25, "z2": 1.0}  # the period at the episode's end is out
    tasks = [task("a", "late", 0.2), task("b", "on_time", 0.1)]
    decisions = [
        {"time_s": float(i), "anchor": "z1", "task_id": t, "kind": "first"} for i, t in enumerate("ab")
    ]
    assert early_on_time(decisions, tasks, 1) == 0.0 and early_on_time(decisions, tasks, 5) == 0.5
    assert math.isnan(early_on_time([], tasks, 5))


# --- across seeds -------------------------------------------------------------------------------------------


def test_summary_over_seeds_by_hand(tmp_path: Path) -> None:
    # on-time shares 0.8, 0.9, 0.7, 0.85, 0.75 over 20 tasks each: the hand-checked interval above
    runs = []
    for seed, share in zip(range(3000, 3005), (0.8, 0.9, 0.7, 0.85, 0.75), strict=True):
        k = round(share * 20)
        rows = [task(f"t{i}", "on_time" if i < k else "drop_late", 0.1 if i < k else None) for i in range(20)]
        runs.append(run_metrics(write_run(tmp_path, seed, rows), 10))
    other = run_metrics(write_run(tmp_path, 3000, [task("x", "on_time", 0.1)], method="random"), 10)
    s = summarise([*runs, other])
    assert list(s) == [("S2", "greedy"), ("S2", "random")]
    ci = s[("S2", "greedy")]["on_time"]
    assert ci.n == 5 and ci.mean == pytest.approx(0.8) and ci.half == pytest.approx(0.098162, abs=1e-6)
    assert s[("S2", "random")]["on_time"].n == 1
    with pytest.raises(ValueError, match="twice"):
        summarise([runs[0], runs[0]])
