"""The run's logs (PLAN M3.10; final-design 16.3; docs/METRICS.md; QUESTIONS Q28)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcnme.config import Config, load_config
from pcnme.experiments.city import City
from pcnme.metrics.runlog import EPOCH_NS, line_protocol

pq = pytest.importorskip("pyarrow.parquet")


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


@pytest.fixture(scope="module")
def run_dir(cfg: Config, tmp_path_factory: pytest.TempPathFactory) -> Path:
    city = City(cfg, "S2", "greedy", 2000, n_cars=40).run(3.0)
    return city.write(tmp_path_factory.mktemp("results"))


def table(d: Path, name: str) -> list[dict[str, object]]:
    return pq.read_table(d / f"{name}.parquet").to_pylist()  # type: ignore[no-any-return]


def test_the_run_writes_every_table_under_its_run_id(run_dir: Path) -> None:
    # METRICS.md: results/<run_id>/, run_id = <scenario>-<method>-<seed>-<mode>.
    assert run_dir.name == "S2-greedy-2000-fast"
    for name in ("tasks", "attempts", "zones", "pipes", "positions", "events"):
        assert (run_dir / f"{name}.parquet").exists()
    assert (run_dir / "run.json").exists() and (run_dir / "points.lp").exists()


def test_every_row_has_the_common_tags(run_dir: Path) -> None:
    for name in ("tasks", "zones", "positions"):
        row = table(run_dir, name)[0]
        tags = (row["run_id"], row["scenario"], row["method"], row["seed"], row["mode"])
        assert tags == ("S2-greedy-2000-fast", "S2", "greedy", 2000, "fast")


def test_every_car_position_every_100_ms(run_dir: Path) -> None:
    # Q28: one row per car per GPS period: 40 cars, ticks at 0, 0.1, 0.2, ... until the drain ends.
    rows = table(run_dir, "positions")
    ticks = sorted({round(float(r["time_s"]), 6) for r in rows})  # type: ignore[arg-type]
    assert ticks[:3] == [0.0, 0.1, 0.2]
    assert len(rows) == 40 * len(ticks)
    assert set(rows[0]) >= {"car_id", "x_m", "y_m", "speed_mps", "heading_x", "heading_y", "zone"}


def test_attach_and_handover_events(run_dir: Path) -> None:
    rows = table(run_dir, "events")
    kinds = {r["kind"] for r in rows}
    assert "attach" in kinds and kinds <= {"attach", "handover", "gap", "upload_cut", "car_drop", "uncovered"}
    assert all(r["zone_to"] in ("z1", "z2", "z3", "z4") for r in rows if r["kind"] in ("attach", "handover"))


def test_attempts_carry_their_option_executor_slot_and_timestamps(run_dir: Path) -> None:
    rows = [r for r in table(run_dir, "attempts") if r["result"] == "delivered"]
    assert rows
    for r in rows:
        assert r["option"] in ("here", "n1", "n2", "n3", "n4", "cloud") and r["slot_type"] in ("gpu", "cpu")
        times = [r["t_send_s"], r["t_at_executor_s"], r["t_start_s"], r["t_finish_s"], r["t_result_at_car_s"]]
        assert times == sorted(times)  # type: ignore[type-var]


def test_zone_rows_every_state_period_per_slot_type(run_dir: Path) -> None:
    rows = table(run_dir, "zones")
    want = {(z, q) for z in ("z1", "z2", "z3", "z4") for q in ("gpu", "cpu")}
    assert {(r["zone"], r["slot_type"]) for r in rows} == want
    fields = {"queue_hard", "queue_soft", "busy", "rho", "refusals", "wan_wait_est_s"}
    assert all(set(r) >= fields for r in rows[:5])


def test_task_rows_follow_metrics_md(run_dir: Path) -> None:
    rows = table(run_dir, "tasks")
    fields = {
        "anchor",
        "type",
        "cls",
        "outcome",
        "boulder",
        "latency_s",
        "deadline_s",
        "attempts",
        "energy_j",
        "energy_wasted_j",
        "upload_s",
        "forward_s",
        "wait_s",
        "run_s",
        "return_s",
        "zone_changed",
    }
    assert rows and set(rows[0]) >= fields


def test_the_points_file_has_one_line_per_row(run_dir: Path) -> None:
    names = ("tasks", "attempts", "zones", "pipes", "positions", "events", "decisions", "transitions")
    n_rows = sum(len(table(run_dir, n)) for n in names)
    lines = (run_dir / "points.lp").read_text(encoding="utf-8").splitlines()
    assert len(lines) == n_rows


def test_line_protocol_by_hand() -> None:
    # METRICS.md: fast-mode points are stamped 2026-01-01T00:00:00Z + the simulated time.
    row = {
        "run_id": "S2-greedy-1-fast",
        "scenario": "S2",
        "method": "greedy",
        "seed": 1,
        "mode": "fast",
        "time_s": 1.5,
        "zone": "z1",
        "slot_type": "gpu",
        "busy": 2,
        "rho": 1.0,
        "note": "a b",
    }
    line = line_protocol("zone", row, ("zone", "slot_type"))
    assert line == (
        "zone,run_id=S2-greedy-1-fast,scenario=S2,method=greedy,seed=1,mode=fast,zone=z1,slot_type=gpu "
        f'busy=2i,rho=1.0,note="a b" {EPOCH_NS + 1_500_000_000}'
    )
