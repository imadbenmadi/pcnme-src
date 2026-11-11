"""The viewer's live kinds (PLAN V3, V4): the live testbed read from InfluxDB (sources/live.py) and the
training read from train.py's files (sources/training.py), and their API routes (server.py)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from pcnme.viewer.server import api
from pcnme.viewer.sources.base import Registry, SourceInfo
from pcnme.viewer.sources.live import LiveSource, _ns, parse_csv, read_env
from pcnme.viewer.sources.training import TrainingSource

T0 = "2026-11-11T12:00:00Z"
T0_NS = _ns(T0)


def test_influx_csv_tables_with_their_own_headers_are_read_as_records() -> None:
    text = (
        ",result,table,_time,_value\r\n,_result,0,2026-11-11T12:00:00Z,1\r\n\r\n"
        ",result,table,_time,zone\r\n,_result,1,2026-11-11T12:00:01Z,z2\r\n"
    )
    rows = parse_csv(text)
    assert rows == [
        {"": "", "result": "_result", "table": "0", "_time": "2026-11-11T12:00:00Z", "_value": "1"},
        {"": "", "result": "_result", "table": "1", "_time": "2026-11-11T12:00:01Z", "zone": "z2"},
    ]


def test_influx_times_keep_their_nanoseconds() -> None:
    assert _ns("2026-11-11T12:00:00.000000123Z") - _ns(T0) == 123
    assert _ns("2026-11-11T12:00:01.5Z") - _ns(T0) == 1_500_000_000


def test_the_influx_settings_come_from_docker_env(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text('# secrets\nINFLUX_ORG=pcnme\nINFLUX_TOKEN="tok=1"\n\n', encoding="utf-8")
    assert read_env(f) == {"INFLUX_ORG": "pcnme", "INFLUX_TOKEN": "tok=1"}
    assert read_env(tmp_path / "none") == {}


class FakeInflux:
    """Answers each query by the measurement it filters (the Flux text), recording the queries."""

    bucket = "pcnme"

    def __init__(self, answers: dict[str, list[dict[str, str]]], down: bool = False) -> None:
        self.answers, self.down, self.queries = answers, down, []  # type: ignore[var-annotated]

    def query(self, flux: str) -> list[dict[str, str]]:
        self.queries.append(flux)
        if self.down:
            raise OSError("connection refused")
        for key, rows in self.answers.items():
            if key in flux:
                return rows
        return []


def _row(created: float) -> str:
    task = {
        "task_id": "c003.0.7",
        "time_s": created,
        "anchor": "z2",
        "type": "T1",
        "cls": "hard",
        "outcome": "on_time",
        "attempts": 1,
        "deadline_s": 0.1,
        "latency_s": 0.04,
        "upload_s": 0.01,  # sent right at t_rx: the car and anchor clocks agree here
        "energy_j": 2.0,
    }
    att = {
        "attempt": 0,
        "executor": "z2",
        "option": "here",
        "slot_type": "gpu",
        "result": "delivered",
        "t_send_s": created + 0.01,
        "t_at_executor_s": created + 0.011,
        "t_start_s": created + 0.012,
        "t_finish_s": created + 0.03,
        "t_result_at_car_s": created + 0.04,
        "energy_j": 2.0,
    }
    return json.dumps({"task": task, "attempts": [att]})


def live(down: bool = False) -> tuple[LiveSource, FakeInflux]:
    t0 = T0_NS / 1e9
    fake = FakeInflux(
        {
            "|> first()": [{"_time": T0}],
            '"live_task"': [{"_time": "2026-11-11T12:00:02Z", "_value": _row(t0 + 1.5)}],
            '"live_zone" and r.run == "r1")': [
                {
                    "_time": "2026-11-11T12:00:01Z",
                    "zone": "z2",
                    "scenario": "S2",
                    "method": "pcnme",
                    "seed": "3000",
                    "busy_gpu": "2",
                    "queued_gpu": "1",
                    "updates": "40",
                    "loss": "0.5",
                }
            ],
            '"live_car"': [
                {
                    "_time": "2026-11-11T12:00:01.2Z",
                    "car": "c003",
                    "zone": "z2",
                    "x": "120.04",
                    "y": "80",
                    "v": "11.1",
                }
            ],
            "|> last()": [
                {
                    "_time": "2026-11-11T12:10:00Z",
                    "run": "r1",
                    "scenario": "S2",
                    "method": "pcnme",
                    "seed": "3000",
                },
                {
                    "_time": "2026-11-11T13:00:00Z",
                    "run": "r2",
                    "scenario": "S4",
                    "method": "greedy",
                    "seed": "3001",
                },
            ],
        },
        down,
    )
    return LiveSource(SourceInfo("live", "live-testbed", "Live testbed", map="vfc"), fake), fake  # type: ignore[arg-type]


def test_live_runs_are_listed_newest_first_and_an_absent_influx_is_said_not_raised() -> None:
    src, _ = live()
    got = src.runs()
    assert got["online"] and [r["run"] for r in got["runs"]] == ["r2", "r1"]
    off = live(down=True)[0].runs()
    assert not off["online"] and off["runs"] == [] and "run_live.py" in off["help"]


def test_the_first_delta_has_the_city_and_everything_relative_to_the_runs_first_point() -> None:
    src, _ = live()
    d = src.delta("r1")
    assert d["meta"]["scenario"] == "S2" and d["meta"]["method"] == "pcnme" and d["meta"]["mode"] == "live"
    assert {z["id"] for z in d["layout"]["zones"]} >= {"z1", "z2", "z3", "z4"}
    (task,) = d["tasks"]
    assert task["id"] == "c003.0.7" and task["car"] == "c003" and task["t"] == pytest.approx(1.5)
    assert task["close"] == pytest.approx(1.54) and task["att"][0]["finish"] == pytest.approx(1.53)
    assert d["zones"] == [["z2", 1.0, {"busy_gpu": 2.0, "queued_gpu": 1.0, "updates": 40.0, "loss": 0.5}]]
    assert d["cars"] == [["c003", "z2", 1.2, 120.0, 80.0, 11.1]]
    assert d["cursor"] == T0_NS + 2_000_000_000  # the newest point seen


def test_a_later_delta_reads_from_a_little_before_its_cursor_and_skips_the_city() -> None:
    src, fake = live()
    cursor = T0_NS + 30_000_000_000
    d = src.delta("r1", cursor)
    assert "layout" not in d and d["cursor"] == cursor  # nothing newer than the cursor in the answers
    assert f"time(v: {cursor - 3_000_000_000})" in fake.queries[-1]  # a zone posts once a second


def _train_dir(root: Path, name: str, episodes: int, log_end: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    rows = ["episode,seed,on_time,mean_e_hat,tasks,updates,mean_loss,decisions_used,wall_s"]
    rows += [f"{i},{1999 + i},0.9{i},1.2,9000,{100 * i},0.5,{1000 * i},60" for i in range(1, episodes + 1)]
    (d / "episodes.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (d / "train.log").write_text(f"start\n{log_end}\n", encoding="utf-8")
    (d / "training.json").write_text(json.dumps({"budget": {"total": 5000, "used": 1000 * episodes}}),
                                     encoding="utf-8")  # fmt: skip
    return d


def test_training_runs_say_running_done_or_stopped(tmp_path: Path) -> None:
    root = tmp_path / "train"
    _train_dir(root, "S2-pcnme", 3, "episode 3: on time 0.93")
    _train_dir(root, "S1-d3qn-greedy", 5, "2026-11-04 07:06:03  done: 5 episodes in 300 s")
    old = _train_dir(root, "S3-qeco", 2, "episode 2: on time 0.92")
    for f in old.iterdir():
        os.utime(f, (1.0, 1.0))  # written long ago, never finished
    src = TrainingSource(SourceInfo("training", "live-training", "Training"), [("vfc", root)])
    states = {r["run"]: (r["state"], r["episodes"], r["method"]) for r in src.runs()}
    assert states == {
        "vfc/S1-d3qn-greedy": ("done", 5, "d3qn-greedy"),
        "vfc/S2-pcnme": ("running", 3, "pcnme"),
        "vfc/S3-qeco": ("stopped", 2, "qeco"),
    }
    c = src.curve("vfc/S2-pcnme")
    assert c["episodes"]["on_time"] == [0.91, 0.92, 0.93] and c["episodes"]["updates"] == [
        100.0,
        200.0,
        300.0,
    ]
    assert c["budget"] == {"total": 5000, "used": 3000} and c["log"][-1] == "episode 3: on time 0.93"
    with pytest.raises(KeyError):
        src.curve("vfc/S9-none")


def test_the_api_routes_the_live_kinds_by_source(tmp_path: Path) -> None:
    src, _ = live()
    root = tmp_path / "train"
    _train_dir(root, "S2-pcnme", 2, "episode 2")
    training = TrainingSource(SourceInfo("training", "live-training", "Training"), [("vfc", root)])
    reg = Registry([src, training])  # type: ignore[list-item]
    assert api(reg, "/api/live/runs", {"source": "live"})["online"]
    d: Any = api(reg, "/api/live", {"source": "live", "run": "r1", "since": "0"})
    assert d["tasks"][0]["id"] == "c003.0.7"
    assert api(reg, "/api/training/runs", {"source": "training"})[0]["run"] == "vfc/S2-pcnme"
    assert api(reg, "/api/training", {"source": "training", "run": "vfc/S2-pcnme"})["state"] == "running"
    with pytest.raises(KeyError):
        api(reg, "/api/live", {"source": "training", "run": "r1"})  # the wrong kind
