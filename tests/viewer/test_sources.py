"""The viewer's replay source (src/pcnme/viewer/sources/replay.py): a run's logs as one replay."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcnme.viewer.sources.base import SourceInfo
from pcnme.viewer.sources.replay import ParquetSource, frames, parse_run_id

pytest.importorskip("pyarrow")


def test_run_ids_split_into_scenario_method_seed_and_mode() -> None:
    r = parse_run_id("S3-d3qn-greedy-3000-fast")
    assert r is not None and (r.scenario, r.method, r.seed, r.mode) == ("S3", "d3qn-greedy", 3000, "fast")
    assert parse_run_id("training-curves") is None


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from pcnme.config import load_config
    from pcnme.experiments.city import City

    out = tmp_path_factory.mktemp("runs")
    City(load_config(), "S1", "greedy", 3000, n_cars=20, usage="evaluation").run(3.0).write(
        out, {"method": "greedy"}
    )
    return out


def test_a_run_becomes_one_replay_with_every_task_and_attempt(run_dir: Path) -> None:
    src = ParquetSource(SourceInfo("vfc", "replay", "VFC grid", map="vfc"), "vfc", [run_dir])
    runs = src.runs()
    assert [r.run_id for r in runs] == ["S1-greedy-3000-fast"]
    rp = src.replay(runs[0].run_id)
    import pyarrow.parquet as pq

    d = run_dir / runs[0].run_id
    assert len(rp["tasks"]["id"]) == pq.read_metadata(d / "tasks.parquet").num_rows
    assert len(rp["attempts"]["task"]) == pq.read_metadata(d / "attempts.parquet").num_rows
    t = rp["tasks"]["t"]
    assert t == sorted(t)  # the page finds the tasks around a time by bisection
    assert all(0 <= k < len(t) for k in rp["attempts"]["task"])
    assert all(c >= s for c, s in zip(rp["tasks"]["close"], t, strict=True))
    n = frames(rp["meta"]["end"])
    assert len(rp["cars"]["ids"]) == 20 and all(len(x) == n for x in rp["cars"]["x"])
    assert set(rp["zones"]) == {"z1", "z2", "z3", "z4"} and "wan" in rp["pipes"]
    assert [z["id"] for z in rp["layout"]["zones"]] == ["z1", "z2", "z3", "z4", "z5"]


def test_an_unknown_run_is_a_key_error(run_dir: Path) -> None:
    src = ParquetSource(SourceInfo("vfc", "replay", "VFC grid", map="vfc"), "vfc", [run_dir])
    with pytest.raises(KeyError):
        src.replay("S9-nothing-1-fast")
