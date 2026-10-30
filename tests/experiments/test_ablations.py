"""Ablations and sweeps as runner options (PLAN M7; final-design 16.4; QUESTIONS Q47, Q48), each on a tiny
run. The switches themselves are tested in tests/zone/test_ablation.py and tests/test_sweep.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pq = pytest.importorskip("pyarrow.parquet")

from pcnme.config import DEFAULT_CONFIG_PATH, load_yaml  # noqa: E402
from pcnme.experiments.cli import main  # noqa: E402
from pcnme.experiments.runner import (  # noqa: E402
    ABLATIONS,
    FACTORS,
    RunSpec,
    run,
    sweep_points,
    w_points,
    with_overrides,
)

TINY = ["--seeds", "3000", "--seconds", "1", "--cars", "10"]


@pytest.fixture(scope="module")
def source() -> dict[str, Any]:
    return load_yaml(DEFAULT_CONFIG_PATH)


def info(d: Path) -> dict[str, Any]:
    return json.loads((d / "run.json").read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def cli(tmp_path: Path, method: str, *extra: str, scenario: str = "S1") -> list[Path]:
    args = ["run", "--scenario", scenario, "--method", method, *TINY, "--out", str(tmp_path), *extra]
    assert main(args) == 0
    return sorted(p.parent for p in tmp_path.rglob("run.json"))


def test_overrides_set_existing_keys_only(source: dict[str, Any]) -> None:
    s = with_overrides(source, [("tof.alpha", 0.5), ("stateplane.r3.enabled", True)])
    assert s["tof"]["alpha"] == 0.5 and s["stateplane"]["r3"]["enabled"] is True
    assert source["tof"]["alpha"] == 0.8  # the original is left alone
    for bad in ("tof.alfa", "nope.x", "tof.alpha.x"):
        with pytest.raises(ValueError, match="unknown config key"):
            with_overrides(source, [(bad, 1)])


def test_the_sweeps_points(source: dict[str, Any]) -> None:
    sp = sweep_points(source, "state-period")
    assert [v for v, _ in sp] == [
        "state-10ms",
        "state-10ms-r3",
        "state-100ms",
        "state-100ms-r3",
        "state-500ms",
        "state-500ms-r3",
    ]
    assert dict(sp[5][1]) == {
        "stateplane.state_period": 0.5,
        "stateplane.stale_after": 1.5,
        "stateplane.r3.enabled": True,
    }
    assert [v for v, _ in sweep_points(source, "alpha")] == [
        "alpha-0.5",
        "alpha-0.65",
        "alpha-0.8",
        "alpha-1.0",
    ]
    cars = sweep_points(source, "cars")
    assert [dict(o)["sweep.cars"] for _, o in cars] == list(FACTORS) and cars[0][0] == "cars-x0.5"
    assert w_points([0.25, 1.0]) == [("w-0.25", (("reward.w", 0.25),)), ("w-1", (("reward.w", 1.0),))]
    with pytest.raises(ValueError, match="unknown sweep"):
        sweep_points(source, "gamma")


def test_an_override_needs_a_variant(source: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="variant"):
        run(source, RunSpec("S1", "greedy", (3000,), overrides=(("tof.alpha", 0.5),)), tmp_path)


@pytest.mark.parametrize("name", sorted(ABLATIONS))
def test_every_named_ablation_runs_in_its_own_directory(tmp_path: Path, name: str) -> None:
    (d,) = cli(tmp_path, "greedy", "--ablation", name)
    assert d.parent.name == name and d.name == "S1-greedy-3000-fast"
    i = info(d)
    assert i["variant"] == name and [tuple(o) for o in i["overrides"]] == [tuple(o) for o in ABLATIONS[name]]
    assert i["broker_messages"] > 0


def test_a_learner_runs_with_gamma_0(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    (d,) = cli(tmp_path, "d3qn-scratch", "--ablation", "gamma-0", "--budget", "60", "--train-seconds", "1")
    assert info(d)["training"]["budget"]["used"] == 60
    assert (tmp_path / "gamma-0" / "S1-d3qn-scratch-train" / "model.pt").exists()


def test_the_ablations_change_the_runs(tmp_path: Path) -> None:
    (d,) = cli(tmp_path / "n", "greedy", "--ablation", "no-neighbours", "--ablation", "no-predictor")
    assert d.parent.name == "no-neighbours+no-predictor"
    rows = pq.read_table(d / "decisions.parquet").to_pylist()
    assert {r["action"] for r in rows} <= {"here", "cloud"} and {r["x_next_stays"] for r in rows} == {1.0}


def test_set_overrides_any_key_with_a_yaml_value(tmp_path: Path) -> None:
    (d,) = cli(tmp_path, "greedy", "--set", "tof.alpha=0.65", "--set", "ablation.tof=false")
    assert d.parent.name == "tof.alpha=0.65+ablation.tof=false"
    assert [tuple(o) for o in info(d)["overrides"]] == [("tof.alpha", 0.65), ("ablation.tof", False)]
    assert main(["run", "--scenario", "S1", "--method", "greedy", *TINY, "--set", "tof.alfa=1"]) == 2
    assert main(["run", "--scenario", "S1", "--method", "greedy", *TINY, "--ablation", "no-gpu"]) == 2


def test_the_state_period_sweep_and_the_broker_load(tmp_path: Path) -> None:
    dirs = cli(tmp_path, "greedy", "--sweep", "state-period")
    assert [d.parent.name for d in dirs] == sorted(
        v for v, _ in sweep_points(load_yaml(DEFAULT_CONFIG_PATH), "state-period")
    )
    msgs = {d.parent.name: info(d)["broker_messages"] for d in dirs}
    assert msgs["state-10ms"] > msgs["state-100ms"] > msgs["state-500ms"]  # state messages dominate


def test_a_sensitivity_sweep_and_the_report(tmp_path: Path) -> None:
    dirs = cli(tmp_path / "res", "greedy", "--sweep", "cars")
    fleets = {d.parent.name: info(d)["fleet"] for d in dirs}
    assert fleets == {"cars-x0.5": 5, "cars-x0.75": 8, "cars-x1.0": 10, "cars-x1.25": 12, "cars-x1.5": 15}
    assert main(["report", str(tmp_path / "res" / "cars-x0.5"), str(tmp_path / "res" / "cars-x1.5"), "--out",
                 str(tmp_path / "rep"), "--no-plots"]) == 0  # fmt: skip
    md = (tmp_path / "rep" / "report.md").read_text(encoding="utf-8")
    assert "| greedy [cars-x0.5] | 1 |" in md and "| greedy [cars-x1.5] | 1 |" in md


def test_the_w_sweep_sets_input_48(tmp_path: Path) -> None:
    dirs = cli(tmp_path, "greedy", "--w", "0.25", "1.0")
    assert [d.parent.name for d in dirs] == ["w-0.25", "w-1"]
    for d, w in zip(dirs, (0.25, 1.0), strict=True):
        assert info(d)["reward_w"] == w
        assert {r["x_w"] for r in pq.read_table(d / "decisions.parquet").to_pylist()} == {w}
