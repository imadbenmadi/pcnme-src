"""The scenario runner and `pcnme run` (PLAN M7; final-design 16.1, 16.3; QUESTIONS Q41-Q43, Q45).

Evaluation runs on seeds 3000-3999 only (16.3), so these tests use them: the runner refuses any other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pq = pytest.importorskip("pyarrow.parquet")

from pcnme.config import DEFAULT_CONFIG_PATH, load_yaml  # noqa: E402
from pcnme.experiments.cli import main  # noqa: E402
from pcnme.experiments.runner import RunSpec, check_seeds, parse_seeds, run  # noqa: E402
from pcnme.sim.rng import SeedError  # noqa: E402

TINY = {"seconds": 1.0, "n_cars": 10}


@pytest.fixture(scope="module")
def source() -> dict[str, Any]:
    return load_yaml(DEFAULT_CONFIG_PATH)


def table(d: Path, name: str) -> list[dict[str, Any]]:
    return pq.read_table(d / f"{name}.parquet").to_pylist()  # type: ignore[no-any-return]


def info(d: Path) -> dict[str, Any]:
    return json.loads((d / "run.json").read_text(encoding="utf-8"))  # type: ignore[no-any-return]


# --- seeds (16.3) ------------------------------------------------------------------------------------------


def test_seed_lists_and_ranges() -> None:
    assert parse_seeds("3000-3002") == (3000, 3001, 3002)
    assert parse_seeds("3000,3005, 3007-3008") == (3000, 3005, 3007, 3008)
    for bad in ("", "3002-3000", "3001-3000", "30a0", "3000,3000", "3000-"):
        with pytest.raises(ValueError):
            parse_seeds(bad)


@pytest.mark.parametrize("seeds", [(2000,), (1500,), (3999, 4000), ()])
def test_the_runner_refuses_seeds_outside_the_evaluation_range(
    source: dict[str, Any], tmp_path: Path, seeds: tuple[int, ...]
) -> None:
    with pytest.raises(SeedError):
        run(source, RunSpec("S1", "greedy", seeds, **TINY), tmp_path)  # type: ignore[arg-type]
    assert not any(tmp_path.iterdir())  # refused before anything ran


def test_training_seeds_are_checked_too(source: dict[str, Any], tmp_path: Path) -> None:
    from pcnme.config import config_from_mapping

    cfg = config_from_mapping(source)
    check_seeds(cfg, (2000, 2999), "training")
    with pytest.raises(SeedError):
        check_seeds(cfg, (3000,), "training")
    with pytest.raises(SeedError):
        run(source, RunSpec("S1", "d3qn-scratch", (3000,), budget=10, train_seed=3000, **TINY), tmp_path)  # type: ignore[arg-type]


def test_unknown_methods_and_scenarios_are_refused(source: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="method"):
        run(source, RunSpec("S1", "dqn", (3000,)), tmp_path)
    with pytest.raises(ValueError, match="scenario"):
        run(source, RunSpec("S9", "greedy", (3000,)), tmp_path)


# --- rules --------------------------------------------------------------------------------------------------


def test_a_rule_runs_each_seed_and_writes_its_logs(source: dict[str, Any], tmp_path: Path) -> None:
    out = run(source, RunSpec("S1", "random", (3000, 3001), **TINY), tmp_path)  # type: ignore[arg-type]
    assert [d.name for d in out.runs] == ["S1-random-3000-fast", "S1-random-3001-fast"]
    for d, seed in zip(out.runs, (3000, 3001), strict=True):
        tasks = table(d, "tasks")
        assert tasks and {(t["method"], t["seed"], t["scenario"]) for t in tasks} == {("random", seed, "S1")}
        i = info(d)
        assert (i["method"], i["usage"], i["seed"], i["training"]) == ("random", "evaluation", seed, None)
    assert out.trained is None


def test_the_same_spec_gives_the_same_logs(source: dict[str, Any], tmp_path: Path) -> None:
    a = run(source, RunSpec("S1", "greedy", (3000,), **TINY), tmp_path / "a")  # type: ignore[arg-type]
    b = run(source, RunSpec("S1", "greedy", (3000,), **TINY), tmp_path / "b")  # type: ignore[arg-type]
    for name in ("tasks", "attempts", "decisions"):
        assert table(a.runs[0], name) == table(b.runs[0], name)


# --- learning methods (16.1) --------------------------------------------------------------------------------

torch_needed = pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None, reason="torch (extra ml) not installed"
)


@torch_needed
def test_a_learner_trains_on_its_budget_then_is_evaluated(source: dict[str, Any], tmp_path: Path) -> None:
    spec = RunSpec("S1", "d3qn-scratch", (3000, 3001), budget=150, train_seconds=1.0, **TINY)  # type: ignore[arg-type]
    out = run(source, spec, tmp_path)
    assert out.trained is not None and out.trained.budget.used == out.trained.budget.total == 150
    t = info(out.runs[0])["training"]
    assert t["budget"] == {"total": 150, "used": 150} and t["episodes"][0]["seed"] == 2000
    assert (tmp_path / "S1-d3qn-scratch-train" / "training.json").exists()
    assert (tmp_path / "S1-d3qn-scratch-train" / "model.pt").exists()
    for d in out.runs:
        assert {r["method"] for r in table(d, "decisions")} == {"d3qn-scratch"}


@torch_needed
def test_the_default_budget_is_the_config_s(source: dict[str, Any]) -> None:
    from pcnme.config import config_from_mapping
    from pcnme.experiments.runner import budget_of

    cfg = config_from_mapping(source)
    assert budget_of(cfg, RunSpec("S1", "qeco", (3000,))) == cfg.fast_mode.train_decisions == 8_750_000
    assert budget_of(cfg, RunSpec("S1", "qeco", (3000,), budget=7)) == 7


@torch_needed
def test_vfc_ppo_runs_with_w_1(source: dict[str, Any], tmp_path: Path) -> None:
    out = run(source, RunSpec("S1", "vfc-ppo", (3000,), budget=100, train_seconds=1.0, **TINY), tmp_path)  # type: ignore[arg-type]
    d = out.runs[0]
    assert info(d)["reward_w"] == 1.0  # Q43: delay only
    assert {r["x_w"] for r in table(d, "decisions")} == {1.0}


@torch_needed
def test_pcnme_needs_its_demonstrations(source: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="make_demos"):  # the runner's own message, before training
        run(source, RunSpec("S1", "pcnme", (3000,), budget=10, **TINY), tmp_path)  # type: ignore[arg-type]


# --- the knee reference (16.1; QUESTIONS Q45) ---------------------------------------------------------------


@pytest.mark.slow
def test_the_knee_reference_replays_its_plan_on_the_evaluation_window(
    source: dict[str, Any], tmp_path: Path
) -> None:
    pytest.importorskip("pymoo")
    src = {**source, "nsga2": {**source["nsga2"], "T_warmup": 1.0, "T_win": 0.3}}
    spec = RunSpec(
        "S2", "knee-reference", (3000,), seconds=3.0, n_cars=40, nsga_population=6, nsga_generations=2
    )
    d = run(src, spec, tmp_path).runs[0]
    w = json.loads((d / "window.json").read_text(encoding="utf-8"))
    assert w["usage"] == "evaluation" and w["seed"] == 3000 and w["tasks"] and w["applied"] > 0
    # the episode up to the horizon is the plan's evaluation: the replayed window gives the planned point
    assert w["replayed"] == [w["f1"], w["f2"]] and [w["f1"], w["f2"]] in w["front"]
    assert info(d)["knee"]["replayed"] == w["replayed"]
    # the plan is the front's knee, and evaluating its genome on the same window gives the same point
    from pcnme.config import config_from_mapping
    from pcnme.expert.knee import knee
    from pcnme.expert.window import WindowSpec, discover, evaluate

    k = knee([tuple(p) for p in w["front"]])
    assert k is not None and list(k.point) == [w["f1"], w["f2"]] and w["w"] == k.w
    cfg = config_from_mapping(src)
    window = discover(cfg, WindowSpec("S2", 3000, 0, 40, usage="evaluation"))
    obj, _ = evaluate(cfg, window, w["genome"])
    assert w["replayed"] == [obj.f1, obj.f2]


def test_the_knee_reference_needs_the_whole_window(source: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="window"):
        run(source, RunSpec("S2", "knee-reference", (3000,), seconds=5.0), tmp_path)


# --- the command line ---------------------------------------------------------------------------------------


def test_pcnme_run_on_the_command_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["run", "--scenario", "S1", "--method", "greedy", "--seeds", "3000-3001", "--seconds", "1"]
    assert main([*args, "--cars", "10", "--out", str(tmp_path)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["S1-greedy-3000-fast", "S1-greedy-3001-fast"]
    assert "S1-greedy-3001-fast" in capsys.readouterr().out
    assert (
        main(["run", "--scenario", "S1", "--method", "greedy", "--seeds", "2000", "--out", str(tmp_path)])
        == 2
    )
    assert "outside the evaluation range" in capsys.readouterr().err
