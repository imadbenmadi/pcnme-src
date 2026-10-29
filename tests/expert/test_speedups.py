"""NSGA-II speed-ups that change no result (13.2.3): the warm-up reused by fork, windows in parallel. Each is
checked by "the same plan gives exactly the same objectives"."""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from pcnme.config import Config, config_from_mapping, load_yaml
from pcnme.expert import window as win
from pcnme.expert.nsga import run_nsga, run_windows
from pcnme.expert.window import FORK, WarmStart, Window, WindowSpec, discover, evaluate, window_seed
from tests.zone.harness import CONFIG

needs_fork = pytest.mark.skipif(not FORK, reason="fork: Linux and WSL only")


@pytest.fixture(scope="module")
def source() -> dict[str, Any]:
    data: dict[str, Any] = copy.deepcopy(load_yaml(CONFIG))
    data["nsga2"]["T_warmup"] = 1.0
    data["nsga2"]["T_win"] = 0.3
    return data


@pytest.fixture(scope="module")
def cfg(source: dict[str, Any]) -> Config:
    return config_from_mapping(source)


def spec(cfg: Config, k: int) -> WindowSpec:
    return WindowSpec("S2", window_seed(cfg, "S2", k), k, n_cars=40)


@pytest.fixture(scope="module")
def window(cfg: Config) -> Window:
    return discover(cfg, spec(cfg, 0))


@needs_fork
def test_the_warm_start_gives_exactly_the_plain_objectives(cfg: Config, window: Window) -> None:
    warm = WarmStart(cfg, window)
    rng = random.Random(9)
    genomes = [[rng.randrange(len(t.options)) for t in window.tasks] for _ in range(3)]
    for g in [*genomes, list(window.greedy_genome)]:
        assert warm.evaluate(g) == evaluate(cfg, window, g)[0]
    assert warm.evaluate(genomes[0]) == warm.evaluate(genomes[0])  # the parent's warm state never moves
    assert warm.run.t < cfg.nsga2.t_warmup


@needs_fork
def test_nsga_with_and_without_the_warm_start_gives_the_same_front(
    source: dict[str, Any], window: Window
) -> None:
    a = run_nsga(source, window, population=4, generations=2, warm=True)
    b = run_nsga(source, window, population=4, generations=2, warm=False)
    assert (a.F, a.genomes) == (b.F, b.genomes)


def test_windows_in_parallel_give_each_windows_own_front(source: dict[str, Any], cfg: Config) -> None:
    specs = [spec(cfg, 0), spec(cfg, 1)]
    together = run_windows(source, specs, population=4, generations=2, workers=2)
    alone = [run_nsga(source, discover(cfg, s), population=4, generations=2) for s in specs]
    assert [(f.F, f.genomes) for f in together] == [(f.F, f.genomes) for f in alone]
    assert [f.window.spec for f in together] == specs


def test_without_fork_the_warm_start_refuses(
    monkeypatch: pytest.MonkeyPatch, cfg: Config, window: Window
) -> None:
    monkeypatch.setattr(win, "FORK", False)
    with pytest.raises(RuntimeError, match="fork"):
        WarmStart(cfg, window)


@needs_fork
def test_the_workers_use_the_warm_start_on_linux(source: dict[str, Any], window: Window) -> None:
    import sys

    from pcnme.expert import nsga

    assert sys.platform.startswith("linux") == FORK
    nsga._init_worker(source, window, warm=True)
    assert isinstance(nsga._WORKER["warm"], WarmStart)
    nsga._init_worker(source, window, warm=False)
    assert nsga._WORKER["warm"] is None
