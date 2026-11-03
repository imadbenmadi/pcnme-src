"""train.py's resume rules: a training stopped at an episode boundary and restarted from its checkpoint ends
exactly as an uninterrupted one; outputs of another config or commit are refused, never mixed."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pcnme.methods.learners import d3qn_session, run_session
from pcnme.pipeline import checkpoint, runinfo
from tests.zone.harness import fixed_delay_config


class Stop(Exception):
    pass


def test_a_resumed_training_ends_exactly_like_an_uninterrupted_one(tmp_path: Path) -> None:
    cfg = fixed_delay_config()
    zones = cfg.layout.active_zones

    def session() -> object:
        return d3qn_session(cfg, "d3qn-scratch", "S1", zones, 500, seconds=3.0, n_cars=30)

    whole = run_session(cfg, session())  # type: ignore[arg-type]

    path = tmp_path / "checkpoint.pt"

    def stop_after_first(s: object) -> None:
        checkpoint.save(s, path)
        raise Stop

    with pytest.raises(Stop):
        run_session(cfg, session(), checkpoint=stop_after_first)  # type: ignore[arg-type]
    resumed = run_session(cfg, checkpoint.load(path))
    assert len(resumed.episodes) == len(whole.episodes)
    assert [e.on_time for e in resumed.episodes] == [e.on_time for e in whole.episodes]
    for z in zones:
        for k, v in whole.states[z].items():
            assert torch.equal(v, resumed.states[z][k]), (z, k)


def test_outputs_of_another_config_are_refused(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    runinfo.check_or_start(manifest, {"config_hash": "aaa", "commit": "c1", "plan": {"budget": (1, 2)}}, {})
    runinfo.check_or_start(manifest, {"config_hash": "aaa", "commit": "c1", "plan": {"budget": (1, 2)}}, {})
    with pytest.raises(runinfo.Mismatch, match="config_hash"):
        runinfo.check_or_start(
            manifest, {"config_hash": "bbb", "commit": "c1", "plan": {"budget": (1, 2)}}, {}
        )
