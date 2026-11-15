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


def test_every_training_job_pins_torch_before_it_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PLAN Q4 (15 Nov): a resumed agent comes unpickled from its checkpoint and never runs its own setup
    (agent/network.py deterministic()), so a resumed worker trained with torch's default threads and its
    losses differed from the uninterrupted run in the 9th digit. Each job now sets it first."""
    from pcnme.pipeline import train

    real = train._deterministic
    torch.set_num_threads(3)
    real()
    assert torch.get_num_threads() == 1  # what a job sets
    calls: list[int] = []
    monkeypatch.setattr(train, "_deterministic", lambda: calls.append(1))
    monkeypatch.setenv("PCNME_DATA_DIR", str(tmp_path))
    d = train.MapPaths("vfc").train("S1", "pcnme")
    d.mkdir(parents=True)
    (d / "trained.pt").write_bytes(b"")
    plan = train.Plan("vfc", ("S1",), 1, None, None, None, None, None, None, 1, 60.0)
    assert train._train_job("vfc", "S1", "pcnme", 0.4, plan, lambda **k: None) == "skipped"
    assert calls == [1]  # set by the job itself, whatever path it takes


def test_a_resume_keeps_the_decisions_used_of_the_checkpointed_episodes(tmp_path: Path) -> None:
    """15 Nov: the resume rewrote episodes.csv from the checkpoint with decisions_used 0 and wall_s 0."""
    from types import SimpleNamespace

    from pcnme.pipeline.train import _rewrite_episodes

    (tmp_path / "episodes.csv").write_text(
        "episode,seed,on_time,mean_e_hat,tasks,updates,mean_loss,decisions_used,wall_s\n"
        "1,2000,0.9,1.1,298,74,0.1,298,54.3\n2,2001,0.8,1.2,287,82,0.4,400,56.9\n3,2002,0.7,1.3,280,90,0.5,500,60.0\n",
        encoding="utf-8",
    )
    ep = SimpleNamespace(seed=2000, on_time=0.9, mean_e_hat=1.1, tasks=298, updates=74, mean_loss=0.1)
    ep2 = SimpleNamespace(seed=2001, on_time=0.8, mean_e_hat=1.2, tasks=287, updates=82, mean_loss=0.4)
    _rewrite_episodes(tmp_path, SimpleNamespace(out=SimpleNamespace(episodes=[ep, ep2])))
    rows = (tmp_path / "episodes.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3  # header + the 2 checkpointed episodes; episode 3 is redone after the resume
    assert rows[1].endswith(",298,54.3") and rows[2].endswith(",400,56.9")
