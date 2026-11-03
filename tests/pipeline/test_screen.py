"""The live screen of train.py / evaluate.py: the stage, the jobs' progress and their latest numbers."""

from __future__ import annotations

from rich.console import Console

from pcnme.pipeline.screen import Screen, hms


def test_the_screen_shows_stage_jobs_progress_and_numbers() -> None:
    s = Screen("train.py  map vfc", plain=False)
    s.stage("learning", ["S1 pcnme", "S1 qeco"])
    s.update({"job": "S1 pcnme", "status": "running", "phase": "pretraining (DQfD)", "done": 50, "total": 100,
              "unit": "updates", "numbers": {"agreement": 0.81}})  # fmt: skip
    s.update({"job": "S1 qeco", "status": "failed", "message": "FAILED: boom"})
    console = Console(record=True, width=160)
    console.print(s._render())
    text = console.export_text()
    for part in ("train.py  map vfc", "stage: learning", "S1 pcnme", "pretraining (DQfD)", "50/100 updates",
                 "agreement 0.81", "FAILED: boom", "jobs 0/2 done"):  # fmt: skip
        assert part in text, part
    assert hms(3725) == "1:02:05" and hms(None) == "-"
