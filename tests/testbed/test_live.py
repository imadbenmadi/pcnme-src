"""Live metrics to InfluxDB (src/pcnme/testbed/live.py; PLAN R2b): off by default, never blocking a zone."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from pcnme.testbed.live import LivePusher, line


def test_a_line_is_influx_line_protocol_with_typed_fields() -> None:
    s = line("live_task", {"zone": "z1", "outcome": "on_time"}, {"latency_s": 0.05, "n": 3, "ok": True}, 123)
    assert s == "live_task,zone=z1,outcome=on_time latency_s=0.05,n=3i,ok=true 123"
    assert line("m", {"run_id": "S2 x"}, {"v": 1.0, "skip": None}, 1) == r"m,run_id=S2\ x v=1.0 1"


class Recorder:
    def __init__(self, status: int = 204) -> None:
        self.status = status
        self.posts: list[tuple[str, bytes, Mapping[str, str]]] = []

    def __call__(self, url: str, body: bytes, headers: Mapping[str, str]) -> int:
        self.posts.append((url, body, headers))
        if self.status == 0:
            raise OSError("refused")
        return self.status


def test_points_are_batched_and_posted_with_the_token() -> None:
    rec = Recorder()
    p = LivePusher("http://influx:8086", "pcnme", "pcnme", "tok", {"run_id": "r1", "zone": "z1"}, post=rec,
                   start=False)  # fmt: skip
    p.task("on_time", "T1", "hard", 0.04, 0.1)
    p.add("live_zone", {}, {"busy_gpu": 2})
    p.flush()
    url, body, headers = rec.posts[0]
    assert url == "http://influx:8086/api/v2/write?org=pcnme&bucket=pcnme&precision=ns"
    assert headers["Authorization"] == "Token tok" and body.count(b"\n") == 1
    assert b"live_task,run_id=r1,zone=z1,outcome=on_time,type=T1,cls=hard" in body and p.sent == 2
    p.flush()
    assert len(rec.posts) == 1  # nothing new: no post


@pytest.mark.parametrize("status", [0, 500])
def test_a_failed_post_drops_the_points_and_never_raises(status: int) -> None:
    p = LivePusher("http://influx:8086", "o", "b", "t", {}, post=Recorder(status), start=False)
    p.add("m", {}, {"v": 1})
    p.flush()
    assert p.failed == 1 and p.sent == 0


def test_live_metrics_are_off_unless_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PCNME_LIVE", raising=False)
    assert LivePusher.from_env({}) is None
