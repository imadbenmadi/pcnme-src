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


def test_string_fields_are_quoted_and_escaped_and_non_finite_floats_left_out() -> None:
    s = line("m", {"zone": "z1"}, {"row": r'a "b" \c', "x": float("nan"), "y": 1.5}, 7)
    assert s == r'm,zone=z1 row="a \"b\" \\c",y=1.5 7'


def test_producers_add_their_lines_on_the_metrics_thread() -> None:
    rec = Recorder()
    p = LivePusher("http://influx:8086", "o", "b", "t", {}, post=rec, start=False)
    p.producer(lambda: ["m v=1i 1"])
    p.flush()
    assert rec.posts[0][1] == b"m v=1i 1" and p.sent == 1


def test_agent_fields_report_the_online_learning_and_nothing_for_a_rule() -> None:
    from pcnme.testbed.live import agent_fields

    class Learner:
        updates, new_samples = 12, 30
        losses = ((12, 0.1, 0.2, 0.3, 0.6, 1.5),)

        def current_epsilon(self) -> float:
            return 0.05

    assert agent_fields(object()) == {}
    assert agent_fields(Learner()) == {
        "updates": 12,
        "samples": 30,
        "loss": 0.6,
        "mean_q": 1.5,
        "epsilon": 0.05,
    }


def test_a_closed_task_goes_out_as_its_rows_in_wall_time_and_the_viewer_reads_it_back() -> None:
    """The live row is the METRICS.md row (record_rows) with the zone's clock times moved to wall time; the
    viewer's live source (sources/live.py) turns it into the replay's task columns, relative to the run."""
    import json

    from pcnme.testbed.live import LiveZone
    from pcnme.viewer.sources.live import task_object
    from tests.zone.harness import Fixed, fixed_delay_config, one_zone, task_now

    s = one_zone(fixed_delay_config(), Fixed("here"))
    task = task_now(s, "T1")
    s.zone.upload(task)
    s.clock.run_until(1.0)
    (record,) = s.closed
    rec = Recorder()
    p = LivePusher("http://i:8086", "o", "b", "t", {"run": "r1", "zone": "z1"}, post=rec, start=False)
    live = LiveZone(p, s.zone, s.cfg, s.clock)
    live.offset = 1000.0  # wall = clock time + 1000 s
    live.closed(record)
    p.flush()
    text = rec.posts[0][1].decode()
    assert text.startswith("live_task,run=r1,zone=z1,outcome=on_time,type=T1")
    raw = text.split(',row="', 1)[1].rsplit('" ', 1)[0].replace('\\"', '"').replace("\\\\", "\\")
    row = json.loads(raw)
    assert row["task"]["time_s"] == pytest.approx(task.created + 1000.0)
    assert row["attempts"][0]["t_send_s"] == pytest.approx(row["attempts"][0]["time_s"])
    obj = task_object(row, 1000.0)  # t0: the run's first point
    assert obj["id"] == str(task.task_id) and obj["outcome"] == "on_time"
    assert obj["t"] == pytest.approx(task.created, abs=1e-4)
    assert obj["close"] == pytest.approx(task.created + obj["latency"], abs=1e-3)
    assert obj["att"][0]["executor"] == "z1" and obj["att"][0]["car"] is not None


def test_a_live_zone_samples_its_slots_and_the_cars_it_serves() -> None:
    from pcnme.testbed.live import LiveZone
    from tests.zone.harness import Fixed, fixed_delay_config, one_zone

    s = one_zone(fixed_delay_config(), Fixed("here"))
    rec = Recorder()
    p = LivePusher("http://i:8086", "o", "b", "t", {"run": "r1", "zone": "z1"}, post=rec, start=False)
    live = LiveZone(p, s.zone, s.cfg, s.clock, sample_s=0.2)
    served = list(s.zone._cars)
    for c in [*served, "stranger"]:
        live.fix(c, 10.0, 20.0, 3.0, 4.0)
    live.start()
    s.clock.run_until(0.5)
    p.flush()
    lines = rec.posts[0][1].decode().splitlines()
    zone_lines = [x for x in lines if x.startswith("live_zone,")]
    assert len(zone_lines) == 2 and "busy_gpu=" in zone_lines[0] and "records=" in zone_lines[0]
    cars = [x for x in lines if x.startswith("live_car,")]
    assert len(cars) == 2 * len(served) and not any("car=stranger" in x for x in cars)
    if served:
        assert "x=10.0,y=20.0,v=5.0" in cars[0]
