"""The testbed entry points (PLAN M9.1 step 6; final-design 15.4): a zone on RealClock with real-network
calls, the car endpoints at its intake (POST /cars, POST /gps, GET /closed), and the cloud."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from pcnme.testbed.clock import RealClock
from pcnme.testbed.codec import decode, encode
from pcnme.testbed.messages import Attach, Gps, GpsBatch
from pcnme.testbed.radio import RemoteCar
from tests.zone.harness import fixed_delay_config

pytest.importorskip("fastapi")


class FakeBus:
    """MqttBus's interface with no broker: every client is alive and publishes go nowhere."""

    def connect(self, client_id: str, delay: Any, on_message: Any, will: Any = None) -> Any:
        class C:
            alive = True

            def subscribe(self, topic: str) -> int:
                return 1

            def unsubscribe(self, topic: str) -> int:
                return 0

            def count(self, topic: str) -> int:
                return 0

            def publish(self, *a: Any, **k: Any) -> None:
                pass

        return C()

    def is_alive(self, client_id: str) -> bool:
        return True


def test_a_testbed_zone_takes_cars_gps_and_lists_closed_tasks() -> None:
    from fastapi.testclient import TestClient

    from pcnme.testbed.zone import build_zone, zone_app

    cfg = fixed_delay_config()
    clock = RealClock()
    from pcnme.testbed.netloop import NetLoop

    net = NetLoop(clock)
    zone, closed = build_zone(
        cfg, "z1", "S2", 3000, clock, FakeBus(), {"cloud": "http://cloud:8000"}, net=net
    )
    assert zone.calls.real_network and cfg.radio.uplink_delay != (0.0, 0.0)
    assert zone.calls._urls["z1"] == "http://127.0.0.1:8000"  # its own executor (watchdog probes, 9.4)
    assert set(zone.peers) == {"z2", "z3", "z4"}  # the other active zones, for the predictor (10.2)
    assert zone._cfg.radio.uplink_delay == (0.0, 0.0)  # netem gives the radio delays (15.3)
    threading.Thread(target=clock.run_for, args=(1.0,), name="zone-loop", daemon=True).start()
    http = TestClient(zone_app(clock, zone, closed, net))
    assert http.post("/cars", content=encode(Attach("c000", 120.0, "http://car-1:8000"))).status_code == 200
    car, rate = zone._cars["c000"]
    assert isinstance(car, RemoteCar) and rate > 0
    x, y = cfg.layout.zone_positions_m["z1"]
    batch = GpsBatch((Gps("c000", x, y, 1.0, 0.0, 0.5),))
    reply = http.post("/gps", content=encode(batch))
    assert reply.status_code == 200 and decode(reply.content) == []
    assert "c000" in zone.registry
    stranger = GpsBatch((Gps("c999", x, y, 1.0, 0.0, 0.6),))  # a car this zone does not serve (Q61)
    assert decode(http.post("/gps", content=encode(stranger)).content) == ["c999"]
    assert decode(http.get("/closed").content) == []
    st = decode(http.get("/status").content)
    assert isinstance(st, dict) and st["cars"] == 1 and st["closed"] == 0
    clock.stop()


def test_a_zone_writes_its_anchored_records_as_rows(tmp_path: Any) -> None:
    from pcnme.testbed.runner import merge
    from pcnme.testbed.zone import write_rows
    from tests.zone.harness import Fixed, one_zone, task_now

    s = one_zone(fixed_delay_config(), Fixed("here"))
    task = task_now(s, "T1")
    s.zone.upload(task)
    s.clock.run_until(1.0)
    path = write_rows(s.cfg, "S2", "greedy", 3000, "z1", s.closed, tmp_path)
    assert path == tmp_path / "S2-greedy-3000-testbed" / "zones" / "z1.json"
    log = merge(path.parent, "S2", "greedy", 3000)
    (row,) = log.rows["tasks"]
    assert row["task_id"] == str(task.task_id) and row["outcome"] == "on_time" and row["mode"] == "testbed"


def test_a_zone_deploys_the_trained_method_from_the_models_folder(tmp_path: Any) -> None:
    """PLAN M9.1 step 7 (Q54): the learning methods run in the zone container, deployed as in fast mode."""
    pytest.importorskip("torch")
    from pcnme.experiments.runner import deployed
    from pcnme.methods.learners import d3qn_session, run_session
    from pcnme.methods.simple import Greedy
    from pcnme.pipeline import checkpoint
    from pcnme.sim.rng import RandomStreams
    from pcnme.testbed.zone import zone_placement

    cfg = fixed_delay_config()
    zones = cfg.layout.active_zones
    trained = run_session(cfg, d3qn_session(cfg, "d3qn-scratch", "S1", zones, 200, seconds=2.0, n_cars=20))
    (tmp_path / "S1-d3qn-scratch").mkdir()
    checkpoint.save(trained, tmp_path / "S1-d3qn-scratch" / "trained.pt")
    streams = RandomStreams.from_config(cfg, 3000, "evaluation")
    z = zones[0]
    agent = zone_placement(cfg, streams, z, "d3qn-scratch", "S1", 3000, tmp_path)
    assert type(agent) is type(deployed(cfg, trained, 3000)(cfg, streams, z))
    assert isinstance(zone_placement(cfg, streams, z, "greedy", "S1", 3000, tmp_path), Greedy)
    with pytest.raises(FileNotFoundError):
        zone_placement(cfg, streams, z, "pcnme", "S1", 3000, tmp_path)


def test_a_testbed_learning_agent_is_connected_to_its_trainer(tmp_path: Any) -> None:
    """13.5, Q42, Q62: on the testbed the deployed agent keeps learning online, in its own trainer process.
    The zone attaches the agent to its transitions (as the city does), the agent's samples go to the trainer,
    and the trainer's weights come back and are installed on the zone's loop. Fails if any link is missing
    (found 11 Nov: the testbed agents never learned)."""
    pytest.importorskip("torch")
    from pcnme.agent.nstep import Sample
    from pcnme.methods.learners import d3qn_session, run_session
    from pcnme.pipeline import checkpoint
    from pcnme.testbed.netloop import NetLoop
    from pcnme.testbed.zone import build_zone

    cfg = fixed_delay_config()
    zones = cfg.layout.active_zones
    trained = run_session(cfg, d3qn_session(cfg, "d3qn-scratch", "S1", zones, 200, seconds=2.0, n_cars=20))
    (tmp_path / "S1-d3qn-scratch").mkdir()
    checkpoint.save(trained, tmp_path / "S1-d3qn-scratch" / "trained.pt")
    clock = RealClock()
    zone, _ = build_zone(cfg, "z1", "S1", 3000, clock, FakeBus(), {"cloud": "http://cloud:8000"},
                         "d3qn-scratch", net=NetLoop(clock), models=tmp_path)  # fmt: skip
    agent: Any = zone._placement
    trainer: Any = zone.trainer  # type: ignore[attr-defined]
    try:
        assert zone.transitions.on_replay_hook == agent._transition  # fed by its zone, as in the city
        assert trainer.connected and agent.trainer is trainer
        before = {k: v.clone() for k, v in agent.online.state_dict().items()}
        s = tuple(0.1 * (i % 7) for i in range(48))
        sample = Sample(s, 0, -1.0, (True,) * 6, False, True, s, (True,) * 6, 0.9,
                        False, 0, 0.0, None, None, 0.0)  # fmt: skip
        agent._add([sample] * 400)  # the zone does not learn in place: the samples go to the trainer
        threading.Thread(target=clock.run_for, args=(60.0,), name="zone-loop", daemon=True).start()
        deadline = threading.Event()
        for _ in range(600):
            if agent.updates > 0:
                break
            deadline.wait(0.1)
        assert trainer.sent == 400 and agent.new_samples >= 400
        assert agent.updates > 0 and trainer.installed > 0  # weights came back and were installed
        after = agent.online.state_dict()
        assert any(not before[k].equal(after[k]) for k in before)
    finally:
        clock.stop()
        trainer.close()
    assert not trainer.process.is_alive()
