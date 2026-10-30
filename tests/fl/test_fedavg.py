"""FedAvg rounds (PLAN M8.1; final-design 13.7, 4.2, 5.2; QUESTIONS Q50)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pcnme.config import Config, load_config  # noqa: E402
from pcnme.fl.fedavg import FedClient, FedServer, FedState, average, model_bits  # noqa: E402
from tests.zone.harness import Fixed, city, fixed_delay_config  # noqa: E402


class Learner:
    """A stand-in for the zone's agent: one weight tensor, a count of new experiences."""

    def __init__(self, value: float, new: int) -> None:
        self.w = {"w": torch.full((10,), value)}
        self.new_experiences = new
        self.global_version = 0
        self.installed: list[int] = []

    def shared_state(self) -> dict[str, object]:
        return self.w

    def mark_uploaded(self) -> None:
        self.new_experiences = 0

    def install(self, weights: dict[str, object], version: int) -> None:
        self.w = {k: v.clone() for k, v in weights.items()}  # type: ignore[attr-defined]
        self.global_version = version
        self.installed.append(version)


def fed_city(learners: dict[str, Learner], **overrides: object):  # type: ignore[no-untyped-def]
    cfg = fixed_delay_config(**overrides)
    c = city(cfg, lambda z: Fixed(None), zone_ids=tuple(learners))
    server = FedServer(c.clock, cfg, c.cloud.mqtt, FedState())
    for z, zone in c.zones.items():
        zone.fl = FedClient(
            z,
            c.clock,
            c.cloud.executor._streams,
            cfg,
            zone.mqtt,
            zone._cloud_path,
            server,
            learners[z],
            alive=lambda zone=zone: zone.mqtt.alive,
        )
    return c, server


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


def test_the_average_is_weighted_by_new_experiences() -> None:
    # 13.7.3: z1 (weights 1, 30 new) and z2 (weights 4, 10 new): 1 x 0.75 + 4 x 0.25 = 1.75.
    got = average({"z1": ({"w": torch.full((3,), 1.0)}, 30), "z2": ({"w": torch.full((3,), 4.0)}, 10)})
    assert torch.allclose(got["w"], torch.full((3,), 1.75))


def test_the_upload_size_is_every_parameter_at_32_bits() -> None:
    assert model_bits({"a": torch.zeros(10, 4), "b": torch.zeros(4)}) == 44 * 32


def test_a_round_averages_the_uploads_and_every_zone_installs_the_new_version() -> None:
    # 13.7.1-3, 18: a round every T_fl = 60 s (fast mode); the upload deadline 30 s later; the zones that sent
    # weights get version 1 = their average by new experiences; a zone that sent nothing gets it too.
    learners = {"z1": Learner(1.0, 30), "z2": Learner(4.0, 10), "z3": Learner(9.0, 0)}
    c, server = fed_city(learners)
    c.clock.run_until(60.0 + 30.0 - 0.001)
    assert server.state.version == 0
    c.clock.run_until(92.0)
    assert server.state.version == 1 and server.state.rounds[0] == (1, 1, ("z1", "z2"))
    for le in learners.values():
        assert le.installed == [1] and torch.allclose(le.w["w"], torch.full((10,), 1.75))


def test_a_zone_with_no_new_experiences_sends_nothing() -> None:
    learners = {"z1": Learner(1.0, 0), "z2": Learner(4.0, 0)}
    c, server = fed_city(learners)
    bits = c.zones["z1"]._cloud_path._wan.bits_sent
    c.clock.run_until(100.0)
    assert server.state.version == 0 and server.state.rounds[0] == (1, 0, ())
    assert c.zones["z1"]._cloud_path._wan.bits_sent == bits


def test_the_upload_crosses_the_shared_wan_pipe() -> None:
    # 4.2: FedAvg weights travel the WAN pipe like any cloud transfer.
    learners = {"z1": Learner(1.0, 5), "z2": Learner(2.0, 0)}
    c, _ = fed_city(learners)
    wan = c.zones["z1"]._cloud_path._wan
    before = wan.bits_sent
    c.clock.run_until(61.0)
    assert wan.bits_sent - before == pytest.approx(10 * 32)


def test_a_late_upload_is_left_out_and_the_round_never_waits() -> None:
    # 13.7.4: a 10 ms deadline is shorter than the WAN crossing (3 + 62.5 ms): the upload arrives late and is
    # left out; there is no new version, and the next round still starts at 120 s.
    learners = {"z1": Learner(1.0, 5), "z2": Learner(2.0, 5)}
    c, server = fed_city(learners, fedavg__upload_deadline=0.010)
    c.clock.run_until(119.0)
    assert server.state.version == 0 and server.state.k == 1
    assert c.zones["z1"].fl.uploads == [(1, False)]
    c.clock.run_until(121.0)
    assert server.state.k == 2


def test_an_offline_zone_is_left_out() -> None:
    learners = {"z1": Learner(1.0, 5), "z2": Learner(3.0, 5)}
    c, server = fed_city(learners)
    c.clock.run_until(30.0)
    c.bus.kill("z2")  # its last will; it gets no fl/round
    c.clock.run_until(95.0)
    assert server.state.rounds[0] == (1, 1, ("z1",))
    assert learners["z2"].installed == [] and learners["z1"].installed == [1]


def test_the_agent_installs_the_global_model_and_resets_its_target(cfg: Config) -> None:
    # 13.7.5: install between decisions; the target network is reset to the installed weights; every
    # decision logs (global version, local updates since the install), 5.5.
    from pcnme.agent.network import make_network
    from pcnme.agent.online import OnlineAgent

    a = OnlineAgent(cfg, "z1", make_network(cfg, 1))
    other = make_network(cfg, 2).state_dict()
    a.updates, a.new_samples = 7, 4  # as if it had trained a while
    assert a.new_experiences == 4
    a.mark_uploaded()
    a.install(other, 3)
    for name, t in a.online.state_dict().items():
        assert torch.equal(t, other[name]) and torch.equal(a.target.state_dict()[name], other[name])
    assert a.model_version == (3, 0) and a.global_version == 3
    assert a.new_experiences == 0
    a.updates, a.new_samples = 9, 6
    assert a.model_version == (3, 2) and a.new_experiences == 2


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["fedavg", "finetune"])
def test_a_federated_city_runs_rounds_with_real_agents(variant: str) -> None:
    # 13.7 end to end: one 61 s episode (+ 30 s drain), 10 cars; the round at T_fl = 60 s
    # closes at 90 s and the download lands about 0.13 s later: every zone ends on version 1.
    from pcnme.agent.network import make_network
    from pcnme.agent.online import OnlineAgent
    from pcnme.experiments.city import City

    cfg = fixed_delay_config(ablation__federated=variant)
    agents = {z: OnlineAgent(cfg, z, make_network(cfg, 1)) for z in ("z1", "z2", "z3", "z4")}
    c = City(cfg, "S2", lambda cf, s, z: agents[z], 2000, n_cars=10).run(61.0)
    assert c.fed.version == 1 and set(c.fed.rounds[0][2]) <= set(agents)
    for a in agents.values():  # each installed version 1 (and kept learning online afterwards)
        assert a.global_version == 1 and a.model_version[0] == 1
        assert (a.personal is not None) == (variant == "finetune")


def test_the_download_takes_time_and_a_zone_that_dies_meanwhile_installs_nothing() -> None:
    # 13.7.5, Q50: the GET pays backhaul and WAN both ways plus the model at B_WAN (~0.13 s here); z2 dies
    # after fl/global but before its download lands, so it never installs version 1.
    learners = {"z1": Learner(1.0, 5), "z2": Learner(3.0, 5)}
    c, server = fed_city(learners)
    c.clock.run_until(90.1)  # fl/global reached the zones at ~90.07 s
    assert server.state.version == 1 and learners["z1"].installed == []
    c.bus.kill("z2")
    c.clock.run_until(90.19)  # the 320 bits at B_WAN take 62.5 ms of the ~0.13 s
    assert learners["z1"].installed == []
    c.clock.run_until(95.0)
    assert learners["z1"].installed == [1] and learners["z2"].installed == []


def test_a_big_model_takes_its_bits_at_b_wan_to_download() -> None:
    # Q50: a model of 0.5 s at B_WAN lands 0.5 s later than the delays alone (~90.2 s).
    b_wan = fixed_delay_config().derived.b_wan
    big = Learner(1.0, 5)
    big.w = {"w": torch.zeros(int(0.5 * b_wan / 32))}
    learners = {"z1": big, "z2": Learner(3.0, 0)}
    c, _ = fed_city(learners)
    c.clock.run_until(90.6)
    assert learners["z2"].installed == []
    c.clock.run_until(91.5)
    assert learners["z2"].installed == [1]
