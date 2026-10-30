"""FedAvg + local fine-tuning and the model version of every decision (PLAN M8; final-design 13.7, 5.5)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pcnme.agent.network import make_network, masked_argmax  # noqa: E402
from pcnme.agent.nstep import build  # noqa: E402
from pcnme.agent.online import OnlineAgent  # noqa: E402
from pcnme.config import Config, load_config  # noqa: E402
from pcnme.zone.decision import OPTIONS  # noqa: E402
from tests.agent.test_inputs import make_ctx  # noqa: E402
from tests.agent.test_losses import chain  # noqa: E402
from tests.zone.harness import Fixed, city, city_task, fixed_delay_config  # noqa: E402


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


def variant(cfg: Config, federated: str, k: int = 5) -> Config:
    return replace(
        cfg,
        ablation=replace(cfg.ablation, federated=federated),
        fedavg=replace(cfg.fedavg, finetune_k=k),
        rl=replace(cfg.rl, batch=4, target_update_every=3),
    )


def same(a: torch.nn.Module, w: dict[str, torch.Tensor]) -> bool:
    return all(torch.equal(t, w[n]) for n, t in a.state_dict().items())


def trained(cfg: Config) -> OnlineAgent:
    agent = OnlineAgent(cfg, "z1", make_network(cfg, 1), epsilon=0.0)
    agent._add(build(chain(), 2, 0.9, 0.1) * 2)  # 8 samples: 2 updates
    return agent


def test_finetune_installs_the_global_model_and_fine_tunes_a_personal_copy_k_steps(cfg: Config) -> None:
    # 13.7 table: the personal copy is the new global model fine-tuned K steps on the zone's replay buffer;
    # the shared model is the global model itself.
    c = variant(cfg, "finetune")
    agent = trained(c)
    g = make_network(c, 2).state_dict()
    agent.install(g, 1)
    assert same(agent.online, g) and same(agent.target, g)
    assert agent.personal is not None and not same(agent.personal, g)
    assert agent.finetune_steps == 5 and agent.model_version == (1, 0)
    # the copy is exactly K steps from the global model: one step fewer is a different copy
    fewer = trained(variant(cfg, "finetune", k=4))
    fewer.install(g, 1)
    assert fewer.finetune_steps == 4 and not same(fewer.personal, agent.personal.state_dict())  # type: ignore[union-attr]


def test_finetune_decides_with_the_personal_copy_and_uploads_the_shared_model(cfg: Config) -> None:
    c = variant(cfg, "finetune")
    agent = trained(c)
    agent.install(make_network(c, 2).state_dict(), 1)
    assert agent.personal is not None
    base = make_ctx(c, "T1")
    views = tuple(replace(v, masked=None) for v in base.views)
    picks_differ = False
    for k in range(40):
        ctx = replace(base, views=views, inputs=tuple(0.05 * ((k * 7 + j) % 23) - 0.4 for j in range(48)))
        x, valid = torch.tensor([ctx.inputs]), torch.tensor([[True] * 6])
        with torch.no_grad():
            mine = OPTIONS[int(masked_argmax(agent.personal(x), valid)[0])]
            shared = OPTIONS[int(masked_argmax(agent.online(x), valid)[0])]
        assert agent.choose(ctx) == mine
        picks_differ |= mine != shared
    assert picks_differ  # the check above can tell the two copies apart
    assert agent.shared_state() is not None and same(agent.online, agent.shared_state())
    assert not same(agent.personal, agent.shared_state())


def test_every_update_trains_both_copies(cfg: Config) -> None:
    # 13.7: in the third variant every transition trains both copies.
    c = variant(cfg, "finetune")
    agent = trained(c)
    agent.install(make_network(c, 2).state_dict(), 1)
    assert agent.personal is not None
    before_s = {n: t.clone() for n, t in agent.online.state_dict().items()}
    before_p = {n: t.clone() for n, t in agent.personal.state_dict().items()}
    agent._add(build(chain(), 2, 0.9, 0.1))  # 4 samples: 1 update
    assert agent.model_version == (1, 1)
    assert not same(agent.online, before_s) and not same(agent.personal, before_p)


def test_the_personal_copy_keeps_its_own_target_network(cfg: Config) -> None:
    # Its target is reset to the installed model and copied from it every target_update_every steps.
    c = variant(cfg, "finetune", k=3)
    agent = trained(c)
    agent.install(make_network(c, 2).state_dict(), 1)
    assert agent.personal is not None
    assert same(agent.personal_target, agent.personal.state_dict())  # copied at fine-tune step 3
    agent._add(build(chain(), 2, 0.9, 0.1))  # update 3 of the zone: the shared copy's target is copied
    assert same(agent.target, agent.online.state_dict())
    assert not same(agent.personal_target, agent.personal.state_dict())  # its own count: 4 steps


def test_fedavg_decides_with_the_shared_model(cfg: Config) -> None:
    agent = trained(variant(cfg, "fedavg"))
    agent.install(make_network(cfg, 2).state_dict(), 1)
    assert agent.personal is None and agent.finetune_steps == 0


def test_an_empty_replay_gives_an_untuned_personal_copy(cfg: Config) -> None:
    c = variant(cfg, "finetune")
    agent = OnlineAgent(c, "z1", make_network(c, 1))
    g = make_network(c, 2).state_dict()
    agent.install(g, 1)
    assert agent.personal is not None and same(agent.personal, g) and agent.finetune_steps == 0


class Versioned(Fixed):
    model_version = (3, 5)


@pytest.mark.parametrize(("placement", "version"), [(Versioned("here"), (3, 5)), (Fixed("here"), (0, 0))])
def test_every_decision_logs_the_model_version_it_used(placement: Fixed, version: tuple[int, int]) -> None:
    # 13.7.5, 5.5: (global version, local updates since its install); a rule method logs (0, 0).
    seq = {"z1": placement, "z2": Fixed(None)}
    c = city(fixed_delay_config(), lambda z: seq[z], zone_ids=("z1", "z2"))
    c.clock.run_until(0.1)
    c.zones["z1"].upload(city_task(c, "T1"))
    c.clock.run_until(1.0)
    assert [d.model_version for d in c.zones["z1"].decisions] == [version]


class FirstOpen:
    model_version = (3, 5)

    def choose(self, ctx):  # type: ignore[no-untyped-def]
        return next((v.option for v in ctx.views if v.masked is None), None)


def test_the_decision_rows_carry_the_model_version(cfg: Config) -> None:
    # decisions.parquet (METRICS.md): model_version and local_updates.
    from pcnme.experiments.city import City

    c = City(cfg, "S2", lambda cf, s, z: FirstOpen(), 2000, n_cars=10).run(2.0)
    c._decision_rows()  # what write() logs
    rows = c.log.rows["decisions"]
    assert rows and all((r["model_version"], r["local_updates"]) == (3, 5) for r in rows)


def test_each_install_restarts_the_personal_target_count(cfg: Config) -> None:
    # K = 2 < target_update_every = 3: after each install the personal target is still the installed model.
    c = variant(cfg, "finetune", k=2)
    agent = trained(c)
    agent.install(make_network(c, 2).state_dict(), 1)
    g2 = make_network(c, 3).state_dict()
    agent.install(g2, 2)
    assert agent.personal_target is not None and same(agent.personal_target, g2)
