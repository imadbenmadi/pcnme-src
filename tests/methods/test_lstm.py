"""Tang & Wong and QECO (PLAN M7; final-design 16.1, 18: H = 10; QUESTIONS Q44)."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pcnme.agent.inputs import Decision  # noqa: E402
from pcnme.config import Config, load_config  # noqa: E402
from pcnme.methods.lstm import LstmAgent, LstmQNet, deployed_lstm, make_lstm, plain_dqn, train_lstm  # noqa: E402
from tests.agent.test_inputs import make_ctx  # noqa: E402
from tests.agent.test_losses import chain  # noqa: E402

ZONES = ("z1", "z2", "z3", "z4")
OPTS = ["here", "n1", "n2", "n3", "n4", "cloud"]


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


@pytest.mark.parametrize("variant", ["tang-wong", "qeco"])
def test_an_lstm_over_the_last_h_inputs_then_dueling_heads(cfg: Config, variant: str) -> None:
    net = make_lstm(cfg, variant, 1)
    assert net.history == cfg.rl.lstm_history_h == 10
    q = net(torch.zeros(3, 10 * 48))
    assert q.shape == (3, 6)
    assert hasattr(net, "mix") == (variant == "qeco")  # QECO joins the current inputs (Q44)
    with pytest.raises(ValueError):
        LstmQNet("dqn", 10)


def test_the_plain_dqn_setup(cfg: Config) -> None:
    p = plain_dqn(cfg)
    assert (p.dqfd.lambda_n, p.dqfd.lambda_margin, p.dqfd.per.alpha, p.dqfd.per.beta_final) == (0, 0, 0, 0)
    assert p.dqfd.lambda_l2 == cfg.dqfd.lambda_l2 and p.rl == cfg.rl


def decision(k: int, inputs: tuple[float, ...]) -> Decision:
    return Decision(float(k), "z1", f"T{k}", 1, "first", inputs, (None,) * 6, 0)


def test_the_state_is_the_zero_padded_history_of_the_last_h_decisions(cfg: Config) -> None:
    a = LstmAgent(cfg, "z1", make_lstm(cfg, "tang-wong", 1), learn=False)
    x = [tuple(float(k + 1) for _ in range(48)) for k in range(12)]
    first = a._window(x[0])
    assert first[: 9 * 48] == (0.0,) * (9 * 48) and first[9 * 48 :] == x[0]
    for k in range(11):
        a._decision(decision(k, x[k]))
    w = a._window(x[11])  # the last 9 decisions, then the current inputs
    assert [w[i * 48] for i in range(10)] == [float(k) for k in range(3, 13)]
    ctx = replace(make_ctx(cfg, "T1"), inputs=x[11])
    assert a.state(ctx) == w


def test_transitions_become_samples_over_histories(cfg: Config) -> None:
    a = LstmAgent(cfg, "z1", make_lstm(cfg, "qeco", 1), learn=False)
    c = chain()
    for t in c:
        a._decision(Decision(t.time, "z1", t.task_id, 1, "first", t.s, (None,) * 6, 0))
    a._transition(c[0])
    a._transition(c[3])  # the last one: no s'
    s0, s3 = a.replay.agent
    assert len(s0.s) == 480 and s0.has1 and s0.s1 is not None and s0.s1[-48:] == c[1].s
    assert s0.g1 == pytest.approx(0.9) and not s0.hasn
    assert not s3.has1 and s3.s1 is None


@pytest.mark.parametrize("eps", [0.0, 1.0])
def test_a_masked_action_is_never_chosen(cfg: Config, eps: float) -> None:
    agent = LstmAgent(cfg, "z1", make_lstm(cfg, "tang-wong", 1), epsilon=eps, learn=False)
    base = make_ctx(cfg, "T1")
    rng = random.Random(0)
    for _ in range(200):
        mask = [rng.random() < 0.5 for _ in range(6)]
        if not any(mask):
            mask[rng.randrange(6)] = True
        views = tuple(replace(v, masked=None if m else "full") for v, m in zip(base.views, mask, strict=True))
        ctx = replace(base, views=views, inputs=tuple(rng.uniform(-1, 2) for _ in range(48)))
        a = agent.choose(ctx)
        assert a is not None and mask[OPTS.index(a)]


@pytest.mark.parametrize("variant", ["tang-wong", "qeco"])
def test_training_stops_at_the_budget_and_deploys(cfg: Config, variant: str) -> None:
    t = train_lstm(cfg, variant, "S2", ZONES, 300, seconds=2.0, n_cars=40)
    assert t.budget.used == 300 and t.schedule == cfg.rl.epsilon_baselines and t.expert == ()
    deployed = deployed_lstm(cfg, t, ZONES, 3000)
    for z in ZONES:
        sd = deployed[z].online.state_dict()
        assert all(torch.equal(sd[k], t.states[z][k]) for k in sd)
    assert sum(len(a.replay.agent) for a in deployed.values()) == 0


def test_tang_wong_reads_the_lstms_last_step_and_qeco_the_current_inputs(cfg: Config) -> None:
    x = torch.zeros(1, 480)
    last = x.clone()
    last[0, -48:] = 1.0  # only the current step differs
    tw = make_lstm(cfg, "tang-wong", 2)
    assert not torch.allclose(tw(x), tw(last))
    qe = make_lstm(cfg, "qeco", 2)
    with torch.no_grad():
        for p in qe.lstm.parameters():
            p.zero_()  # the LSTM says nothing: only the joined current inputs can move Q
    first = x.clone()
    first[0, :48] = 1.0
    assert not torch.allclose(qe(x), qe(last)) and torch.allclose(qe(x), qe(first))


def test_each_decision_keeps_its_own_history_and_an_episode_starts_empty(cfg: Config) -> None:
    from types import SimpleNamespace

    a = LstmAgent(cfg, "z1", make_lstm(cfg, "tang-wong", 1), learn=False)
    x = [tuple(float(k + 1) for _ in range(48)) for k in range(3)]
    for k in range(3):
        a._decision(decision(k, x[k]))
    h = a.histories[("T2", 1)]
    assert [h[i * 48] for i in range(10)] == [0.0] * 7 + [1.0, 2.0, 3.0]  # x0, x1, x2 once each
    a.attach(SimpleNamespace(transitions=SimpleNamespace(on_decision_hook=None, on_replay_hook=None)))
    assert a._window(x[0])[: 9 * 48] == (0.0,) * (9 * 48) and a.histories == {}
