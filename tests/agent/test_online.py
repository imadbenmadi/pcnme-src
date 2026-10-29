"""Online learning in every zone (PLAN M6.5, M6 acceptance; final-design 13.5, 13.6, 11.2)."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pcnme.agent.network import make_network  # noqa: E402
from pcnme.agent.nstep import build  # noqa: E402
from pcnme.agent.online import OnlineAgent, OnlineChain, train_online  # noqa: E402
from pcnme.config import Config, load_config  # noqa: E402
from pcnme.experiments.city import City  # noqa: E402
from pcnme.zone.layout import NeighbourTable, Site  # noqa: E402
from tests.agent.test_inputs import make_ctx  # noqa: E402
from tests.agent.test_losses import chain  # noqa: E402

ZONES = ("z1", "z2", "z3", "z4")


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


# --- the online n-step chain ----------------------------------------------------------------------------


def test_the_online_chain_gives_the_offline_samples_in_any_arrival_order() -> None:
    c = chain()
    oc = OnlineChain(2, 0.9, 0.1)
    for t in c:
        oc.decision((t.task_id, t.attempt))
    out = []
    for t in (c[2], c[0], c[3], c[1]):  # rewards become final in another order than the decisions
        out += oc.transition(t)
    assert len(out) == 3  # sample 3 waits: its chain of 2 is not there (it is the last)
    out += oc.flush()
    assert out == build(c, 2, 0.9, 0.1)
    assert oc.order == [] and oc.next == 0  # ready for the next episode


def test_a_sample_waits_until_its_n_steps_are_final() -> None:
    c = chain()
    oc = OnlineChain(2, 0.9, 0.1)
    for t in c:
        oc.decision((t.task_id, t.attempt))
    assert oc.transition(c[1]) == []  # sample 0 needs c[0] and c[1]
    assert len(oc.transition(c[0])) == 1


# --- acting (11.2) ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("eps", [0.0, 1.0])
def test_a_masked_action_is_never_chosen(cfg: Config, eps: float) -> None:
    # PLAN M6 acceptance: random masks, the greedy (eps 0) and the random (eps 1) branch.
    agent = OnlineAgent(cfg, "z1", make_network(cfg, 1), epsilon=eps, learn=False)
    base = make_ctx(cfg, "T1")
    rng = random.Random(0)
    for _ in range(300):
        mask = [rng.random() < 0.5 for _ in range(6)]
        if not any(mask):
            mask[rng.randrange(6)] = True
        views = tuple(replace(v, masked=None if m else "full") for v, m in zip(base.views, mask, strict=True))
        inputs = tuple(rng.uniform(-1, 2) for _ in range(48))
        ctx = replace(base, views=views, inputs=inputs)
        a = agent.choose(ctx)
        assert a is not None and mask[["here", "n1", "n2", "n3", "n4", "cloud"].index(a)]


def test_every_option_masked_makes_no_decision(cfg: Config) -> None:
    agent = OnlineAgent(cfg, "z1", make_network(cfg, 1), learn=False)
    base = make_ctx(cfg, "T1")
    closed = replace(base, views=tuple(replace(v, masked="full") for v in base.views), inputs=(0.0,) * 48)
    assert agent.choose(closed) is None


def test_greedy_acting_takes_the_masked_argmax_and_exploring_is_uniform(cfg: Config) -> None:
    net = make_network(cfg, 2)
    base = make_ctx(cfg, "T1")
    every = tuple(replace(v, masked=None) for v in base.views)  # all 6 open: a random pick is visible
    ctx = replace(base, views=every, inputs=tuple(0.1 * k for k in range(48)))
    valid = torch.tensor([[v.masked is None for v in ctx.views]])
    q = net(torch.tensor([ctx.inputs], dtype=torch.float32))
    best = ["here", "n1", "n2", "n3", "n4", "cloud"][int(q.masked_fill(~valid, -1e9).argmax())]
    greedy = OnlineAgent(cfg, "z1", net, epsilon=0.0, learn=False)
    assert all(greedy.choose(ctx) == best for _ in range(50))
    assert OnlineAgent(cfg, "z1", net, learn=False).eps == cfg.rl.epsilon_online == 0.01
    explorer = OnlineAgent(cfg, "z1", net, epsilon=1.0, learn=False)
    counts = Counter(explorer.choose(ctx) for _ in range(600))
    open_ = [v.option for v in ctx.views if v.masked is None]
    assert set(counts) == set(open_) and min(counts.values()) > 600 / len(open_) * 0.7


# --- learning (13.5) ---------------------------------------------------------------------------------------


def test_one_update_per_four_transitions_and_the_target_copy(cfg: Config) -> None:
    # rl.updates_per_transition 0.25; a hard target copy every target_update_every (3 here) updates.
    fast = replace(cfg, rl=replace(cfg.rl, target_update_every=3, batch=4))
    agent = OnlineAgent(fast, "z1", make_network(fast, 3))

    def same() -> bool:
        a, b = agent.target.state_dict(), agent.online.state_dict()
        return all(torch.equal(a[k], b[k]) for k in a)

    samples = build(chain(), 2, 0.9, 0.1) * 5  # 20 samples
    agent._add(samples[:12])
    assert agent.updates == 3 and same()  # copied at update 3, nothing since
    agent._add(samples[12:])
    assert agent.updates == 5 and not same()  # updates 4 and 5 moved the online network only
    assert [x[0] for x in agent.losses] == [1, 2, 3, 4, 5]


def test_the_expert_part_stays_through_online_learning(cfg: Config) -> None:
    expert = [replace(s, expert=True) for s in build(chain(), 2, 0.9, 0.1)]
    agent = OnlineAgent(cfg, "z1", make_network(cfg, 4), expert)
    agent._add(build(chain(), 2, 0.9, 0.1) * 30)
    assert len(agent.replay.expert) == 4 and all(s.expert for s in agent.replay.expert)


# --- the neighbour order (13.6) and whole episodes ---------------------------------------------------------


def test_the_neighbour_positions_are_permuted() -> None:
    sites = [
        Site(z, x, y, 350) for z, (x, y) in {"z2": (900, 300), "z3": (300, 900), "z4": (900, 900)}.items()
    ]
    t = NeighbourTable.at_start(Site("z1", 300, 300, 350), sites, 900)
    assert t.positions == ["z2", "z3", "z4", None]
    t.permute([3, 2, 0, 1])
    assert t.positions == [None, "z4", "z2", "z3"]
    with pytest.raises(ValueError):
        t.permute([0, 0, 1, 2])


def test_training_shuffles_the_neighbour_order_per_episode(cfg: Config) -> None:
    def order(seed: int, shuffle: bool) -> dict[str, list[str | None]]:
        c = City(cfg, "S2", "greedy", seed, n_cars=10, shuffle_neighbours=shuffle)
        return {z: list(zone.neighbours.positions) for z, zone in c.zones.items()}

    plain = order(2000, False)
    assert plain["z1"] == ["z2", "z3", "z4", None]
    a, b = order(2000, True), order(2001, True)
    assert a == order(2000, True) and a != b and a != plain
    assert all(sorted(map(str, a[z])) == sorted(map(str, plain[z])) for z in a)  # the same zones, moved


def test_online_episodes_repeat_bit_for_bit(cfg: Config) -> None:
    def run() -> tuple[object, list[torch.Tensor]]:
        net = make_network(cfg, 5)
        agents = {z: OnlineAgent(cfg, z, net) for z in ZONES}
        eps = train_online(cfg, agents, "S2", [2000], 2.0, n_cars=40)
        assert all(a.chain.order == [] for a in agents.values())  # flushed at the episode end
        return eps, [p.detach().clone() for p in agents["z1"].online.parameters()]

    (e1, w1), (e2, w2) = run(), run()
    assert e1 == e2 and all(torch.equal(a, b) for a, b in zip(w1, w2, strict=True))
    assert e1[0].tasks > 100 and e1[0].updates > 0 and 0 < e1[0].on_time <= 1  # type: ignore[index]


def test_episode_stats_by_hand(cfg: Config) -> None:
    from pcnme.agent.online import episode_stats
    from tests.expert.test_window import rec

    e_ref, e_bar = cfg.derived.e_ref, cfg.derived.e_bar
    rs = [
        rec(cfg, "T1", "on_time", 0.05, 1.5 * e_ref["T1"]),
        rec(cfg, "T4", "late", 1.5, 2.0 * e_ref["T4"]),
        rec(cfg, "T1", "drop_lost", None, 0.0),
    ]
    on_time, mean_e = episode_stats(cfg, rs)
    assert on_time == pytest.approx(1 / 3)  # late is delivered, not on time (16.3)
    assert mean_e == pytest.approx((1.5 + 2.0 + e_bar) / 3)
    assert episode_stats(cfg, []) == (0.0, 0.0)
