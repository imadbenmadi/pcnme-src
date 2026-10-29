"""M6 acceptance tests (PLAN M6; final-design 11.2, 11.3, 13.5, 13.6).

- Hand-computed n-step target; margin loss zero on non-expert transitions; masked actions never chosen: the
  tests listed in TESTS (checked to exist).
- Learning sanity on a two-action toy problem: below.
- Pretraining on demonstrations raises the held-out expert agreement: below (slow: it makes its own small
  demonstrations), and the run on the small M5 demonstrations reported in the M6 acceptance PR.
"""

# ruff: noqa: E501  (the references are full test names, which cannot be wrapped)

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pcnme.agent.losses import Batch, dqfd_loss  # noqa: E402
from pcnme.agent.network import make_network, masked_argmax  # noqa: E402
from pcnme.agent.nstep import Sample  # noqa: E402
from pcnme.agent.replay import Replay  # noqa: E402
from pcnme.config import Config, load_config  # noqa: E402

TESTS = {
    "hand-computed n-step target": "tests.agent.test_losses::test_a_hand_computed_n_step_target",
    "n-step samples by hand": "tests.agent.test_losses::test_n_step_samples_by_hand",
    "margin loss zero on non-expert transitions": "tests.agent.test_losses::test_the_margin_loss_is_zero_on_non_expert_transitions",
    "masked actions never chosen (network)": "tests.agent.test_network::test_a_masked_action_is_never_chosen",
    "masked actions never chosen (agent)": "tests.agent.test_online::test_a_masked_action_is_never_chosen",
    "no s' out of the bootstrapped targets (Q36)": "tests.agent.test_losses::test_rows_without_s_prime_add_nothing_to_the_bootstrapped_terms",
    "expert part never overwritten": "tests.agent.test_replay::test_the_expert_part_is_never_overwritten",
    "neighbour order shuffled per episode": "tests.agent.test_online::test_training_shuffles_the_neighbour_order_per_episode",
    "online episodes repeat bit for bit": "tests.agent.test_online::test_online_episodes_repeat_bit_for_bit",
}


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    c = load_config(default_config_path)
    return replace(c, rl=replace(c.rl, lr=1e-3, batch=32, target_update_every=100))


@pytest.mark.parametrize("ref", TESTS.values())
def test_the_acceptance_test_exists(ref: str) -> None:
    module, name = ref.split("::")
    assert callable(getattr(importlib.import_module(module), name)), ref


def test_learning_sanity_on_a_two_action_toy_problem(cfg: Config) -> None:
    # Two states A and B (input 0 = 1 or 0), two valid actions; in A action 0 pays 1 and action 1 pays 0, in B
    # the reverse. The data come from random actions (no expert, one step, discount 0), so only the TD loss
    # teaches. After training the masked argmax is the paying action in both states, and Q is about the pay.
    rng = np.random.default_rng(0)
    valid = (True, True, False, False, False, False)
    data = []
    for _ in range(400):
        a_state = bool(rng.random() < 0.5)
        s = (1.0 if a_state else 0.0,) + (0.5,) * 47
        a = int(rng.integers(2))
        r = 1.0 if (a == 0) == a_state else 0.0
        # one step with a zero discount: y = r (a sample with no s' would have no TD target at all, Q36)
        data.append(Sample(s, a, r, valid, False, True, s, valid, 0.0, False, 0, 0.0, None, None, 0.0))
    net, target = make_network(cfg, 1), make_network(cfg, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    replay = Replay(cfg, np.random.default_rng(1))
    for s in data:
        replay.add(s)
    for _ in range(600):
        drawn = replay.draw(32, 1.0)
        parts = dqfd_loss(net, target, Batch.of(drawn.samples, drawn.weights), cfg)
        opt.zero_grad()
        parts.total.backward()
        opt.step()
        replay.update(drawn.index, parts.td_errors.tolist())
    states = torch.tensor([(1.0,) + (0.5,) * 47, (0.0,) + (0.5,) * 47])
    q = net(states).detach()
    assert masked_argmax(q, torch.tensor([valid, valid])).tolist() == [0, 1]
    assert q[0, 0] == pytest.approx(1.0, abs=0.15) and q[0, 1] == pytest.approx(0.0, abs=0.15)


@pytest.mark.slow
def test_pretraining_on_demonstrations_raises_the_held_out_expert_agreement() -> None:
    # Small demonstrations made here: 4 short S2 windows; each "front" is the greedy and the energy-greedy
    # plans (C9), replayed with their own w. Agreement on held-out windows rises from the untrained network's.
    from pcnme.agent.dqfd import pretrain
    from pcnme.expert.demos import demonstrate_front
    from pcnme.expert.window import WindowSpec, discover, energy_greedy, plan_of, window_seed
    from tests.zone.harness import fixed_delay_config

    c = fixed_delay_config(nsga2__T_warmup=1.0, nsga2__T_win=0.3)
    c = replace(c, dqfd=replace(c.dqfd, eval_every=100, patience=5, holdout_share=0.25))
    demos = []
    for k in range(4):
        w = discover(c, WindowSpec("S2", window_seed(c, "S2", k), k, n_cars=40))
        energy, eo = plan_of(c, w, energy_greedy)
        points = [(w.greedy.f1, w.greedy.f2), (eo.f1, eo.f2)]
        for d in demonstrate_front(c, w, [w.greedy_genome, energy], points):
            demos += [(("S2", k), t.sample) for t in d.transitions]
    out = pretrain(c, demos, np.random.default_rng(3), seed=4, updates=1500)
    (_, untrained), *_ = out.curve.agreement
    assert out.best_agreement > untrained + 0.2 and out.best_agreement > 0.6
