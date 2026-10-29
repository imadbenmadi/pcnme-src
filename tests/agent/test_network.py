"""The network and the masked argmax (PLAN M6.1; final-design 11.2, 11.3.7, 18)."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

torch = pytest.importorskip("torch")

from pcnme.agent.network import (  # noqa: E402
    N_ACTIONS,
    DuelingQNet,
    deterministic,
    device,
    make_network,
    masked_argmax,
    masked_max,
)
from pcnme.config import Config, load_config  # noqa: E402


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


def test_48_inputs_two_hidden_layers_of_256_and_dueling_heads(cfg: Config) -> None:
    net = make_network(cfg, 1)
    q = net(torch.zeros(5, 48))
    assert q.shape == (5, N_ACTIONS) == (5, 6)
    n = sum(p.numel() for p in net.parameters())
    assert n == (48 * 256 + 256) + (256 * 256 + 256) + (256 * 6 + 6) + (256 * 1 + 1)


def test_q_is_v_plus_the_centred_advantage(cfg: Config) -> None:
    net = make_network(cfg, 2)
    x = torch.randn(4, 48, generator=torch.Generator().manual_seed(0))
    z = net.body(x)
    a, v = net.advantage(z), net.value(z)
    assert torch.allclose(net(x), v + a - a.mean(dim=1, keepdim=True))
    assert torch.allclose(net(x).mean(dim=1, keepdim=True), v, atol=1e-6)  # mean_a Q = V


def test_without_dueling_q_is_the_advantage_head() -> None:
    net = DuelingQNet(48, (8,), 6, dueling=False)
    x = torch.ones(2, 48)
    assert net.value is None and torch.equal(net(x), net.advantage(net.body(x)))


def test_the_weights_come_from_the_seed_alone(cfg: Config) -> None:
    state = torch.random.get_rng_state()
    a, b, c = make_network(cfg, 7), make_network(cfg, 7), make_network(cfg, 8)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters(), strict=True))
    assert not torch.equal(next(a.parameters()), next(c.parameters()))
    assert torch.equal(torch.random.get_rng_state(), state)  # torch's global generator untouched


def test_masked_argmax_takes_the_best_valid_action() -> None:
    q = torch.tensor([[5.0, 1.0, 9.0, 0.0, 0.0, 2.0], [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]])
    valid = torch.tensor([[True, True, False, False, False, True], [False, False, False, False, False, True]])
    assert masked_argmax(q, valid).tolist() == [0, 5]  # 9.0 is masked
    assert masked_max(q, valid).tolist() == [5.0, -6.0]
    with pytest.raises(ValueError, match="masked"):
        masked_argmax(q, torch.zeros(2, 6, dtype=torch.bool))
    with pytest.raises(ValueError, match="masked"):  # one row all masked is enough
        masked_argmax(q, torch.tensor([[True] * 6, [False] * 6]))


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.floats(-1e6, 1e6, allow_nan=False), min_size=6, max_size=6),
    st.lists(st.booleans(), min_size=6, max_size=6).filter(any),
)
def test_a_masked_action_is_never_chosen(qs: list[float], mask: list[bool]) -> None:
    q, valid = torch.tensor([qs]), torch.tensor([mask])
    a = int(masked_argmax(q, valid)[0])
    f32 = q[0].tolist()  # compare in the network's float32
    assert mask[a] and f32[a] == max(x for x, m in zip(f32, mask, strict=True) if m)


def test_cpu_runs_are_deterministic(cfg: Config) -> None:
    deterministic()
    assert torch.are_deterministic_algorithms_enabled() and torch.get_num_threads() == 1
    x = torch.randn(64, 48, generator=torch.Generator().manual_seed(3))
    assert torch.equal(make_network(cfg, 4)(x), make_network(cfg, 4)(x))
    assert device().type in ("cpu", "cuda")


@pytest.mark.gpu
def test_the_network_runs_on_cuda(cfg: Config) -> None:
    net = make_network(cfg, 5).to("cuda")
    q = net(torch.zeros(3, 48, device="cuda"))
    assert q.device.type == "cuda" and masked_argmax(
        q, torch.ones(3, 6, dtype=torch.bool, device="cuda")
    ).shape == (3,)


def test_the_hidden_layers_are_relu(cfg: Config) -> None:
    net = make_network(cfg, 6)
    assert [type(m).__name__ for m in net.body] == ["Linear", "ReLU", "Linear", "ReLU"]
