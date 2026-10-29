"""The agent's network (final-design 11.2, 11.3.7, 18 RL rows; PLAN M6.1). THIS TOUCHES THE AGENT.

48 inputs -> 256 -> 256 -> dueling heads (V: 1, A: 6), Q(s, a) = V(s) + A(s, a) - mean_a A(s, a) (Wang et
al.'s mean form). The agent takes the highest unmasked Q (11.2); the double-DQN target uses only the actions
valid in s', with the same masks as acting (11.3.7): masked_argmax for both. Engineering choices: ReLU,
PyTorch's default initialisation, float32.

Determinism (CLAUDE.md): the weights come from a seed of the run's stream for the purpose; on CPU torch runs
with deterministic algorithms and one thread, so a run repeats bit for bit on one machine. CUDA (the desktop's
GTX 1070) runs the same code; GPU-only tests carry @pytest.mark.gpu.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from pcnme.agent.inputs import N_INPUTS
from pcnme.config import Config
from pcnme.zone.decision import OPTIONS

N_ACTIONS = len(OPTIONS)  # 11.2: here, n1-n4, cloud


class DuelingQNet(nn.Module):
    def __init__(
        self,
        n_in: int = N_INPUTS,
        hidden: Sequence[int] = (256, 256),
        n_actions: int = N_ACTIONS,
        dueling: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_in
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        self.body = nn.Sequential(*layers)
        self.dueling = dueling
        self.advantage = nn.Linear(width, n_actions)
        self.value = nn.Linear(width, 1) if dueling else None

    def forward(self, x: Tensor) -> Tensor:
        z = self.body(x)
        a = self.advantage(z)
        if self.value is None:
            return a
        return self.value(z) + a - a.mean(dim=-1, keepdim=True)


def make_network(cfg: Config, seed: int) -> DuelingQNet:
    """18: 48 -> 256 -> 256 -> dueling heads, its weights drawn from `seed` alone (torch's global generator
    is left as it was)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return DuelingQNet(N_INPUTS, cfg.rl.hidden, N_ACTIONS, cfg.rl.dueling)


def masked_argmax(q: Tensor, valid: Tensor) -> Tensor:
    """The best valid action per row (11.2, 11.3.7). Every row needs at least one valid action: an all-masked
    task makes no decision (7.3)."""
    if not bool(valid.any(dim=-1).all()):
        raise ValueError("a row with every action masked has no decision")
    return q.masked_fill(~valid, float("-inf")).argmax(dim=-1)


def masked_max(q: Tensor, valid: Tensor) -> Tensor:
    """max over the valid actions per row."""
    return q.gather(-1, masked_argmax(q, valid).unsqueeze(-1)).squeeze(-1)


def device() -> torch.device:
    """CUDA when there is one (the desktop), else the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def deterministic() -> None:
    """Make CPU runs repeatable: deterministic algorithms, one thread (the network is small: one thread is
    also the fastest)."""
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(1)
