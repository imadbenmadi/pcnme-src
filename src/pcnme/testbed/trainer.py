"""A learning zone's online trainer in its own process (final-design 13.5, 15.5; QUESTIONS Q62, 11 Nov).

On the testbed a zone that learns gets two CPUs: one for the zone (decisions, queues, MQTT), one for its
trainer. The zone keeps acting with its agent and turns its decisions and transitions into samples as in
fast mode (agent/online.py, the n-step chain; the LSTM baselines' histories; PPO's rollout), but sends them
to the trainer instead of learning on its own loop thread. The trainer holds the same agent, deployed from
the same trained model and seed, adds the samples to its replay and makes the updates
(rl.updates_per_transition; PPO: one update per rollout), and every `testbed.trainer_sync_s` sends its
newest weights back. The zone installs them on its loop, so always between two decisions (Q62). Fast mode
is unchanged: there the agent learns in place.

Messages, zone -> trainer: ("samples", [Sample]), ("rollout", [(Sample, logp)]), ("stop",).
Trainer -> zone: ("weights", state_dict, updates, new_samples, last loss row).
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Any

PROTOCOL = 1  # bump when a message changes


def _net(agent: Any) -> Any:
    """The network that decides: the D3QN family and the LSTM baselines `online`, PPO `net`."""
    return agent.net if hasattr(agent, "rollout") else agent.online


def build_agent(cfg: Any, zone_id: str, method: str, scenario: str, seed: int, models: Path) -> Any:
    """The deployed agent, exactly as the zone builds it (testbed/zone.py zone_placement)."""
    from pcnme.sim.rng import RandomStreams
    from pcnme.testbed.zone import zone_placement

    usage = "evaluation" if seed >= 3000 else "training"
    streams = RandomStreams.from_config(cfg, seed, usage)
    return zone_placement(cfg, streams, zone_id, method, scenario, seed, models)


def trainer_main(cfg: Any, zone_id: str, method: str, scenario: str, seed: int, models: str,
                 inbox: Any, outbox: Any) -> None:  # fmt: skip
    """The trainer process: learn from the zone's samples, send the weights back every trainer_sync_s."""
    import torch

    torch.set_num_threads(1)  # its own CPU (15.5, Q62)
    agent = build_agent(cfg, zone_id, method, scenario, seed, Path(models))
    is_ppo = hasattr(agent, "rollout")
    learn = type(agent)._add if not is_ppo else None  # the in-place learning of fast mode
    sync = cfg.testbed.trainer_sync_s
    sent_at, sent_updates = time.monotonic(), -1
    while True:
        try:
            msg = inbox.get(timeout=sync)
        except queue.Empty:
            msg = None
        if msg is not None:
            kind = msg[0]
            if kind == "stop":
                break
            if kind == "samples" and learn is not None:
                learn(agent, msg[1])
            elif kind == "rollout" and is_ppo:
                agent.rollout = list(msg[1])
                agent.update()
                agent.rollout = []
        now = time.monotonic()
        if agent.updates != sent_updates and now - sent_at >= sync:
            state = {k: v.detach().clone() for k, v in _net(agent).state_dict().items()}
            last = agent.losses[-1] if agent.losses else None
            outbox.put(("weights", state, agent.updates, getattr(agent, "new_samples", 0), last))
            sent_at, sent_updates = now, agent.updates


class RemoteTrainer:
    """The zone's side: forwards the agent's samples to the trainer process and installs its weights."""

    def __init__(self, agent: Any, clock: Any, sync_s: float, process: Any, inbox: Any, outbox: Any) -> None:
        self.agent, self.clock, self.sync_s = agent, clock, sync_s
        self.process, self.inbox, self.outbox = process, inbox, outbox
        self.sent = self.installed = 0  # samples sent, weight sets installed
        self.updates = 0  # the trainer's update count at the last install
        self._connect()

    @classmethod
    def start(cls, cfg: Any, agent: Any, clock: Any, zone_id: str, method: str, scenario: str, seed: int,
              models: Path) -> RemoteTrainer:  # fmt: skip
        ctx = mp.get_context("spawn")  # Windows and Linux alike (CLAUDE.md)
        inbox, outbox = ctx.Queue(), ctx.Queue()
        p = ctx.Process(target=trainer_main, name=f"trainer-{zone_id}", daemon=True,
                        args=(cfg, zone_id, method, scenario, seed, str(models), inbox, outbox))  # fmt: skip
        p.start()
        return cls(agent, clock, cfg.testbed.trainer_sync_s, p, inbox, outbox)

    def _connect(self) -> None:
        """The agent stops learning in place: its samples (or PPO rollouts) go to the trainer."""
        a = self.agent
        if hasattr(a, "rollout"):  # PPO: a full rollout is one update

            def update() -> None:
                self.inbox.put(("rollout", list(a.rollout)))
                self.sent += len(a.rollout)

            a.update = update
        else:

            def add(samples: Any) -> None:
                samples = list(samples)
                if samples:
                    a.new_samples += len(samples)
                    self.inbox.put(("samples", samples))
                    self.sent += len(samples)

            a._add = add
        a.trainer = self  # the zone's agent is connected (tests, the pre-flight check)
        self.clock.schedule_in(self.sync_s, self._poll)

    @property
    def connected(self) -> bool:
        return getattr(self.agent, "trainer", None) is self and self.process.is_alive()

    def _poll(self) -> None:
        """On the zone's loop, so between two decisions: install the newest weights the trainer sent."""
        newest = None
        while True:
            try:
                newest = self.outbox.get_nowait()
            except queue.Empty:
                break
        if newest is not None:
            _, state, updates, _samples, last = newest
            _net(self.agent).load_state_dict(state)
            self.agent.updates = self.updates = updates
            if last is not None:
                self.agent.losses.append(last)
                del self.agent.losses[:-1]  # the zone keeps the newest only (display)
            self.installed += 1
        self.clock.schedule_in(self.sync_s, self._poll)

    def close(self, timeout: float = 5.0) -> None:
        try:
            self.inbox.put(("stop",))
            self.process.join(timeout)
        finally:
            if self.process.is_alive():
                self.process.terminate()
