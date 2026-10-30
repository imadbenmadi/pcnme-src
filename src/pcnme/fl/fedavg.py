"""FedAvg rounds (final-design 13.7, 5.2, 5.3, 4.2; PLAN M8).

1. The cloud starts a round every T_fl (fast mode: simulated time) and publishes fl/round = {k, the global
   version, the upload deadline}.
2. Each zone with new experiences uploads its shared model's weights and the number of new experiences
   (POST /fl/rounds/k) through its own backhaul and the shared WAN pipe, like any cloud transfer; a zone with
   none sends nothing.
3. At the deadline the cloud averages the weights it received, by share of new experiences, and announces
   version k on fl/global; zones download it (GET /fl/models/k) and install it between two decisions.
4. A late or offline zone is left out of that round; the round never waits. fl/global is retained, so a
   returning zone gets the newest model (QUESTIONS Q50).

The global state (the version and its weights) lives in FedState, which outlives one fast-mode episode
(a City), so rounds continue across the episodes of a training run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pcnme.bus.bus import Client
from pcnme.config import Config
from pcnme.net import hops
from pcnme.net.series import CloudPath, SeriesTransfer
from pcnme.sim.clock import EventClock
from pcnme.sim.rng import RandomStreams

if TYPE_CHECKING:  # torch only for the types: a rule-based run needs no torch (the CLI test)
    import torch

ROUND, GLOBAL = "fl/round", "fl/global"  # 5.2
Weights = Mapping[str, "torch.Tensor"]


@dataclass(frozen=True, slots=True)
class RoundMsg:
    """fl/round (5.2): round k, the global version now, the upload deadline (on the cloud's clock: for
    logs; a zone decides only with durations, 5.4)."""

    k: int
    version: int
    deadline_in: float  # seconds from the announcement


@dataclass(frozen=True, slots=True)
class GlobalMsg:
    """fl/global (5.2): version k is ready."""

    version: int


@dataclass
class FedState:
    """The cloud's FedAvg state across episodes: the round counter, the global version and its weights."""

    k: int = 0
    version: int = 0
    weights: dict[str, torch.Tensor] | None = None
    rounds: list[tuple[int, int, tuple[str, ...]]] = field(default_factory=list)  # (k, version, zones in it)


def model_bits(weights: Weights) -> float:
    """The upload's size: every parameter as a 32-bit float."""
    return float(sum(t.numel() for t in weights.values()) * 32)


def average(uploads: Mapping[str, tuple[Weights, int]]) -> dict[str, torch.Tensor]:
    """13.7.3: the weights averaged by share of new experiences."""
    total = sum(n for _, n in uploads.values())
    out: dict[str, torch.Tensor] = {}
    for w, n in uploads.values():
        for name, t in w.items():
            part = t.detach().float() * (n / total)
            out[name] = part if name not in out else out[name] + part
    return out


class FedServer:
    """The cloud's FedAvg service in one fast-mode run."""

    def __init__(self, clock: EventClock, cfg: Config, mqtt: Client, state: FedState) -> None:
        self._clock = clock
        self._cfg = cfg.fedavg
        self._mqtt = mqtt
        self.state = state
        self._open: int | None = None  # the round taking uploads
        self._uploads: dict[str, tuple[Weights, int]] = {}
        clock.schedule(clock.now + self._cfg.t_fl_fast_mode, self._start)

    def _start(self) -> None:
        s = self.state
        s.k += 1
        self._open, self._uploads = s.k, {}
        msg = RoundMsg(s.k, s.version, self._cfg.upload_deadline)
        self._mqtt.publish(ROUND, msg, qos=1, key=("fl", "round", s.k))
        k = s.k
        self._clock.schedule(self._clock.now + self._cfg.upload_deadline, lambda: self._close(k))
        self._clock.schedule(self._clock.now + self._cfg.t_fl_fast_mode, self._start)

    def submit(self, zone: str, k: int, weights: Weights, n_new: int) -> bool:
        """POST /fl/rounds/k arrives: kept only for the open round, before its deadline (13.7.4)."""
        if k != self._open or n_new <= 0:
            return False
        self._uploads[zone] = (weights, n_new)
        return True

    def _close(self, k: int) -> None:
        if self._open != k:
            return
        self._open = None
        s = self.state
        if not self._uploads:
            s.rounds.append((k, s.version, ()))
            return  # nothing new: no new version
        s.weights = average(self._uploads)
        s.version = k
        s.rounds.append((k, k, tuple(sorted(self._uploads))))
        self._mqtt.publish(GLOBAL, GlobalMsg(k), qos=1, key=("fl", "global", k), retained=True)


class FedClient:
    """One zone's side (13.7.2-5): upload on fl/round when it has new experiences; download and install on
    fl/global. `learner` is the zone's agent (OnlineAgent): it gives its shared weights and new-experience
    count, and installs a model between two decisions."""

    def __init__(
        self,
        zone_id: str,
        clock: EventClock,
        streams: RandomStreams,
        cfg: Config,
        mqtt: Client,
        path: CloudPath,
        server: FedServer,
        learner: Any,
        alive: Callable[[], bool],
    ) -> None:
        self.zone_id = zone_id
        self._clock, self._streams, self._cfg = clock, streams, cfg
        self._path, self._server, self._learner, self._alive = path, server, learner, alive
        mqtt.subscribe(ROUND)
        mqtt.subscribe(GLOBAL)
        self.uploads: list[tuple[int, bool]] = []  # (round, kept)

    def on_round(self, msg: RoundMsg) -> None:
        n = self._learner.new_experiences
        if n <= 0:
            return  # 13.7.2: nothing new, nothing sent
        weights = {k: v.detach().clone() for k, v in self._learner.shared_state().items()}
        self._learner.mark_uploaded()
        lk, key = self._cfg.links, ("fl", self.zone_id, msg.k)
        d_bh = hops.delay(self._streams, hops.BH_OUT, lk.backhaul_delay, key, 0)
        d_wan = hops.delay(self._streams, hops.WAN_IN, lk.wan_delay, key, 0)

        def arrived(tr: SeriesTransfer) -> None:
            self.uploads.append((msg.k, self._server.submit(self.zone_id, msg.k, weights, n)))

        self._path.send(model_bits(weights), d_bh, d_wan, arrived)  # through the shared WAN pipe (4.2)

    def on_global(self, msg: GlobalMsg) -> None:
        """GET /fl/models/k: the request and the weights back (Q50: WAN + backhaul delays and the push at
        B_WAN, no queue: the WAN pipe models the uplink), then the install, between two decisions."""
        state = self._server.state
        if state.weights is None or msg.version <= self._learner.global_version:
            return
        weights, version = state.weights, state.version
        lk, key = self._cfg.links, ("fl", self.zone_id, "get", version)
        there = hops.delay(self._streams, hops.CALL_BH, lk.backhaul_delay, key, 0)
        there += hops.delay(self._streams, hops.CALL_WAN, lk.wan_delay, key, 0)
        back = hops.delay(self._streams, hops.REPLY_WAN, lk.wan_delay, key, 0)
        back += hops.delay(self._streams, hops.REPLY_BH, lk.backhaul_delay, key, 0)
        back += model_bits(weights) / self._cfg.derived.b_wan

        def install() -> None:
            if self._alive() and version > self._learner.global_version:
                self._learner.install(weights, version)

        self._clock.schedule(self._clock.now + there + back, install)
