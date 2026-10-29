"""Greedy and Random, over the options the masks leave (final-design 16.1, 7.2, 7.3, 12.4).

Greedy: the earliest estimated finish among unmasked options; ties by the 12.4 energy estimate, then by option
order (here first: an engineering choice). Random: uniform among unmasked options, from its own sequential
stream "random_method" (QUESTIONS Q4). Both return None when every option is masked (7.3). Both read the
views the anchor built (ctx.views); called without them (tests), they build them.
"""

from __future__ import annotations

from pcnme.config import Config
from pcnme.energy import EnergyModel
from pcnme.sim.rng import RandomStreams
from pcnme.zone.decision import OPTIONS, DecisionContext, Option, OptionView
from pcnme.zone.masks import views


def greedy_pick(options: list[OptionView] | tuple[OptionView, ...]) -> OptionView | None:
    free = [o for o in options if o.masked is None]
    if not free:
        return None
    return min(free, key=lambda o: (o.finish, o.energy, OPTIONS.index(o.option)))


def _views(cfg: Config, em: EnergyModel, ctx: DecisionContext) -> tuple[OptionView, ...]:
    return ctx.views or tuple(views(cfg, em, ctx))


class Greedy:
    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._em = EnergyModel(config)

    def choose(self, ctx: DecisionContext) -> Option | None:
        pick = greedy_pick(_views(self._cfg, self._em, ctx))
        return None if pick is None else pick.option  # type: ignore[return-value]


class Random:
    def __init__(self, config: Config, streams: RandomStreams, zone_id: str) -> None:
        self._cfg = config
        self._em = EnergyModel(config)
        self._g = streams.generator("random_method", zone_id)

    def choose(self, ctx: DecisionContext) -> Option | None:
        free = [o for o in _views(self._cfg, self._em, ctx) if o.masked is None]
        if not free:
            return None
        return free[int(self._g.integers(len(free)))].option  # type: ignore[return-value]
