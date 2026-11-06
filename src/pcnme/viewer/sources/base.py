"""The viewer's data-source interface (viewer/README.md).

A source is one thing the viewer can show. Each has a `kind`, which decides the API it answers:

    replay          a finished run, read from its logs: `runs()` and `replay(run_id)`      (sources/replay.py)
    live-testbed    the Docker testbed while it runs (MQTT -> WebSocket)                   (later, PLAN V3)
    live-training   an RL agent while it learns: episodes, loss, on-time, decisions        (later)

The page asks /api/sources for the list and picks the view by kind. A source that is not ready yet
(`status="development"`) is listed so the page can show its state, and answers nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Kind = Literal["replay", "live-testbed", "live-training"]
Status = Literal["ready", "development"]


@dataclass(frozen=True)
class SourceInfo:
    id: str  # unique, used in URLs (for example "vfc")
    kind: Kind
    label: str  # what the switcher shows
    status: Status = "ready"
    map: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)  # free details for the page (development steps, ...)


@runtime_checkable
class Source(Protocol):
    @property
    def info(self) -> SourceInfo: ...


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    scenario: str
    method: str
    seed: int
    mode: str  # fast or testbed


@runtime_checkable
class ReplaySource(Source, Protocol):
    def runs(self) -> list[RunInfo]: ...

    def replay(self, run_id: str) -> dict[str, Any]: ...

    def status(self) -> dict[str, Any]:
        """The design's scenarios, methods and seeds, and where each scenario x method stands."""
        ...


@dataclass(frozen=True)
class Placeholder:
    """A source listed in the switcher before it exists (the page shows its development state)."""

    info: SourceInfo


class Registry:
    def __init__(self, sources: list[Source]) -> None:
        self._by_id = {s.info.id: s for s in sources}

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "id": s.info.id,
                "kind": s.info.kind,
                "label": s.info.label,
                "status": s.info.status,
                "map": s.info.map,
                "notes": s.info.notes,
            }
            for s in self._by_id.values()
        ]

    def get(self, source_id: str) -> Source:
        return self._by_id[source_id]

    def replay_source(self, source_id: str) -> ReplaySource:
        s = self.get(source_id)
        if s.info.status != "ready" or not isinstance(s, ReplaySource):
            raise KeyError(f"{source_id} is not a ready replay source")
        return s
