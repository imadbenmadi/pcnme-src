"""The testbed's wire format (PLAN M9.1 step 2): the bus and call payloads as JSON.

Dataclasses and named tuples become {"$": type, ...fields}, tuples {"$t": [...]}, mappings {"$d": [[k, v],
...]} (keys keep their type); only the types registered here travel, so a message can never build an
arbitrary object. JSON carries no clock of another machine beyond what the payload already holds (5.4).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any

from pcnme.car.car import Confirm, Result, Task
from pcnme.ids import AttemptKey, SnapshotVersion, TaskId
from pcnme.node.executor import Accepted, Ignored, Refused, Report, TaskRequest
from pcnme.node.state import SlotState
from pcnme.zone.statestore import Snapshot
from pcnme.zone.zone import AckMessage, GpsMessage, ResultMessage, ZoneStatus

_TYPES: dict[str, type] = {
    t.__name__: t
    for t in (
        TaskId,
        SnapshotVersion,
        AttemptKey,
        TaskRequest,
        Accepted,
        Refused,
        Ignored,
        Report,
        ResultMessage,
        ZoneStatus,
        GpsMessage,
        AckMessage,
        Confirm,
        Result,
        Task,
        Snapshot,
        SlotState,
    )
}


def _to(v: Any) -> Any:
    name = type(v).__name__
    if _TYPES.get(name) is type(v):
        if dataclasses.is_dataclass(v):
            return {"$": name, **{f.name: _to(getattr(v, f.name)) for f in dataclasses.fields(v)}}
        return {"$": name, "v": [_to(x) for x in v]}  # type: ignore[attr-defined]  # a named tuple
    if isinstance(v, tuple):
        return {"$t": [_to(x) for x in v]}
    if isinstance(v, list):
        return [_to(x) for x in v]
    if isinstance(v, Mapping):
        return {"$d": [[_to(k), _to(x)] for k, x in v.items()]}
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    raise TypeError(f"{name} cannot travel on the testbed (not a registered payload type)")


def _from(v: Any) -> Any:
    if isinstance(v, list):
        return [_from(x) for x in v]
    if not isinstance(v, dict):
        return v
    if "$t" in v:
        return tuple(_from(x) for x in v["$t"])
    if "$d" in v:
        return {_from(k): _from(x) for k, x in v["$d"]}
    name = v.get("$")
    t = _TYPES.get(name)  # type: ignore[arg-type]
    if t is None:
        raise ValueError(f"unknown payload type {name!r}")
    if dataclasses.is_dataclass(t):
        return t(**{k: _from(x) for k, x in v.items() if k != "$"})
    return t(*(_from(x) for x in v["v"]))


def encode(payload: object) -> bytes:
    return json.dumps(_to(payload), separators=(",", ":")).encode()


def decode(data: bytes) -> object:
    return _from(json.loads(data))
