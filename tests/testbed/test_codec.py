"""The testbed's message codec (PLAN M9.1 step 2): every payload of the bus and the HTTP calls round-trips
through JSON unchanged, types included."""

from __future__ import annotations

import json
import math

import pytest

from pcnme.car.car import Confirm, Result, Task
from pcnme.ids import AttemptKey, SnapshotVersion, TaskId
from pcnme.node.executor import Accepted, Ignored, Refused, Report, TaskRequest
from pcnme.node.state import SlotState
from pcnme.testbed.codec import decode, encode
from pcnme.zone.statestore import Snapshot
from pcnme.zone.zone import AckMessage, GpsMessage, ResultMessage, ZoneStatus

TID = TaskId("c003", 2, 1187)
KEY = AttemptKey(TID, "z1", 2)
REPORT = Report(KEY, "T1", "done", "gpu", 1.25, 1.27, 0.02, 0.5)
SLOTS = {
    "gpu": SlotState(2, 32, 3, 1, 0.03, 0.2, (0.0, 0.012), 1),
    "cpu": SlotState(2, 16, 0, 0, 0.0, 0.0, (0.0, 0.0), 0),
}
PAYLOADS: list[object] = [
    TID,
    KEY,
    SnapshotVersion(1, 42),
    TaskRequest(KEY, "T1", "hard", 0.1, 0.073),
    Accepted("gpu", 0.004),
    Refused("late"),
    Ignored(),
    REPORT,
    ResultMessage("result", "z2", REPORT),
    ZoneStatus("z1", "online", 300.0, 300.0, 350.0),
    GpsMessage("c003", 1.5, 2.5, 10.0, 0.0, "z1", 12.3),
    AckMessage(Confirm(TID, "z1", 2, 0.081), "z2"),
    Result(TID, "z1", 2),
    Snapshot("z1", SnapshotVersion(0, 7), SLOTS, 0.012, None, 0.25),
    Task(TID, "c003", "T4", "soft", 1.0, 1.0, 1.2e6, 24000.0, 3.5),
    ("c003", "z1"),
    [1, (2, 3), {"a": None}],
]


@pytest.mark.parametrize("payload", PAYLOADS, ids=lambda p: type(p).__name__)
def test_every_payload_round_trips_through_json(payload: object) -> None:
    text = encode(payload)
    assert isinstance(text, bytes) and json.loads(text) is not None
    back = decode(text)
    assert back == payload and type(back) is type(payload)


def test_nested_types_survive() -> None:
    s = decode(encode(PAYLOADS[13]))
    assert (
        isinstance(s, Snapshot)
        and isinstance(s.version, SnapshotVersion)
        and isinstance(s.slots["gpu"], SlotState)
    )
    assert isinstance(s.slots["gpu"].remaining, tuple)
    k = decode(encode(KEY))
    assert isinstance(k, AttemptKey) and isinstance(k.task_id, TaskId)


def test_infinity_survives() -> None:
    assert math.isinf(decode(encode(float("inf"))))  # type: ignore[arg-type]


def test_an_unknown_type_is_refused() -> None:
    class Foo:
        pass

    with pytest.raises(TypeError, match="Foo"):
        encode(Foo())
    with pytest.raises(ValueError, match="Bogus"):
        decode(b'{"$": "Bogus"}')
