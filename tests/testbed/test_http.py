"""The HTTP adapter (PLAN M9.1 step 4; final-design 5.3): POST, GET and DELETE /tasks between a caller's
RealClock loop and a target served by FastAPI on its own loop; replies come back on the caller's loop."""

from __future__ import annotations

import threading
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from pcnme.ids import AttemptKey, SnapshotVersion, TaskId
from pcnme.node.executor import Accepted, TaskRequest
from pcnme.testbed.clock import RealClock
from pcnme.testbed.http import HttpCalls, make_app
from pcnme.zone.statestore import Snapshot

KEY = AttemptKey(TaskId("c1", 0, 5), "z1", 1)
SNAP = Snapshot("z2", SnapshotVersion(0, 1), {}, 0.0, None, 0.0)


class FakeExecutor:
    def __init__(self) -> None:
        self.cancelled: list[AttemptKey] = []
        self.thread = ""

    def probe(self, key: AttemptKey) -> tuple[str, float | None]:
        return ("queued", 0.25)

    def cancel(self, key: AttemptKey) -> bool:
        self.cancelled.append(key)
        return True


class FakeTarget:
    def __init__(self) -> None:
        self.executor = FakeExecutor()
        self.got: list[TaskRequest] = []
        self.thread = ""

    def handle(self, req: TaskRequest) -> tuple[Any, Snapshot]:
        self.got.append(req)
        self.thread = threading.current_thread().name
        return Accepted("gpu", 0.01), SNAP


def serve(target: FakeTarget) -> tuple[RealClock, TestClient]:
    clock = RealClock()
    threading.Thread(target=clock.run_for, args=(2.0,), name="target-loop", daemon=True).start()
    return clock, TestClient(make_app(clock, target))


def test_post_get_delete_end_to_end() -> None:
    target = FakeTarget()
    tclock, http = serve(target)
    caller = RealClock()
    calls = HttpCalls(caller, {"z2": ""}, client=http)
    out: list[object] = []
    seen_on: list[str] = []
    req = TaskRequest(KEY, "T1", "hard", 0.1, 0.07)
    at: list[bool] = []
    calls.post(
        "z2",
        None,
        req,
        0.0,
        lambda r: (out.append(r), seen_on.append(threading.current_thread().name)),
        lambda: at.append(True),
    )
    calls.probe("z2", caller.now + 0.01, KEY, 0.0, out.append)
    calls.cancel("z2", caller.now, KEY)
    caller.run_for(0.5)
    tclock.stop()
    assert target.got == [req] and target.thread == "target-loop" and at == [True]
    assert (Accepted("gpu", 0.01), SNAP) in out and ("queued", 0.25) in out
    assert target.executor.cancelled == [KEY]
    assert set(seen_on) == {threading.current_thread().name}  # replies run on the caller's loop


def test_a_dead_target_never_replies() -> None:
    caller = RealClock()
    calls = HttpCalls(caller, {"z9": "http://127.0.0.1:9"}, timeout=0.2)
    out: list[object] = []
    calls.post("z9", None, TaskRequest(KEY, "T1", "hard", 0.1, 0.07), 0.0, out.append)
    caller.run_for(0.6)
    assert out == []  # the caller's own timers notice (9.3)


def test_the_server_refuses_a_wrong_payload() -> None:
    _, http = serve(FakeTarget())
    from pcnme.testbed.codec import encode

    assert http.post("/tasks", content=encode(KEY)).status_code == 422
