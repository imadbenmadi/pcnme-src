"""The HTTP adapter (PLAN M9.1 step 4; final-design 5.3): POST, GET and DELETE /tasks between a caller's
RealClock loop and a target served by FastAPI on its own loop; replies come back on the caller's loop."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from pcnme.ids import AttemptKey, SnapshotVersion, TaskId
from pcnme.node.executor import Accepted, TaskRequest
from pcnme.testbed.clock import RealClock
from pcnme.testbed.http import HttpCalls, make_app
from pcnme.zone.statestore import Snapshot

pytest.importorskip("fastapi")

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


def serve(target: FakeTarget) -> tuple[RealClock, Any]:
    from fastapi.testclient import TestClient

    clock = RealClock()
    threading.Thread(target=clock.run_for, args=(2.0,), name="target-loop", daemon=True).start()
    return clock, TestClient(make_app(clock, target))


def calls_to(target: FakeTarget, caller: RealClock) -> tuple[RealClock, HttpCalls]:
    """HttpCalls on a NetLoop whose clients reach the target's app in-process (httpx.ASGITransport)."""
    import httpx

    from pcnme.testbed.netloop import NetLoop

    tclock = RealClock()
    threading.Thread(target=tclock.run_for, args=(2.0,), name="target-loop", daemon=True).start()
    net = NetLoop(caller, transport=httpx.ASGITransport(app=make_app(tclock, target)))
    return tclock, HttpCalls(caller, {"z2": "http://z2"}, net)


def test_post_get_delete_end_to_end() -> None:
    target = FakeTarget()
    caller = RealClock()
    tclock, calls = calls_to(target, caller)
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
    from pcnme.testbed.netloop import NetLoop

    caller = RealClock()
    calls = HttpCalls(caller, {"z9": "http://127.0.0.1:9"}, NetLoop(caller, timeout=0.2))
    out: list[object] = []
    calls.post("z9", None, TaskRequest(KEY, "T1", "hard", 0.1, 0.07), 0.0, out.append)
    caller.run_for(0.6)
    assert out == []  # the caller's own timers notice (9.3)


def test_the_server_refuses_a_wrong_payload() -> None:
    _, http = serve(FakeTarget())
    from pcnme.testbed.codec import encode

    assert http.post("/tasks", content=encode(KEY)).status_code == 422


class FakeZone(FakeTarget):
    zone_id = "z2"

    def __init__(self) -> None:
        super().__init__()
        self.rx: dict[object, Any] = {}
        self.uploads: list[tuple[object, float, float]] = []

    def upload(self, task: Any, rate: float, b_car: float, on_rx: Any) -> None:
        self.uploads.append((task.task_id, rate, b_car))
        self.rx[task.task_id] = on_rx

    def cut_upload(self, task_id: object) -> bool:
        return self.rx.pop(task_id, None) is not None


def _upload(seq: int) -> Any:
    from pcnme.car.car import Task
    from pcnme.ids import TaskId
    from pcnme.testbed.messages import Upload

    task = Task(TaskId("c1", 0, seq), "c1", "T1", "hard", 0.1, 0.1, 1.2e6, 2.4e4, 0.0)
    return Upload(task, b_car=0.09, rate=5e8)


def test_a_car_upload_is_answered_at_t_rx_and_a_cut_one_is_refused() -> None:
    from pcnme.ids import TaskId
    from pcnme.testbed.codec import decode, encode
    from pcnme.testbed.messages import UploadReply

    zone = FakeZone()
    tclock, http = serve(zone)
    replies: dict[int, Any] = {}

    def send(seq: int) -> None:
        replies[seq] = http.post("/tasks", content=encode(_upload(seq)))

    a, b = threading.Thread(target=send, args=(1,)), threading.Thread(target=send, args=(2,))
    a.start(), b.start()
    for _ in range(200):
        if len(zone.rx) == 2:
            break
        threading.Event().wait(0.01)
    tclock.call_soon(lambda: zone.rx.pop(TaskId("c1", 0, 1))())  # task 1 reaches t_rx
    assert http.request("DELETE", "/uploads", content=encode(TaskId("c1", 0, 2))).status_code == 200
    a.join(2), b.join(2)
    tclock.stop()
    assert zone.uploads == [
        (TaskId("c1", 0, 1), 5e8, 0.09),
        (TaskId("c1", 0, 2), 5e8, 0.09),
    ] or zone.uploads == [
        (TaskId("c1", 0, 2), 5e8, 0.09),
        (TaskId("c1", 0, 1), 5e8, 0.09),
    ]
    assert replies[1].status_code == 200 and decode(replies[1].content) == UploadReply(
        TaskId("c1", 0, 1), "z2"
    )
    assert replies[2].status_code == 409  # cut before its last bit (9.7): nothing left behind


def test_on_sent_runs_on_the_callers_loop_once_the_body_is_written() -> None:
    target = FakeTarget()
    caller = RealClock()
    tclock, calls = calls_to(target, caller)
    sent: list[str] = []
    out: list[object] = []
    calls.post(
        "z2",
        None,
        TaskRequest(KEY, "T1", "hard", 0.1, 0.07),
        0.0,
        out.append,
        on_sent=lambda: sent.append(threading.current_thread().name),
    )
    caller.run_for(0.5)
    tclock.stop()
    assert sent == [threading.current_thread().name] and len(out) == 1


def test_a_body_that_is_not_a_payload_is_refused_not_crashed() -> None:
    _, http = serve(FakeTarget())
    assert http.post("/tasks", content=b"x" * 10_000).status_code == 422


def test_a_post_carries_the_tasks_input_bytes_and_the_target_reads_the_request() -> None:
    """5.3: POST /tasks carries the request + the input, so the shaped links see the real transfer (4.2)."""
    import httpx

    from pcnme.testbed.netloop import NetLoop

    seen: list[int] = []
    target = FakeTarget()
    tclock = RealClock()
    threading.Thread(target=tclock.run_for, args=(2.0,), name="target-loop", daemon=True).start()
    app = make_app(tclock, target)

    async def measuring(scope: Any, receive: Any, send: Any) -> None:
        async def counted() -> Any:
            m = await receive()
            seen.append(len(m.get("body", b"")))
            return m

        await app(scope, counted, send)

    caller = RealClock()
    net = NetLoop(caller, transport=httpx.ASGITransport(app=measuring))
    calls = HttpCalls(caller, {"z2": "http://z2"}, net, input_bytes={"T1": 76_000})
    out: list[object] = []
    calls.post("z2", None, TaskRequest(KEY, "T1", "hard", 0.1, 0.07), 0.0, out.append)
    caller.run_for(0.5)
    tclock.stop()
    assert sum(seen) > 76_000 and target.got and len(out) == 1
