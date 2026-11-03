"""The HTTP adapter (PLAN M9.1 step 4; final-design 5.3, 15.2): the zone's calls interface over real HTTP.

- HttpCalls (the caller): POST /tasks, GET /tasks (probe), DELETE /tasks (cancel), each sent from a worker
  thread at the request's time; the reply is handed back to the caller's loop thread (RealClock.call_soon).
  A target that does not answer (dead, unreachable, timed out) gives no reply: the caller's own timers (upload
  timeout, probe timeout, 9.3) notice, as in fast mode. `back` is unused: the reply's delay is real.
- make_app (the target): a FastAPI app; each request runs on the target's loop thread, so zone code never
  runs on a server thread. Bodies are testbed/codec.py JSON.
"""

# No `from __future__ import annotations`: FastAPI must resolve `request: Request`, imported inside make_app.
import asyncio
import concurrent.futures
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from pcnme.bus.calls import Target
from pcnme.ids import AttemptKey, TaskId
from pcnme.node.executor import Reply, Status, TaskRequest
from pcnme.testbed.clock import RealClock
from pcnme.testbed.codec import decode, encode
from pcnme.testbed.messages import Upload, UploadReply
from pcnme.zone.statestore import Snapshot

WORKERS = 16


class HttpCalls:
    def __init__(
        self, clock: RealClock, urls: Mapping[str, str], *, client: Any = None, timeout: float = 5.0
    ) -> None:
        import httpx

        self._clock = clock
        self._urls = dict(urls)
        self._http = client if client is not None else httpx.Client(timeout=timeout)
        self._pool = concurrent.futures.ThreadPoolExecutor(WORKERS, thread_name_prefix="http")

    real_network = True  # netem gives the delays (QUESTIONS Q53): requests leave when the zone calls

    def _send(
        self,
        method: str,
        target: str,
        path: str,
        body: bytes,
        on_reply: Callable[[Any], None],
        on_sent: Callable[[], None] | None = None,
    ) -> None:
        url = self._urls[target] + path

        def chunks() -> Iterator[bytes]:
            yield body
            if on_sent is not None:  # the body's last chunk has been handed to the socket (C7, Q53)
                self._clock.call_soon(on_sent)

        def work() -> None:
            try:
                r = self._http.request(method, url, content=chunks() if on_sent else body)
            except Exception:  # no answer: lost with the target, the caller's timers notice (9.3)
                return
            if r.status_code == 200:
                reply = decode(r.content)
                self._clock.call_soon(lambda: on_reply(reply))

        self._pool.submit(work)

    def post(
        self,
        target: str,
        arrive_at: float | None,
        req: TaskRequest,
        back: float,
        on_reply: Callable[[tuple[Reply, Snapshot]], None],
        at_target: Callable[[], None] = lambda: None,
        on_sent: Callable[[], None] | None = None,
    ) -> None:
        """`arrive_at` and `back` are the fast-mode model's; on the testbed the request leaves now (Q53).
        `at_target` runs as it leaves: the caller cannot see the arrival (logging only)."""
        at_target()
        self._send("POST", target, "/tasks", encode(req), on_reply, on_sent)

    def probe(
        self,
        target: str,
        arrive_at: float,
        key: AttemptKey,
        back: float,
        on_answer: Callable[[tuple[Status, float | None]], None],
    ) -> None:
        self._send("GET", target, "/tasks", encode(key), on_answer)

    def cancel(self, target: str, arrive_at: float, key: AttemptKey) -> None:
        self._send("DELETE", target, "/tasks", encode(key), lambda r: None)


class Cut(Exception):
    """An upload cut by a handover or a gap before its last bit (9.7)."""


def make_app(clock: RealClock, target: Target, timeout: float = 5.0) -> Any:
    from fastapi import FastAPI, HTTPException, Request, Response

    app = FastAPI()

    async def on_loop(fn: Callable[[], object]) -> Response:
        fut: concurrent.futures.Future[object] = concurrent.futures.Future()

        def run() -> None:
            try:
                fut.set_result(fn())
            except Exception as e:  # pragma: no cover - surfaced to the caller as a 500
                fut.set_exception(e)

        clock.call_soon(run)
        # Awaited, not blocking: the server keeps taking requests while the zone loop works.
        result = await asyncio.wait_for(asyncio.wrap_future(fut), timeout)
        return Response(encode(result), media_type="application/json")

    async def body(request: Request, kind: type) -> Any:
        obj = decode(await request.body())
        if not isinstance(obj, kind):
            raise HTTPException(422, f"expected a {kind.__name__}")
        return obj

    uploads: dict[object, concurrent.futures.Future[object]] = {}  # task id -> its reply, until t_rx or a cut

    @app.post("/tasks")
    async def post(request: Request) -> Response:
        """A placed attempt from a zone (TaskRequest), or a car's upload (Upload, 5.3, 15.3): the upload runs
        through the zone's software radio pipe and is answered at t_rx; a cut upload gets 409 (9.7)."""
        obj = decode(await request.body())
        if isinstance(obj, TaskRequest):
            return await on_loop(lambda: target.handle(obj))
        if not isinstance(obj, Upload):
            raise HTTPException(422, "expected a TaskRequest or an Upload")
        zone: Any = target
        tid = obj.task.task_id
        fut: concurrent.futures.Future[object] = concurrent.futures.Future()
        uploads[tid] = fut

        def rx() -> None:
            uploads.pop(tid, None)
            fut.set_result(UploadReply(tid, zone.zone_id))

        clock.call_soon(lambda: zone.upload(obj.task, rate=obj.rate, b_car=obj.b_car, on_rx=rx))
        try:
            reply = await asyncio.wait_for(asyncio.wrap_future(fut), timeout)
        except Cut:
            raise HTTPException(409, "upload cut before its last bit") from None
        return Response(encode(reply), media_type="application/json")

    @app.delete("/uploads")
    async def cut(request: Request) -> Response:
        tid = await body(request, TaskId)

        def run() -> bool:
            done = bool(target.cut_upload(tid))  # type: ignore[attr-defined]
            fut = uploads.pop(tid, None)
            if done and fut is not None:
                fut.set_exception(Cut())
            return done

        return await on_loop(run)

    @app.get("/tasks")
    async def get(request: Request) -> Response:
        key = await body(request, AttemptKey)
        return await on_loop(lambda: target.executor.probe(key))

    @app.delete("/tasks")
    async def delete(request: Request) -> Response:
        key = await body(request, AttemptKey)
        return await on_loop(lambda: target.executor.cancel(key))

    return app
