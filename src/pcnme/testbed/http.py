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
from collections.abc import Callable, Mapping
from typing import Any

from pcnme.bus.calls import Target
from pcnme.ids import AttemptKey
from pcnme.node.executor import Reply, Status, TaskRequest
from pcnme.testbed.clock import RealClock
from pcnme.testbed.codec import decode, encode
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

    def _send(
        self, method: str, target: str, path: str, body: bytes, on_reply: Callable[[Any], None]
    ) -> None:
        url = self._urls[target] + path

        def work() -> None:
            try:
                r = self._http.request(method, url, content=body)
            except Exception:  # no answer: lost with the target, the caller's timers notice (9.3)
                return
            if r.status_code == 200:
                reply = decode(r.content)
                self._clock.call_soon(lambda: on_reply(reply))

        self._pool.submit(work)

    def _at(self, at: float | None, fn: Callable[[], None]) -> None:
        if at is None:
            fn()
        else:
            self._clock.schedule(at, fn)

    def post(
        self,
        target: str,
        arrive_at: float | None,
        req: TaskRequest,
        back: float,
        on_reply: Callable[[tuple[Reply, Snapshot]], None],
        at_target: Callable[[], None] = lambda: None,
    ) -> None:
        def go() -> None:
            at_target()
            self._send("POST", target, "/tasks", encode(req), on_reply)

        self._at(arrive_at, go)

    def probe(
        self,
        target: str,
        arrive_at: float,
        key: AttemptKey,
        back: float,
        on_answer: Callable[[tuple[Status, float | None]], None],
    ) -> None:
        self._at(arrive_at, lambda: self._send("GET", target, "/tasks", encode(key), on_answer))

    def cancel(self, target: str, arrive_at: float, key: AttemptKey) -> None:
        self._at(arrive_at, lambda: self._send("DELETE", target, "/tasks", encode(key), lambda r: None))


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

    @app.post("/tasks")
    async def post(request: Request) -> Response:
        req = await body(request, TaskRequest)
        return await on_loop(lambda: target.handle(req))

    @app.get("/tasks")
    async def get(request: Request) -> Response:
        key = await body(request, AttemptKey)
        return await on_loop(lambda: target.executor.probe(key))

    @app.delete("/tasks")
    async def delete(request: Request) -> Response:
        key = await body(request, AttemptKey)
        return await on_loop(lambda: target.executor.cancel(key))

    return app
