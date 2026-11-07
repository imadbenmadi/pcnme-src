"""Serving helpers for the testbed entry points (PLAN M9.1 step 6): run a FastAPI app beside the RealClock
loop, and run request handlers on that loop (zone code never runs on a server thread, testbed/http.py)."""

# No `from __future__ import annotations`: FastAPI must resolve `request: Request` (imported locally).

import asyncio
import concurrent.futures
import threading
from collections.abc import Callable
from typing import Any

from pcnme.testbed.clock import RealClock

PORT = 8000


async def on_loop(clock: RealClock, fn: Callable[[], object], timeout: float = 5.0) -> object:
    """Run `fn` on the loop thread and await its result without blocking the server."""
    fut: concurrent.futures.Future[object] = concurrent.futures.Future()

    def run() -> None:
        try:
            fut.set_result(fn())
        except Exception as e:  # pragma: no cover - surfaced as a 500
            fut.set_exception(e)

    clock.call_soon(run)
    return await asyncio.wait_for(asyncio.wrap_future(fut), timeout)


def serve(app: Any, port: int = PORT) -> threading.Thread:  # pragma: no cover - containers only
    import uvicorn

    t = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": "0.0.0.0", "port": port, "log_level": "warning"},
        name="http",
        daemon=True,
    )
    t.start()
    return t


def results_app(clock: RealClock, fleet: Any) -> Any:
    """The car container's POST /results (9.5): the Confirm in the reply, 204 for no confirm."""
    from fastapi import FastAPI, Request, Response

    from pcnme.car.car import Result
    from pcnme.testbed.codec import decode, encode

    app = FastAPI()

    @app.post("/results")
    async def results(request: Request) -> Response:
        res = decode(await request.body())
        if fleet.tracer is not None and isinstance(res, Result):
            fleet.tracer.mark(res.task_id, res.attempt, "result at the car", None)
        confirm = await on_loop(clock, lambda: fleet.result(res))
        if confirm is None:
            return Response(status_code=204)
        return Response(encode(confirm), media_type="application/json")

    return app
