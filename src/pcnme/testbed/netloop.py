"""One network thread per testbed process (PLAN M9.3): every outgoing HTTP request of a zone (its calls to
peers and the cloud, its results to the cars) runs on one asyncio loop with one AsyncClient per target,
instead of a thread per request. Profiled on the testbed: a zone with about 250 HTTP threads starved its
own loop thread of the GIL (a 100 ms state publish took 64 ms), and every task timed out.

`request` is called from the zone's loop thread; `on_done(status, body)` runs back on that thread
(RealClock.call_soon). A request that fails (refused, reset, timed out) reports status 0: the caller's own
timers notice a lost reply (9.3), as in fast mode.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from typing import Any

from pcnme.testbed.clock import RealClock

OnDone = Callable[[int, bytes], None]
PER_TARGET = 32  # warm connections per target (15.4)


class NetLoop:
    def __init__(self, clock: RealClock, *, timeout: float = 5.0, transport: Any = None) -> None:
        """`transport`: an httpx transport for every client (tests: httpx.ASGITransport)."""
        self._clock = clock
        self._timeout = timeout
        self._transport = transport
        self._clients: dict[str, Any] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="net", daemon=True)
        self._thread.start()

    def _client(self, base: str) -> Any:
        import httpx

        c = self._clients.get(base)
        if c is None:
            limits = httpx.Limits(max_connections=PER_TARGET, max_keepalive_connections=PER_TARGET)
            c = httpx.AsyncClient(
                base_url=base, timeout=self._timeout, limits=limits, transport=self._transport
            )
            self._clients[base] = c
        return c

    def request(
        self,
        method: str,
        base: str,
        path: str,
        body: bytes,
        on_done: OnDone,
        on_sent: Callable[[], None] | None = None,
    ) -> None:
        """Send now (from the zone's loop thread); `on_sent` once the body's last chunk is out (C7)."""

        async def chunks() -> AsyncIterator[bytes]:
            yield body
            if on_sent is not None:
                self._clock.call_soon(on_sent)

        async def go() -> None:
            try:
                r = await self._client(base).request(method, path, content=chunks() if on_sent else body)
                status, content = r.status_code, r.content
            except Exception:  # no answer: the caller's timers notice (9.3)
                status, content = 0, b""
            self._clock.call_soon(lambda: on_done(status, content))

        asyncio.run_coroutine_threadsafe(go(), self._loop)

    def close(self) -> None:
        async def shut() -> None:
            for c in self._clients.values():
                await c.aclose()

        asyncio.run_coroutine_threadsafe(shut(), self._loop).result(5)
        self._loop.call_soon_threadsafe(self._loop.stop)
