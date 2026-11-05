"""One network thread per testbed process (PLAN M9.3): every outgoing HTTP request of a zone (its calls to
peers and the cloud, its results to the cars) runs on one asyncio loop, instead of a thread per request.
Profiled on the testbed: a zone with about 250 HTTP threads starved its own loop thread of the GIL (a 100 ms
state publish took 64 ms), and every task timed out.

Each target gets a small pool of keep-alive HTTP/1.1 connections (`Pool`, plain asyncio streams). Profiled
again (5 Nov): httpx's pool re-scanned every connection and polled every socket on each request, about two
thirds of the busiest zone's CPU at S2's load. The targets are our own uvicorn servers, so plain HTTP/1.1
with Content-Length (or chunked) replies is all that is needed. Tests may still pass an httpx transport.

`request` is called from the zone's loop thread; `on_done(status, body)` runs back on that thread
(RealClock.call_soon). A request that fails (refused, reset, timed out) reports status 0: the caller's own
timers notice a lost reply (9.3), as in fast mode.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable
from typing import Any
from urllib.parse import urlsplit

from pcnme.testbed.clock import RealClock

# httpcore runs `import sniffio` on every request and falls back to asyncio when it fails; sniffio is not
# installed (anyio 4 dropped it), so each request searched the whole import path: about a quarter of the
# busiest zone's CPU (profiled 5 Nov). A None entry makes the failed import immediate; asyncio is the answer.
if "sniffio" not in sys.modules and importlib.util.find_spec("sniffio") is None:
    sys.modules["sniffio"] = None  # type: ignore[assignment]

OnDone = Callable[[int, bytes], None]
PER_TARGET = 32  # warm connections per target (15.4)
IDLE_S = 4.0  # an idle connection older than this is closed, not reused (uvicorn closes idle ones after 5 s)

Conn = tuple[asyncio.StreamReader, asyncio.StreamWriter, float]


class Pool:
    """At most `size` keep-alive HTTP/1.1 connections to one target; a request waits for a free one."""

    def __init__(self, base: str, size: int) -> None:
        u = urlsplit(base)
        self._host, self._port, self._netloc = u.hostname or "localhost", u.port or 80, u.netloc
        self._idle: list[Conn] = []
        self._free = asyncio.Semaphore(size)

    async def _connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        while self._idle:
            reader, writer, since = self._idle.pop()
            if reader.at_eof() or writer.is_closing() or time.monotonic() - since > IDLE_S:
                writer.close()
                continue
            return reader, writer
        return await asyncio.open_connection(self._host, self._port)

    async def request(
        self, method: str, path: str, body: bytes, on_sent: Callable[[], None] | None
    ) -> tuple[int, bytes]:
        async with self._free:
            reader, writer = await self._connect()
            try:
                head = (
                    f"{method} {path} HTTP/1.1\r\nHost: {self._netloc}\r\nContent-Length: {len(body)}\r\n\r\n"
                )
                writer.write(head.encode("ascii") + body)
                await writer.drain()
                if on_sent is not None:
                    on_sent()
                status, content, keep = await _reply(reader)
            except BaseException:
                writer.close()
                raise
            if keep:
                self._idle.append((reader, writer, time.monotonic()))
            else:
                writer.close()
            return status, content


async def _reply(reader: asyncio.StreamReader) -> tuple[int, bytes, bool]:
    """One HTTP/1.1 response: (status, body, the connection can be reused)."""
    lines = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1").split("\r\n")
    status = int(lines[0].split(" ", 2)[1])
    headers = {k.strip().lower(): v.strip() for k, _, v in (h.partition(":") for h in lines[1:] if h)}
    keep = headers.get("connection", "").lower() != "close"
    if "content-length" in headers:
        return status, await reader.readexactly(int(headers["content-length"])), keep
    if headers.get("transfer-encoding", "").lower() == "chunked":
        parts = []
        while n := int((await reader.readuntil(b"\r\n")).split(b";")[0], 16):
            parts.append(await reader.readexactly(n))
            await reader.readexactly(2)
        await reader.readuntil(b"\r\n")  # no trailers
        return status, b"".join(parts), keep
    if status in (204, 304) or 100 <= status < 200:
        return status, b"", keep
    return status, await reader.read(), False  # until the server closes


class NetLoop:
    def __init__(self, clock: RealClock, *, timeout: float = 5.0, transport: Any = None) -> None:
        """`transport`: an httpx transport for every client (tests: httpx.ASGITransport)."""
        self._clock = clock
        self._timeout = timeout
        self._transport = transport
        self._clients: dict[str, Any] = {}
        self._pools: dict[str, Pool] = {}
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

    def _pool(self, base: str) -> Pool:
        p = self._pools.get(base)
        if p is None:
            p = self._pools[base] = Pool(base, PER_TARGET)
        return p

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

        def sent() -> None:
            if on_sent is not None:
                self._clock.call_soon(on_sent)

        async def chunks() -> AsyncIterator[bytes]:
            yield body
            sent()

        async def go() -> None:
            try:
                if self._transport is None:
                    status, content = await asyncio.wait_for(
                        self._pool(base).request(method, path, body, sent if on_sent else None), self._timeout
                    )
                else:
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
            for p in self._pools.values():
                for _, writer, _ in p._idle:
                    writer.close()

        asyncio.run_coroutine_threadsafe(shut(), self._loop).result(5)
        self._loop.call_soon_threadsafe(self._loop.stop)
