"""The zone's network thread (testbed/netloop.py; PLAN M9.3)."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import pytest

pytest.importorskip("httpx")


def test_httpcore_finds_its_async_library_without_searching_the_import_path() -> None:
    """Profiled on the testbed (5 Nov): httpcore runs `import sniffio` on every request; without sniffio
    installed each one failed after a full path search, a quarter of the busiest zone's CPU."""
    import pcnme.testbed.netloop  # noqa: F401

    if importlib.util.find_spec("sniffio") is None:
        assert "sniffio" in sys.modules and sys.modules["sniffio"] is None
    from httpcore._synchronization import current_async_library

    async def which() -> str:
        return current_async_library()

    import asyncio

    assert asyncio.run(which()) == "asyncio"


class _Server:
    """A tiny HTTP/1.1 server: echoes the body with Content-Length (or chunked), keeps connections alive."""

    def __init__(self, chunked: bool = False) -> None:
        self.connections = 0
        self.chunked = chunked

    async def handle(self, reader: Any, writer: Any) -> None:
        self.connections += 1
        while True:
            try:
                head = await reader.readuntil(b"\r\n\r\n")
            except Exception:
                break
            n = int(
                next(h.split(b":")[1] for h in head.split(b"\r\n") if h.lower().startswith(b"content-length"))
            )
            body = await reader.readexactly(n)
            out = body[::-1]
            if self.chunked:
                payload = b"%x\r\n%s\r\n0\r\n\r\n" % (len(out), out) if out else b"0\r\n\r\n"
                writer.write(b"HTTP/1.1 201 Created\r\nTransfer-Encoding: chunked\r\n\r\n" + payload)
            else:
                writer.write(b"HTTP/1.1 201 Created\r\nContent-Length: %d\r\n\r\n%s" % (len(out), out))
            await writer.drain()
        writer.close()


def _serve(server: _Server) -> tuple[str, Any]:
    import asyncio
    import threading

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    srv = asyncio.run_coroutine_threadsafe(asyncio.start_server(server.handle, "127.0.0.1", 0), loop).result(
        5
    )
    port = srv.sockets[0].getsockname()[1]
    return f"http://127.0.0.1:{port}", loop


@pytest.mark.parametrize("chunked", [False, True])
def test_requests_get_their_replies_and_reuse_warm_connections(chunked: bool) -> None:
    import threading

    from pcnme.testbed.clock import RealClock
    from pcnme.testbed.netloop import NetLoop

    server = _Server(chunked)
    base, _ = _serve(server)
    clock = RealClock()
    net = NetLoop(clock, timeout=2.0)
    got: list[tuple[int, bytes]] = []
    sent: list[int] = []
    done = threading.Event()

    def send(i: int) -> None:
        def on_done(status: int, body: bytes) -> None:
            got.append((status, body))
            if len(got) == 20:
                done.set()

        net.request("POST", base, "/x", b"abc%d" % i, on_done, on_sent=lambda: sent.append(i))

    def first() -> None:
        for i in range(10):
            send(i)

    def second() -> None:
        for i in range(10, 20):
            send(i)

    clock.call_soon(first)
    clock.call_soon(lambda: clock.schedule(clock.now + 0.3, second))
    threading.Thread(target=clock.run_for, args=(3.0,), daemon=True).start()
    assert done.wait(5)
    assert sorted(got) == sorted((201, (b"abc%d" % i)[::-1]) for i in range(20))
    assert sorted(sent) == list(range(20))
    assert server.connections <= 10  # the second batch rode the warm connections


def test_a_refused_target_reports_status_zero() -> None:
    import threading

    from pcnme.testbed.clock import RealClock
    from pcnme.testbed.netloop import NetLoop

    clock = RealClock()
    net = NetLoop(clock, timeout=0.5)
    got: list[int] = []
    done = threading.Event()
    clock.call_soon(
        lambda: net.request("POST", "http://127.0.0.1:9", "/x", b"", lambda s, b: (got.append(s), done.set()))
    )
    threading.Thread(target=clock.run_for, args=(3.0,), daemon=True).start()
    assert done.wait(5) and got == [0]
