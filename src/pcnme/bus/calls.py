"""Direct calls in fast mode: the HTTP calls of final-design 5.3 (POST, GET, DELETE /tasks) (15.2).

"Direct call on the target object after the modelled transfer time": the caller works out when its request
arrives (an input pays its pipes, 4.1; a GET or DELETE carries no bulk data, so only the link delay), the
target handles it then, and the reply returns after the reply path's delay (replies carry no bulk data, 4.2).
A dead target handles nothing and never replies; the caller's own timer (upload timeout, probe timeout,
9.3) notices.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypeVar

from pcnme.sim.clock import EventClock

if TYPE_CHECKING:
    from pcnme.ids import AttemptKey
    from pcnme.node.executor import Executor, Reply, Status, TaskRequest
    from pcnme.zone.statestore import Snapshot

R = TypeVar("R")


def call(
    clock: EventClock,
    arrive_at: float,
    target_alive: Callable[[], bool],
    handler: Callable[[], R],
    reply_delay: Callable[[R], float],
    on_reply: Callable[[R], None],
    caller_alive: Callable[[], bool] = lambda: True,
) -> None:
    """At `arrive_at` the target handles the request (if it is alive); `reply_delay(reply)` later the caller
    gets the reply (if it is alive)."""

    def at_target() -> None:
        if not target_alive():
            return  # no reply: lost with the target
        reply = handler()
        back = clock.now + reply_delay(reply)
        clock.schedule(back, lambda: on_reply(reply) if caller_alive() else None)

    clock.schedule(arrive_at, at_target)


class Target(Protocol):
    """What a zone or the cloud serves over HTTP (5.3)."""

    def handle(self, req: TaskRequest) -> tuple[Reply, Snapshot]: ...  # POST /tasks

    @property
    def executor(self) -> Executor: ...  # GET and DELETE /tasks/{id}


class Calls(Protocol):
    """The zone's HTTP calls (5.3) to another zone or the cloud. `arrive_at`: when the request reaches the
    target (the zone models its own pipes, 4.1), None if it is there now; `back`: the reply's delay. Fast
    mode (FastCalls) calls the target object; the testbed (testbed/http.py) sends real requests.
    `real_network`: netem gives the delays, so the zone models none (QUESTIONS Q53); `on_sent` then marks
    the end of the zone's own send (C7)."""

    real_network: bool

    def post(
        self,
        target: str,
        arrive_at: float | None,
        req: TaskRequest,
        back: float,
        on_reply: Callable[[tuple[Reply, Snapshot]], None],
        at_target: Callable[[], None] = ...,
        on_sent: Callable[[], None] | None = ...,
    ) -> None: ...

    def probe(
        self,
        target: str,
        arrive_at: float,
        key: AttemptKey,
        back: float,
        on_answer: Callable[[tuple[Status, float | None]], None],
    ) -> None: ...

    def cancel(self, target: str, arrive_at: float, key: AttemptKey) -> None: ...


class FastCalls:
    """Fast mode: the target object handles the request at `arrive_at` (if it is alive), the reply comes
    `back` later (if the caller is alive): exactly `call` above, so a run is unchanged."""

    real_network = False  # the zone models its pipes; `on_sent` is never used

    def __init__(
        self,
        clock: EventClock,
        resolve: Callable[[str], Target],
        alive: Callable[[str], bool],
        caller_alive: Callable[[], bool],
    ) -> None:
        self._clock, self._resolve, self._alive, self._caller_alive = clock, resolve, alive, caller_alive

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
        def handled() -> tuple[Reply, Snapshot]:
            at_target()
            return self._resolve(target).handle(req)

        if arrive_at is None:  # the request is at the target now (the cloud path modelled its arrival)
            if self._alive(target):
                r = handled()
                self._clock.schedule(self._clock.now + back, lambda: on_reply(r))
            return
        call(
            self._clock,
            arrive_at,
            lambda: self._alive(target),
            handled,
            lambda r: back,
            on_reply,
            self._caller_alive,
        )

    def probe(
        self,
        target: str,
        arrive_at: float,
        key: AttemptKey,
        back: float,
        on_answer: Callable[[tuple[Status, float | None]], None],
    ) -> None:
        call(
            self._clock,
            arrive_at,
            lambda: self._alive(target),
            lambda: self._resolve(target).executor.probe(key),
            lambda r: back,
            on_answer,
            self._caller_alive,
        )

    def cancel(self, target: str, arrive_at: float, key: AttemptKey) -> None:
        call(
            self._clock,
            arrive_at,
            lambda: self._alive(target),
            lambda: self._resolve(target).executor.cancel(key),
            lambda r: 0.0,
            lambda r: None,
        )
