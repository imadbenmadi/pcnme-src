"""Direct calls in fast mode: the HTTP calls of final-design 5.3 (POST, GET, DELETE /tasks) (15.2).

"Direct call on the target object after the modelled transfer time": the caller works out when its request
arrives (an input pays its pipes, 4.1; a GET or DELETE carries no bulk data, so only the link delay), the
target handles it then, and the reply returns after the reply path's delay (replies carry no bulk data, 4.2).
A dead target handles nothing and never replies; the caller's own timer (upload timeout, probe timeout,
9.3) notices.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from pcnme.sim.clock import EventClock

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
