"""The calls interface changes nothing in fast mode (PLAN M9.1 step 4): a short S2 city run gives the same
record digest as before the interface existed (taken on main at 054bc3d)."""

from __future__ import annotations

import hashlib

from pcnme.config import load_config
from pcnme.experiments.city import City


def digest() -> str:
    c = City(load_config(), "S2", "greedy", 2000, n_cars=60).run(3.0)
    rows = sorted(
        (str(r.task.task_id), r.anchor, r.executor, r.outcome, repr(r.closed_at), len(r.attempts))
        for r in c.records
    )
    return hashlib.sha256(repr(rows).encode()).hexdigest()


GOLDEN = "23f26a77daa19acf49fb84496006134dc2b8873b3f10588c3147cd968d722e74"


def test_the_fast_mode_run_is_unchanged() -> None:
    assert digest() == GOLDEN
