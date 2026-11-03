"""The live screen of train.py and evaluate.py (rich): the map, the stage, one row per job (phase, progress
bar, done / total, elapsed, ETA, latest numbers) and the last messages. Jobs run in worker processes and
report through a queue (`Reporter`); `run_jobs` drains it while they run. Without a terminal (a pipe, a log)
it prints one line per job event instead.
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import queue as queue_mod
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


def hms(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds == float("inf"):
        return "-"
    s = int(seconds)
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}"


class Reporter:
    """A job's handle to the screen: progress, numbers, a message. Picklable (the queue is a manager's)."""

    def __init__(self, q: Any, job: str) -> None:
        self.q, self.job = q, job

    def __call__(self, **msg: Any) -> None:
        self.q.put({"job": self.job, **msg})


@dataclass
class Row:
    job: str
    phase: str = "queued"
    done: float = 0.0
    total: float = 0.0
    unit: str = ""
    started: float | None = None
    ended: float | None = None
    status: str = "queued"  # queued | running | done | failed | skipped
    numbers: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        return (self.ended or time.monotonic()) - self.started

    @property
    def eta(self) -> float | None:
        if self.status != "running" or not self.total or self.done <= 0:
            return None
        return self.elapsed * (self.total - self.done) / self.done


class Screen:
    def __init__(self, title: str, *, plain: bool | None = None) -> None:
        self.title = title
        self.plain = (not sys.stdout.isatty()) if plain is None else plain
        self.stage_name = ""
        self.stage_started = time.monotonic()
        self.started = time.monotonic()
        self.rows: dict[str, Row] = {}
        self.messages: deque[str] = deque(maxlen=6)
        self._live: Any = None

    # --- lifecycle ------------------------------------------------------------------------------------------

    def __enter__(self) -> Screen:
        if not self.plain:
            from rich.live import Live

            self._live = Live(self._render(), refresh_per_second=2, screen=False, transient=False)
            self._live.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._live is not None:
            self._live.update(self._render())
            self._live.__exit__(*exc)

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    # --- what the stages call -------------------------------------------------------------------------------

    def stage(self, name: str, jobs: Sequence[str] = ()) -> None:
        self.stage_name, self.stage_started = name, time.monotonic()
        self.rows = {j: Row(j) for j in jobs}
        self.say(f"stage: {name}" + (f" ({len(jobs)} jobs)" if jobs else ""))

    def say(self, text: str) -> None:
        line = f"{time.strftime('%H:%M:%S')}  {text}"
        self.messages.append(line)
        if self.plain:
            print(line, flush=True)
        self.refresh()

    def update(self, msg: dict[str, Any]) -> None:
        row = self.rows.setdefault(msg["job"], Row(msg["job"]))
        before = row.status
        if row.started is None and msg.get("status", "running") == "running":
            row.started = time.monotonic()
        for k in ("phase", "done", "total", "unit", "status"):
            if k in msg:
                setattr(row, k, msg[k])
        if row.status == "queued":
            row.status = "running"
        row.numbers.update(msg.get("numbers", {}))
        if row.status in ("done", "failed", "skipped") and row.ended is None:
            row.ended = time.monotonic()
        if "message" in msg:
            self.say(f"{row.job}: {msg['message']}")
        elif self.plain and row.status != before:
            self.say(f"{row.job}: {row.status} {row.phase}")
        self.refresh()

    # --- drawing --------------------------------------------------------------------------------------------

    def _render(self) -> Any:
        from rich.console import Group
        from rich.panel import Panel
        from rich.progress_bar import ProgressBar
        from rich.table import Table
        from rich.text import Text

        rows = list(self.rows.values())
        n_done = sum(r.status in ("done", "skipped") for r in rows)
        running = [r for r in rows if r.status == "running"]
        stage_eta = None
        if rows and n_done:
            per = (time.monotonic() - self.stage_started) / max(n_done, 1)
            stage_eta = per * (len(rows) - n_done)
        head = Text.assemble(
            (self.title, "bold"),
            f"   stage: {self.stage_name}   jobs {n_done}/{len(rows)} done, {len(running)} running   ",
            f"elapsed {hms(time.monotonic() - self.started)}   stage ETA {hms(stage_eta)}",
        )
        t = Table(expand=True, show_edge=False, pad_edge=False)
        for c, kw in (
            ("job", {"no_wrap": True}),
            ("phase", {"no_wrap": True}),
            ("progress", {"ratio": 2}),
            ("done", {"justify": "right", "no_wrap": True}),
            ("elapsed", {"justify": "right"}),
            ("ETA", {"justify": "right"}),
            ("latest", {"ratio": 3}),
        ):
            t.add_column(c, **kw)
        shown = (
            running
            + [r for r in rows if r.status == "failed"]
            + [r for r in rows if r.status == "queued"][:3]
        )
        for r in shown:
            bar = ProgressBar(total=max(r.total, 1), completed=min(r.done, r.total or 1), width=None)
            done = f"{r.done:,.0f}/{r.total:,.0f} {r.unit}" if r.total else ""
            nums = "  ".join(
                f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in r.numbers.items()
            )
            style = {"failed": "red", "queued": "dim"}.get(r.status, "")
            t.add_row(r.job, r.phase, bar, done, hms(r.elapsed), hms(r.eta), nums, style=style)
        log = Text("\n".join(self.messages), style="dim")
        return Panel(Group(head, t, log), title="PCNME", border_style="cyan")


def run_jobs(
    screen: Screen,
    fn: Callable[..., Any],
    jobs: Sequence[tuple[str, tuple[Any, ...]]],
    workers: int,
) -> dict[str, Any]:
    """Run fn(*args, reporter) for each (name, args) in worker processes (spawn: the same on Windows and
    Linux), `workers` at a time, showing their reports. A failed job is shown and logged; the others go on.
    Returns name -> result, or the exception."""
    out: dict[str, Any] = {}
    if not jobs:
        return out
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        q = manager.Queue()
        with cf.ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx) as pool:
            futures = {pool.submit(fn, *args, Reporter(q, name)): name for name, args in jobs}
            pending = set(futures)
            while pending:
                _drain(screen, q)
                finished, pending = cf.wait(pending, timeout=0.5)
                for f in finished:
                    name = futures[f]
                    _drain(screen, q)
                    try:
                        out[name] = f.result()
                        if screen.rows.get(name, Row(name)).status not in ("done", "skipped"):
                            screen.update({"job": name, "status": "done"})
                    except Exception as e:  # shown, logged by the job; the other jobs go on
                        out[name] = e
                        screen.update({"job": name, "status": "failed", "message": f"FAILED: {e!r}"[:300]})
            _drain(screen, q)
    return out


def _drain(screen: Screen, q: Any) -> None:
    while True:
        try:
            screen.update(q.get_nowait())
        except queue_mod.Empty:
            return
        except (EOFError, BrokenPipeError):
            return
