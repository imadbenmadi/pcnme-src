"""run_live.py: one long testbed run to watch as it goes (PLAN R2b, V3): every zone pushes its tasks, slots,
cars and its agent's online learning to InfluxDB; Grafana (http://localhost:3000, dashboard "PCNME live") and
the viewer (`uv run python view.py`, source "Live testbed") show them while the run goes.

Not a measurement: the live points cost the zones a little CPU, so the thesis numbers come from run_testbed.py
(live off). The run's rows are still written, to data/vfc/testbed/live/results/<run>/, apart from the batch's.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from pcnme.pipeline.evaluate import METHODS


def live_name(scenario: str, method: str, seed: int, now: float | None = None) -> str:
    """A live run's name: unique per start, readable in Grafana's and the viewer's run lists."""
    stamp = time.strftime("%m%d-%H%M%S", time.localtime(now))
    return f"{scenario}-{method}-{seed}-{stamp}"


def main(argv: list[str] | None = None) -> int:
    from pcnme.config import load_config
    from pcnme.paths import MapPaths
    from pcnme.testbed.runner import run

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--scenario", default="S2")
    ap.add_argument("--method", default="pcnme", choices=METHODS)
    ap.add_argument("--seed", type=int, default=3000, help="an evaluation seed (16.3)")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--containers", type=int, default=2, help="car containers, 20 cars each")
    a = ap.parse_args(argv)
    name = live_name(a.scenario, a.method, a.seed)
    paths = MapPaths("vfc")
    print(f"PCNME live run {name}: {a.scenario}, {a.method}, seed {a.seed}, {20 * a.containers} cars, "
          f"{a.minutes:g} min")  # fmt: skip
    print("  Grafana: http://localhost:3000/d/pcnme-live   (user admin, GRAFANA_PASSWORD in docker/.env)")
    print("  Viewer:  uv run python view.py   -> Live testbed")
    out = run(load_config(), a.scenario, a.seed, a.method, 60 * a.minutes, paths.testbed / "live",
              a.containers, models=paths.root / "train", live=name)  # fmt: skip
    print(report(out, a.method))
    print(f"Done: {out}. Grafana and InfluxDB stay up; stop them with "
          "docker compose -f docker/compose.yaml --profile testbed down")  # fmt: skip
    return 0


def report(run_dir: Any, method: str) -> str:
    """The run's on-time share, each zone's online updates and trainer, and the health check's findings."""
    import json
    from pathlib import Path

    from pcnme.config import load_config
    from pcnme.metrics.stats import run_metrics
    from pcnme.pipeline.testbed import health

    d = Path(run_dir)
    m = run_metrics(d, load_config().fast_mode.early_decisions).metrics
    lines = [f"  on time {m['on_time']:.1%}, dropped {m['dropped']:.1%}, failed {m['failed']:.1%}, "
             f"mean latency {1000 * m['latency_mean_s']:.0f} ms"]  # fmt: skip
    for f in sorted(d.glob("counts-*.json")):
        c = json.loads(f.read_text(encoding="utf-8"))
        zone = f.stem.removeprefix("counts-")
        if "agent updates" in c:
            lines.append(f"  {zone}: {c['agent updates']} online updates, "
                         f"{c.get('agent samples', 0)} samples, "
                         f"{c.get('trainer weights installed', 0)} weight installs, "
                         f"trainer {'alive' if c.get('trainer alive') else 'not running'}")  # fmt: skip
        else:
            lines.append(f"  {zone}: a rule, nothing learns")
    problems = health(d, method, load_config().testbed.preflight_max_no_ack)
    lines.append("  health: ok" if not problems else "  health: " + "; ".join(problems))
    return "\n".join(lines)
