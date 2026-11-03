"""Where the first-results runs are (tools/first_results.py): one line per scenario x method.

    python tools/status.py [results/first]

Phases: queued -> training (time since start; a run writes nothing until training ends) -> evaluating k/20
(one directory per finished evaluation seed) -> done.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

METHODS = ("greedy", "random", "pcnme", "d3qn-scratch", "d3qn-greedy", "vfc-ppo", "tang-wong", "qeco")
SCENARIOS = ("S1", "S2", "S3", "S4", "S5", "S6")
SEEDS = 20


def phase(out: Path, s: str, m: str) -> str:
    done = sum((out / f"{s}-{m}-{3000 + k}-fast").exists() for k in range(SEEDS))
    if done == SEEDS:
        return "done"
    log = out / "logs" / f"{s}-{m}.log"
    if not log.exists():
        return "queued"
    if done or (out / f"{s}-{m}-train").exists() or m in ("greedy", "random"):
        return f"evaluating {done}/{SEEDS}"
    mins = (time.time() - log.stat().st_ctime) / 60
    return f"training {mins // 60:.0f}h{mins % 60:02.0f}m"


def main() -> None:
    default = Path(os.environ.get("PCNME_DATA_DIR", ".")) / "results" / "first"
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    w = 17
    print("method".ljust(14) + "".join(s.ljust(w) for s in SCENARIOS))
    counts: dict[str, int] = {}
    for m in METHODS:
        cells = [phase(out, s, m) for s in SCENARIOS]
        for c in cells:
            key = c.split()[0]
            counts[key] = counts.get(key, 0) + 1
        print(m.ljust(14) + "".join(c.ljust(w) for c in cells))
    total = len(METHODS) * len(SCENARIOS)
    print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + f" (of {total})")


if __name__ == "__main__":
    main()
