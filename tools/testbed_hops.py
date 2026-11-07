"""Where a task's time goes on the testbed, hop by hop (PLAN Q1; final-design 5.4: the shared host clock,
for analysis only). Needs a run made with PCNME_TRACE=<n> (every n-th task traced).

    set PCNME_TRACE=5 && uv run --extra testbed python -m pcnme.testbed.runner --scenario S4 ...
    uv run python tools/testbed_hops.py data/vfc/testbed/results/S4-greedy-3000-testbed
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from pcnme.testbed.trace import HOPS, hop_durations


def main() -> None:
    run = Path(sys.argv[1])
    rows = [
        json.loads(line) for f in sorted(run.glob("trace-*.jsonl")) for line in f.read_text().splitlines()
    ]
    if not rows:
        raise SystemExit(f"no trace-*.jsonl in {run}: run the testbed with PCNME_TRACE=<n>")
    hops = hop_durations(rows)
    print(f"{run.name}: {len(rows)} trace events\n")
    print("| hop | n | median ms | p90 ms |")
    print("| --- | --- | --- | --- |")
    for name, _, _ in HOPS:
        v = sorted(hops[name])
        if v:
            p90 = v[min(len(v) - 1, int(0.9 * len(v)))]
            print(f"| {name} | {len(v)} | {1000 * statistics.median(v):.1f} | {1000 * p90:.1f} |")


if __name__ == "__main__":
    main()
