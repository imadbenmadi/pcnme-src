"""Training curves per scenario (docs/results-first.md): the on-time share and mean ê of every training
episode, one line per learning method, from <out>/<scenario>-<method>-train/training.json
(experiments/runner.py).

    python tools/plot_training.py results/first [--out docs/figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or a.runs
    out.mkdir(parents=True, exist_ok=True)
    by_scenario: dict[str, dict[str, list[dict[str, float]]]] = {}
    for f in sorted(a.runs.glob("*-train/training.json")):
        scenario, method = f.parent.name.removesuffix("-train").split("-", 1)
        by_scenario.setdefault(scenario, {})[method] = json.loads(f.read_text(encoding="utf-8"))["episodes"]
    for scenario, methods in sorted(by_scenario.items()):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
        for method, eps in sorted(methods.items()):
            x = range(1, len(eps) + 1)
            ax1.plot(x, [100 * e["on_time"] for e in eps], label=method)
            ax2.plot(x, [e["mean_e_hat"] for e in eps], label=method)
        ax1.set(xlabel="training episode", ylabel="on time %", title=f"{scenario}: on-time share")
        ax2.set(xlabel="training episode", ylabel="mean ê", title=f"{scenario}: mean ê")
        ax1.legend(fontsize=7)
        fig.tight_layout()
        path = out / f"training-{scenario}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
