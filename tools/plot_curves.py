"""Training curves from a run's learning.parquet and episodes.parquet (PLAN M6.6): loss, held-out expert
agreement, on-time share and mean ê per episode, as one PNG next to the files.

    python tools/plot_curves.py results/<run_id>
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args(argv)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    learning = args.run_dir / "learning.parquet"
    if learning.exists():
        rows = pq.read_table(learning).to_pylist()
        for zone, phase in sorted({(r["zone"], r["phase"]) for r in rows}):
            rs = [
                r
                for r in rows
                if (r["zone"], r["phase"]) == (zone, phase) and r.get("total_loss") is not None
            ]
            ax[0, 0].plot([r["update"] for r in rs], [r["total_loss"] for r in rs], label=f"{phase} {zone}")
        agree = sorted(
            (r["update"], r["expert_agreement"]) for r in rows if r.get("expert_agreement") is not None
        )
        ax[0, 1].plot([u for u, _ in agree], [a for _, a in agree], marker="o")
    episodes = args.run_dir / "episodes.parquet"
    if episodes.exists():
        es = pq.read_table(episodes).to_pylist()
        ax[1, 0].plot([e["episode"] for e in es], [e["on_time"] for e in es], marker="o")
        ax[1, 1].plot([e["episode"] for e in es], [e["mean_e_hat"] for e in es], marker="o")
    titles = (
        "loss per update block",
        "held-out expert agreement",
        "on-time share per episode",
        "mean ê per episode",
    )
    for a, t in zip(ax.flat, titles, strict=True):
        a.set_title(t)
    ax[0, 0].set_yscale("log")
    if ax[0, 0].get_legend_handles_labels()[0]:
        ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    out = args.run_dir / "curves.png"
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
