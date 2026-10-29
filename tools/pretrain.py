"""DQfD pretraining on stored demonstrations (PLAN M6.4; final-design 13.5).

Loads <demos>/*.parquet (tools/make_demos.py), pretrains one network and prints the held-out expert
agreement per measurement; saves the best network (torch state dict) and the curve (JSON) to <out>.

    python tools/pretrain.py [--updates 200000] [--eval-every 1000] [--seed 2000]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import torch

from pcnme.agent.dqfd import pretrain
from pcnme.config import load_config
from pcnme.expert.demos import load_samples
from pcnme.sim.rng import RandomStreams


def main(argv: Sequence[str] | None = None) -> None:
    root = Path(os.environ.get("PCNME_DATA_DIR", ".")) / "results"
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demos", type=Path, default=root / "demos")
    ap.add_argument("--out", type=Path, default=root / "pretrain")
    ap.add_argument("--updates", type=int, default=None, help="default: dqfd.pretrain_updates")
    ap.add_argument("--eval-every", type=int, default=None, help="default: dqfd.eval_every")
    ap.add_argument("--seed", type=int, default=2000, help="a training seed (fast_mode.seed_split)")
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.eval_every is not None:
        cfg = replace(cfg, dqfd=replace(cfg.dqfd, eval_every=args.eval_every))
    demos = load_samples(sorted(args.demos.glob("*.parquet")))
    streams = RandomStreams.from_config(cfg, args.seed, "training")
    rng = streams.generator("agent", "pretrain", "replay")
    init = int(streams.generator("agent", "pretrain", "init").integers(2**31 - 1))
    expert = sum(s.expert for _, s in demos)
    print(f"{len(demos)} demonstration transitions ({expert} expert), {len({g for g, _ in demos})} windows")
    t = time.perf_counter()
    out = pretrain(cfg, demos, rng, init, updates=args.updates)
    wall = time.perf_counter() - t
    print(f"held-out windows: {sorted(out.held_out)}")
    for u, a in out.curve.agreement:
        print(f"  update {u:7d}: held-out expert agreement {a:.3f}")
    stop = "stopped early" if out.stopped_early else "budget used"
    print(f"{out.updates} updates in {wall:.0f} s ({stop}); best agreement {out.best_agreement:.3f}")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(out.network.state_dict(), args.out / "network.pt")
    (args.out / "curve.json").write_text(
        json.dumps(
            {"agreement": out.curve.agreement, "losses": out.curve.losses[:: max(1, args.eval_every or 100)]}
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
