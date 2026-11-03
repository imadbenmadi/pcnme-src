"""PCNME training: NSGA-II, w*, demonstrations, DQfD pretraining, online training of PCNME and the
learning baselines. See RUN_TRAINING.md.

uv run python train.py --maps vfc
"""

from pcnme.pipeline.train import main

if __name__ == "__main__":  # the guard keeps spawned worker processes from rerunning main
    raise SystemExit(main())
