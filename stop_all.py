"""Stop everything PCNME runs on this machine: containers, networks, the entry points, the ports. See READ.md.

uv run python stop_all.py [--images] [--volumes] [--dry-run]
"""

from pcnme.pipeline.stop import main

if __name__ == "__main__":
    raise SystemExit(main())
