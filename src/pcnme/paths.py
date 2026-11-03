"""Where data lives (data/DATA.md): $PCNME_DATA_DIR, by default the repository's data/ folder, with one folder
per map:

    data/<map>/nsga/        NSGA-II fronts (<scenario>-w<k>.json, ignored) and fronts.csv (committed)
    data/<map>/w_star.json  the map's w* and the knee weights it is the median of (committed)
    data/<map>/demos/       demonstrations (<kind>-<scenario>-w<k>.parquet, ignored) and demos.csv (committed)
    data/<map>/train/<scenario>-<method>/   checkpoint.pt and model.pt (ignored), episodes.csv (committed)
    data/<map>/results/     summary.csv, tables, figures (committed); runs/<run_id>/ Parquet (ignored)
    data/<map>/traces/      trace maps: replayable SUMO traces (ignored)
    data/<map>/logs/        one log file per stage
    data/<map>/manifest.json   the training run: config hash, commit, seeds, budget, times
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Where outputs go: $PCNME_DATA_DIR, by default data/ (train.py --smoke uses data/smoke/)."""
    return Path(os.environ.get("PCNME_DATA_DIR", str(REPO / "data"))).resolve()


def maps_dir() -> Path:
    """Downloaded map data (LuST): $PCNME_MAPS_DIR, by default data/maps/; never moved by --smoke."""
    return Path(os.environ.get("PCNME_MAPS_DIR", str(REPO / "data" / "maps"))).resolve()


@dataclass(frozen=True)
class MapPaths:
    """The folders of one map's outputs."""

    name: str

    @property
    def root(self) -> Path:
        return data_dir() / self.name

    @property
    def nsga(self) -> Path:
        return self.root / "nsga"

    @property
    def w_star(self) -> Path:
        return self.root / "w_star.json"

    @property
    def demos(self) -> Path:
        return self.root / "demos"

    def train(self, scenario: str, method: str) -> Path:
        return self.root / "train" / f"{scenario}-{method}"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def runs(self) -> Path:
        return self.results / "runs"

    @property
    def traces(self) -> Path:
        return self.root / "traces"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def calibrated(self) -> Path:
        """Trace maps: the coverage measured on the traces, merged into the map's config (pcnme.maps)."""
        return self.root / "calibrated.yaml"
