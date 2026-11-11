"""A map's configuration (final-design 16.2, 18): config/default.yaml with the map selected (`map:`), its
`overrides` and, for a trace map, the coverage measured on its traces (data/<map>/calibrated.yaml) merged in.
The config hash names exactly what a run used, so outputs of different configs are never mixed (train.py)."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from pcnme.config import DEFAULT_CONFIG_PATH, Config, config_from_mapping, load_yaml
from pcnme.paths import MapPaths, maps_dir


def merge(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """Deep merge: mappings merge key by key, anything else replaces."""
    for k, v in extra.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), dict):
            merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def map_source(name: str, path: Path = DEFAULT_CONFIG_PATH, *, measured: bool = True) -> dict[str, Any]:
    """The config mapping of one map. `measured`: merge data/<map>/calibrated.yaml when it exists."""
    source = load_yaml(path)
    if name not in source["maps"]:
        raise KeyError(f"no map {name!r} in {path} (maps: {', '.join(source['maps'])})")
    source["map"] = name
    merge(source, source["maps"][name].get("overrides") or {})
    cal = MapPaths(name).calibrated
    if measured and cal.exists():
        merge(source, yaml.safe_load(cal.read_text(encoding="utf-8")) or {})
    return source


def map_config(name: str, path: Path = DEFAULT_CONFIG_PATH) -> Config:
    return config_from_mapping(map_source(name, path))


# How the testbed runs, not what is measured: left out of the identity, so adding or tuning them (11 Nov, Q62)
# does not mark trained models, evaluations or testbed rows as made with another config.
TESTBED_RUNTIME = ("zone_cpus", "zone_cpus_learning", "trainer_sync_s", "preflight_s", "preflight_max_no_ack")


def config_hash(source: Mapping[str, Any]) -> str:
    """A short hash of the full resolved mapping (sorted keys): equal hashes, equal configs; the testbed's
    runtime settings (TESTBED_RUNTIME) left out."""
    tb = source.get("testbed")
    if isinstance(tb, Mapping):
        source = {**source, "testbed": {k: v for k, v in tb.items() if k not in TESTBED_RUNTIME}}
    text = json.dumps(source, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def trace_source(cfg: Config) -> Path:
    """A trace map's SUMO scenario folder (under data/maps/, data/DATA.md)."""
    return maps_dir() / cfg.map.source
