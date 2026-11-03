"""Checkpoints: whole Python objects (a training session with its networks, optimisers, replay buffers and
random generators; a trained method) written with torch.save. The config's frozen mappings
(types.MappingProxyType) are made picklable here. Writes are atomic: a crash never leaves half a file."""

from __future__ import annotations

import copyreg
import types
from pathlib import Path
from typing import Any


def _proxy(d: dict[Any, Any]) -> types.MappingProxyType[Any, Any]:
    return types.MappingProxyType(d)


copyreg.pickle(types.MappingProxyType, lambda m: (_proxy, (dict(m),)))


def save(obj: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load(path: Path) -> Any:
    import torch

    return torch.load(path, weights_only=False)  # our own files only (data/<map>/train/)
