"""Seeded random streams, one per purpose (final-design 15.2; PLAN M0.3; QUESTIONS Q4-Q6).

Every random draw of a run comes from one run seed. Purposes are told apart by name, never by position, so
adding a purpose later never changes the draws of the others.

Two kinds of purpose:

- **Sequential** (arrivals, mobility, nsga): draws that never depend on decisions. `generator(purpose, *key)`
  gives a `numpy.random.Generator` per key (for example one per car), drawn in order.
- **Keyed** (run_times, link_delays, radio): draws that would otherwise shift when a decision changes the
  order of events. `uniform(purpose, *key)` is a pure function of (seed, purpose, key): the same task gets
  the same draw in every NSGA-II plan of a window (common random numbers, 13.2). Q6 fixes the keys: a run
  time by (task_id, attempt), mapped through the inverse CDF of the row the task runs on; a delay by
  (task_id, attempt, link/hop). Keys hold nothing decision-dependent besides the attempt number.

Keyed draws are a keyed BLAKE2b hash of the key (a counter-based generator, as in Random123): about 2 us a
draw, and the same bits on every platform and NumPy version.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from pcnme.config import Config

KeyPart = int | str
Kind = Literal["sequential", "keyed"]

# random_method: M2.8; agent: the network's initial weights, ε-exploration and replay sampling, keyed per
# zone and use (M6); neighbour_order: the per-episode permutation of the neighbour positions (13.6, M6.5)
SEQUENTIAL: tuple[str, ...] = ("arrivals", "mobility", "nsga", "random_method", "agent", "neighbour_order")
KEYED: tuple[str, ...] = ("run_times", "link_delays", "radio")  # radio = radio up/downlink delays (Q5)
PURPOSES: Mapping[str, Kind] = {
    **dict.fromkeys(SEQUENTIAL, "sequential"),
    **dict.fromkeys(KEYED, "keyed"),
}

_MAX_PART = 2**64  # integer key parts are packed as unsigned 64-bit numbers
_UNIT = 2.0**-53  # one step of a 53-bit float mantissa


class SeedError(ValueError):
    """A run seed is not a whole number in the range its usage allows (fast_mode.seed_split)."""


def _digest(data: bytes, person: bytes) -> int:
    """64 bits of BLAKE2b, domain-separated by `person`."""
    return int.from_bytes(hashlib.blake2b(data, digest_size=8, person=person).digest(), "little")


def _check_part(part: object) -> KeyPart:
    if isinstance(part, bool) or not isinstance(part, int | str):
        raise TypeError(f"a key part must be a whole number or text, got {type(part).__name__} {part!r}")
    if isinstance(part, int) and not 0 <= part < _MAX_PART:
        raise TypeError(f"an integer key part must be in [0, 2**64), got {part}")
    return part


def _encode(parts: tuple[KeyPart, ...]) -> bytes:
    """Injective: a type tag per part, text with its length, so ("ab", "c") != ("a", "bc") != ("1", 2)."""
    out = bytearray()
    for part in parts:
        if isinstance(part, int):
            out += b"\x00" + struct.pack("<Q", part)
        else:
            text = part.encode("utf-8")
            out += b"\x01" + struct.pack("<I", len(text)) + text
    return bytes(out)


def _spawn_key(purpose: str, parts: tuple[KeyPart, ...]) -> tuple[int, ...]:
    """A SeedSequence spawn key: the purpose's name hash, then a (tag, value) pair per part."""
    key = [_digest(purpose.encode("utf-8"), b"pcnme-purpose")]
    for part in parts:
        if isinstance(part, int):
            key += [0, part]
        else:
            key += [1, _digest(part.encode("utf-8"), b"pcnme-keypart")]
    return tuple(key)


def _purpose(name: str, kind: Kind) -> str:
    if name not in PURPOSES:
        raise KeyError(f"unknown random purpose {name!r}; known: {', '.join(PURPOSES)}")
    if PURPOSES[name] != kind:
        use = "generator()" if PURPOSES[name] == "sequential" else "uniform()"
        raise ValueError(f"{name!r} is a {PURPOSES[name]} purpose: draw it with {use}")
    return name


class RandomStreams:
    """All random streams of one run, derived from its seed (final-design 15.2)."""

    def __init__(self, seed: int, usage: str, seed_split: Mapping[str, range]) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise SeedError(f"a run seed must be a whole number >= 0, got {type(seed).__name__} {seed!r}")
        if usage not in seed_split:
            raise SeedError(f"unknown seed usage {usage!r}; known: {', '.join(seed_split)}")
        allowed = seed_split[usage]
        if seed not in allowed:
            # 16.3 (Maus et al.): evaluation seeds never plan, demonstrate or train.
            raise SeedError(f"seed {seed} is outside the {usage} range {allowed.start}-{allowed.stop - 1}")
        self._seed = seed
        self._usage = usage
        self._seed_bytes = struct.pack("<Q", seed)
        self._generators: dict[tuple[str, tuple[KeyPart, ...]], np.random.Generator] = {}

    @classmethod
    def from_config(cls, config: Config, seed: int, usage: str) -> RandomStreams:
        return cls(seed, usage, config.fast_mode.seed_split)

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def usage(self) -> str:
        return self._usage

    def generator(self, purpose: str, *key: KeyPart) -> np.random.Generator:
        """The generator of a sequential purpose for one key (for example a car id). Asking again for the same
        purpose and key returns the same object, so the stream continues and never restarts."""
        parts = tuple(_check_part(p) for p in key)
        index = (_purpose(purpose, "sequential"), parts)
        if index not in self._generators:
            sequence = np.random.SeedSequence(self._seed, spawn_key=_spawn_key(purpose, parts))
            self._generators[index] = np.random.Generator(np.random.PCG64(sequence))
        return self._generators[index]

    def uniform(self, purpose: str, *key: KeyPart) -> float:
        """A keyed draw of a keyed purpose, uniform in the open interval (0, 1) (safe for an inverse CDF).

        A pure function of (seed, purpose, key): call order and other draws never change it (Q6).
        """
        name = _purpose(purpose, "keyed")
        if not key:
            raise ValueError(f"a keyed draw of {name!r} needs a key (task_id, attempt, ...)")
        parts = tuple(_check_part(p) for p in key)
        bits = _digest(self._seed_bytes + _encode(parts), name.encode("utf-8"))
        return ((bits >> 11) + 0.5) * _UNIT  # 53 bits, centred in their step: never 0, never 1
