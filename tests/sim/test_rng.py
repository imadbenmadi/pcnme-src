"""Seeded random streams (PLAN M0.3; final-design 15.2, 13.2; QUESTIONS Q4-Q6)."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats

from pcnme.config import load_config
from pcnme.sim.rng import KEYED, PURPOSES, SEQUENTIAL, RandomStreams, SeedError

SPLIT = {
    "nsga2_and_demos": range(1000, 2000),
    "training": range(2000, 3000),
    "evaluation": range(3000, 4000),
}


def streams(seed: int = 2000, usage: str = "training") -> RandomStreams:
    return RandomStreams(seed, usage, SPLIT)


# --- R1: the purposes ---------------------------------------------------------------------------------------


def test_the_six_purposes_of_plan_m0_3() -> None:
    # Q6: these never depend on decisions; random_method (the Random baseline, 16.1) added by name in M2.8
    # (Q4)
    assert set(SEQUENTIAL) == {"arrivals", "mobility", "nsga", "random_method"}
    assert set(KEYED) == {"run_times", "link_delays", "radio"}  # Q5: radio = radio up/downlink delays
    assert set(PURPOSES) == set(SEQUENTIAL) | set(KEYED)


def test_keyed_purpose_names_fit_the_hash_domain_tag() -> None:
    # A keyed purpose's name is its BLAKE2b "person" string, which holds at most 16 bytes.
    assert all(len(p.encode("utf-8")) <= 16 for p in KEYED)


def test_unknown_purpose_is_refused() -> None:
    with pytest.raises(KeyError, match="speeds"):
        streams().generator("speeds", 1)
    with pytest.raises(KeyError, match="delays"):
        streams().uniform("delays", 1)


def test_a_keyed_purpose_cannot_be_drawn_in_sequence_and_back() -> None:
    # R6 (Q6): a run time drawn in event order would break common random numbers.
    with pytest.raises(ValueError, match="keyed"):
        streams().generator("run_times", 1)
    with pytest.raises(ValueError, match="sequential"):
        streams().uniform("arrivals", 1)


# --- R3: same seed, same draws; independence ----------------------------------------------------------------


def test_same_seed_gives_identical_sequential_draws() -> None:
    a = streams().generator("arrivals", 7).random(1000)
    b = streams().generator("arrivals", 7).random(1000)
    assert np.array_equal(a, b)


def test_same_seed_gives_identical_keyed_draws() -> None:
    keys = [(car, 0, seq, attempt) for car, seq, attempt in itertools.product(range(5), range(20), (1, 2))]
    assert [streams().uniform("run_times", *k) for k in keys] == [
        streams().uniform("run_times", *k) for k in keys
    ]


def test_different_seeds_give_different_draws() -> None:
    assert not np.array_equal(
        streams(2000).generator("arrivals", 7).random(100), streams(2001).generator("arrivals", 7).random(100)
    )
    assert streams(2000).uniform("run_times", 7, 0, 1, 1) != streams(2001).uniform("run_times", 7, 0, 1, 1)


def test_different_purposes_are_independent() -> None:
    s = streams()
    sequences = {p: s.generator(p, 7).random(20_000) for p in SEQUENTIAL}
    for p, q in itertools.combinations(sequences, 2):
        r = np.corrcoef(sequences[p], sequences[q])[0, 1]
        assert abs(r) < 0.03, (p, q, r)  # |r| ~ N(0, 1/sqrt(n)) = 0.007 when independent
    keyed = {p: np.array([s.uniform(p, i, 0, 1, 1) for i in range(20_000)]) for p in KEYED}
    for p, q in itertools.combinations(keyed, 2):
        r = np.corrcoef(keyed[p], keyed[q])[0, 1]
        assert abs(r) < 0.03, (p, q, r)


def test_different_keys_of_one_purpose_are_independent() -> None:
    s = streams()
    a, b = s.generator("mobility", 1).random(20_000), s.generator("mobility", 2).random(20_000)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.03


def test_drawing_one_purpose_does_not_move_another() -> None:
    # Adding draws of one kind (for example a new radio delay) must not change the arrivals.
    quiet = streams().generator("arrivals", 3).random(50)
    busy = streams()
    arrivals = busy.generator("arrivals", 3)
    first = arrivals.random(25)
    busy.generator("mobility", 3).random(1000)
    busy.generator("nsga", 0).random(1000)
    [busy.uniform("radio", i, 0, 1, 1, 0) for i in range(100)]
    assert np.array_equal(np.concatenate([first, arrivals.random(25)]), quiet)


def test_the_same_sequential_key_gives_the_same_generator_object() -> None:
    # Asking twice must continue the stream, never restart it (a restart would repeat draws).
    s = streams()
    g = s.generator("arrivals", 3)
    g.random(10)
    assert s.generator("arrivals", 3) is g


# --- R2: streams follow the purpose's name, not its position ------------------------------------------------


def test_golden_values_pin_the_streams() -> None:
    # Regression pin: these exact values must never change, or old runs stop being reproducible. Same on
    # Linux and Windows (CI runs both). Change them only on purpose, in a PR that says why.
    s = streams(2000)
    assert s.generator("arrivals", 1).random(3).tolist() == GOLDEN_ARRIVALS
    assert s.uniform("run_times", 1, 0, 1, 1) == GOLDEN_RUN_TIME
    assert s.uniform("link_delays", 1, 0, 1, 1, "bh") == GOLDEN_LINK_DELAY


# --- R4: keyed draws do not depend on order (common random numbers, 13.2) -----------------------------------


def test_keyed_draws_do_not_depend_on_call_order() -> None:
    keys = [(car, 0, seq, 1) for car in range(4) for seq in range(25)]
    forward = {k: streams().uniform("run_times", *k) for k in keys}
    s = streams()
    backward = {k: s.uniform("run_times", *k) for k in reversed(keys)}
    s.generator("arrivals", 0).random(500)  # unrelated draws in between change nothing
    assert forward == backward


def test_the_attempt_number_changes_the_draw() -> None:
    # A retry is a new attempt (Q6: the attempt number is the one decision-dependent part of the key).
    s = streams()
    assert s.uniform("run_times", 4, 0, 9, 1) != s.uniform("run_times", 4, 0, 9, 2)


def test_each_hop_of_a_transfer_gets_its_own_delay() -> None:
    # Q6: delays are keyed by (task_id, attempt, link/hop); a cloud transfer has two hops (4.1).
    s = streams()
    assert s.uniform("link_delays", 4, 0, 9, 1, "bh") != s.uniform("link_delays", 4, 0, 9, 1, "wan")


def test_key_parts_are_unambiguous() -> None:
    # A text part never collides with an integer part, and splitting differently gives different keys.
    s = streams()
    assert s.uniform("radio", 1, 2) != s.uniform("radio", 12)
    assert s.uniform("radio", "1", 2) != s.uniform("radio", 1, 2)
    assert s.uniform("radio", "ab", "c") != s.uniform("radio", "a", "bc")
    assert s.uniform("radio", 1, 2) != s.uniform("radio", 1, 2, 0)


@pytest.mark.parametrize("bad", [-1, True, 1.5, None, b"x"])
def test_bad_key_parts_are_refused(bad: Any) -> None:
    with pytest.raises(TypeError):
        streams().uniform("radio", 1, bad)
    with pytest.raises(TypeError):
        streams().generator("arrivals", bad)


def test_an_empty_key_is_refused_for_keyed_draws() -> None:
    with pytest.raises(ValueError, match="key"):
        streams().uniform("run_times")


# --- R5: keyed uniforms lie in (0, 1) and are uniform -------------------------------------------------------


def test_keyed_uniforms_are_uniform_in_the_open_interval() -> None:
    s = streams()
    u = np.array([s.uniform("run_times", car, 0, seq, 1) for car in range(20) for seq in range(1000)])
    assert u.min() > 0.0 and u.max() < 1.0
    assert stats.kstest(u, "uniform").pvalue > 1e-3  # fixed seed: deterministic, not flaky


# --- R7: seed ranges (fast_mode.seed_split; final-design 16.3) ----------------------------------------------


@pytest.mark.parametrize(
    ("seed", "usage"),
    [(1000, "nsga2_and_demos"), (1999, "nsga2_and_demos"), (2000, "training"), (3999, "evaluation")],
)
def test_a_seed_inside_its_range_is_accepted(seed: int, usage: str) -> None:
    s = RandomStreams(seed, usage, SPLIT)
    assert (s.seed, s.usage) == (seed, usage)


@pytest.mark.parametrize(
    ("seed", "usage"), [(1999, "training"), (3000, "training"), (2500, "evaluation"), (0, "nsga2_and_demos")]
)
def test_a_seed_outside_its_range_is_refused(seed: int, usage: str) -> None:
    with pytest.raises(SeedError, match=usage):
        RandomStreams(seed, usage, SPLIT)


def test_an_unknown_usage_is_refused() -> None:
    with pytest.raises(SeedError, match="testing"):
        RandomStreams(2000, "testing", SPLIT)


@pytest.mark.parametrize("bad", [True, 2000.0, "2000", -5])
def test_a_seed_must_be_a_whole_number(bad: Any) -> None:
    with pytest.raises(SeedError):
        RandomStreams(bad, "training", SPLIT)


def test_seed_ranges_come_from_the_config(default_config_path: Path) -> None:
    cfg = load_config(default_config_path)
    assert RandomStreams.from_config(cfg, 3000, "evaluation").seed == 3000
    with pytest.raises(SeedError):
        RandomStreams.from_config(cfg, 3000, "training")


# Pinned from the first run of the implementation on 28 Oct (Windows); CI checks them on Linux too.
GOLDEN_ARRIVALS = [0.6067399886828576, 0.31036798823937783, 0.8483922636061795]
GOLDEN_RUN_TIME = 0.39069401835531486
GOLDEN_LINK_DELAY = 0.4548882856058292
