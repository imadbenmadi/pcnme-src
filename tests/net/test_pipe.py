"""FIFO pipe (PLAN M1.2; final-design 4.1, 4.3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pcnme.net.pipe import FifoPipe

B = 1e9  # bit/s
S = 1e6  # bits: 1 ms push at 1 Gbps


# --- R1-R4: hand-computed sequence of three transfers (PLAN M1.2) -----------------------------------------


def test_three_transfers_by_hand() -> None:
    pipe = FifoPipe(B)

    # 1. Empty pipe at t = 0: no wait, push 1 ms, delay 3 ms -> delivered at 4 ms.
    a = pipe.send(now=0.0, size_bits=S, delay=0.003)
    assert (a.wait, a.push, a.push_end, a.delivery) == pytest.approx((0.0, 0.001, 0.001, 0.004))

    # 2. Overlapping, at 0.5 ms: waits for a's push only (0.5 ms), pushes until 2 ms, delay 1 ms -> 3 ms.
    #    The delay does not hold the pipe, so b is delivered before a (3 ms < 4 ms).
    b = pipe.send(now=0.0005, size_bits=S, delay=0.001)
    assert (b.wait, b.push_end, b.delivery) == pytest.approx((0.0005, 0.002, 0.003))
    assert b.delivery < a.delivery

    # 3. Not overlapping, at 5 ms: the pipe has been free since 2 ms -> no wait; delivered at 6 + 2 = 8 ms.
    c = pipe.send(now=0.005, size_bits=S, delay=0.002)
    assert (c.wait, c.push_end, c.delivery) == pytest.approx((0.0, 0.006, 0.008))


def test_transfer_time_is_wait_plus_push_plus_delay() -> None:
    # final-design 4.1: T = W_pipe + S / B + d
    pipe = FifoPipe(B)
    pipe.send(now=0.0, size_bits=3 * S, delay=0.0)  # busy until 3 ms
    t = pipe.send(now=0.001, size_bits=2 * S, delay=0.004)
    assert t.wait == pytest.approx(0.002)
    assert t.push == pytest.approx(0.002)
    assert t.duration == pytest.approx(t.wait + t.push + t.delay) == pytest.approx(0.008)
    assert t.delivery == pytest.approx(t.start + t.duration)


def test_push_time_is_size_over_bandwidth_on_the_wan() -> None:
    # A 150 KB frame (1.2 Mbit) on the 515 Mbps WAN pushes in 1.2e6 / 515e6 = 2.330 ms; a 9 MB T2 clip in
    # 72e6 / 515e6 = 139.8 ms. A second bandwidth guards against a formula that only works at 1 Gbps.
    wan = FifoPipe(515e6)
    frame = wan.send(now=0.0, size_bits=150 * 8000, delay=0.0)
    clip = wan.send(now=0.0, size_bits=9000 * 8000, delay=0.0)
    assert frame.push == pytest.approx(1.2e6 / 515e6, rel=1e-12)
    assert clip.wait == pytest.approx(1.2e6 / 515e6, rel=1e-12)
    assert clip.push == pytest.approx(0.139806, rel=1e-5)


def test_a_long_delay_does_not_hold_the_pipe() -> None:
    # R4: the second transfer waits for the first one's push, never for its delay.
    slow, fast = FifoPipe(B), FifoPipe(B)
    slow.send(now=0.0, size_bits=S, delay=10.0)
    fast.send(now=0.0, size_bits=S, delay=0.0)
    assert slow.send(now=0.0, size_bits=S, delay=0.0).wait == fast.send(now=0.0, size_bits=S, delay=0.0).wait


def test_back_to_back_transfers_queue_their_pushes() -> None:
    pipe = FifoPipe(B)
    waits = [pipe.send(now=0.0, size_bits=S, delay=0.0).wait for _ in range(5)]
    assert waits == pytest.approx([0.0, 0.001, 0.002, 0.003, 0.004])
    assert pipe.t_free == pytest.approx(0.005)


# --- R5: the current wait is known exactly (final-design 4.3) ---------------------------------------------


def test_wait_at_reads_the_wait_without_sending() -> None:
    pipe = FifoPipe(B)
    pipe.send(now=0.0, size_bits=4 * S, delay=0.0)  # busy until 4 ms
    assert pipe.wait_at(0.001) == pytest.approx(0.003)
    assert pipe.wait_at(0.010) == 0.0
    assert pipe.send(now=0.001, size_bits=S, delay=0.0).wait == pytest.approx(0.003)


def test_a_fresh_pipe_has_no_wait() -> None:
    assert FifoPipe(B).wait_at(0.0) == 0.0


# --- Traffic intensity (final-design 2: TI below 1 keeps up, above 1 grows without limit) -------------------


def poisson_run(ti: float, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """n fixed-size transfers with Poisson arrivals at traffic intensity ti: (arrival times, waits)."""
    push = S / B
    rate = ti / push  # transfers per second, so bits per second / B = ti
    arrivals = np.cumsum(np.random.default_rng(seed).exponential(1 / rate, n))
    pipe = FifoPipe(B)
    waits = np.array([pipe.send(now=float(t), size_bits=S, delay=0.0).wait for t in arrivals])
    return arrivals, waits


@pytest.mark.parametrize("ti", [0.5, 0.8])
def test_below_one_the_wait_is_bounded_and_matches_m_d_1(ti: float) -> None:
    # Poisson arrivals and one fixed size make the pipe an M/D/1 queue: mean wait = rho D / (2 (1 - rho)).
    _, waits = poisson_run(ti, 200_000, seed=2000)
    push = S / B
    expected = ti * push / (2 * (1 - ti))
    assert waits.mean() == pytest.approx(expected, rel=0.03)
    # Bounded: the last tenth waits no longer on average than the run as a whole (within noise).
    assert waits[-20_000:].mean() < 1.2 * waits.mean()


@pytest.mark.parametrize("ti", [1.5, 2.0])
def test_above_one_the_wait_grows_linearly_with_slope_ti_minus_one(ti: float) -> None:
    # Work arrives at ti seconds of push per second and leaves at 1, so the backlog grows as (ti - 1) t.
    arrivals, waits = poisson_run(ti, 50_000, seed=2001)
    slope, _ = np.polyfit(arrivals, waits, 1)
    assert slope == pytest.approx(ti - 1, rel=0.03)
    # Linear, not just growing: the fit explains nearly all of the wait.
    residual = waits - np.polyval(np.polyfit(arrivals, waits, 1), arrivals)
    assert residual.std() < 0.01 * waits.max()


def test_exactly_periodic_overload_by_hand() -> None:
    # Arrivals every 1 ms, push 1.5 ms (TI = 1.5): the k-th transfer waits exactly 0.5 k ms.
    pipe = FifoPipe(B)
    waits = [pipe.send(now=k * 0.001, size_bits=1.5 * S, delay=0.0).wait for k in range(10)]
    assert waits == pytest.approx([0.0005 * k for k in range(10)])


# --- R7: refused ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_bad_bandwidth_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="bandwidth"):
        FifoPipe(bad)


@pytest.mark.parametrize(
    ("size", "delay", "match"),
    [
        (0.0, 0.0, "size"),
        (-1.0, 0.0, "size"),
        (math.inf, 0.0, "size"),
        (S, -0.001, "delay"),
        (S, math.nan, "delay"),
    ],
)
def test_bad_transfer_is_refused(size: float, delay: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        FifoPipe(B).send(now=0.0, size_bits=size, delay=delay)


def test_time_cannot_go_backwards() -> None:
    # FIFO: transfers enter the pipe in time order.
    pipe = FifoPipe(B)
    pipe.send(now=1.0, size_bits=S, delay=0.0)
    with pytest.raises(ValueError, match="earlier"):
        pipe.send(now=0.5, size_bits=S, delay=0.0)
    with pytest.raises(ValueError, match="earlier"):
        pipe.wait_at(0.5)
    with pytest.raises(ValueError, match="finite"):
        pipe.send(now=math.nan, size_bits=S, delay=0.0)


def test_equal_times_are_allowed() -> None:
    pipe = FifoPipe(B)
    pipe.send(now=1.0, size_bits=S, delay=0.0)
    assert pipe.send(now=1.0, size_bits=S, delay=0.0).wait == pytest.approx(0.001)
