"""Intake and the anchor deadline (PLAN M2.4; final-design 6A)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcnme.car.car import Task
from pcnme.config import Config, load_config
from pcnme.ids import TaskId
from pcnme.net.radio import RadioTransfer, RadioUplink, shannon_rate
from pcnme.sim.clock import EventClock
from pcnme.zone.intake import AnchorTable, anchor_deadline


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


def task(cfg: Config, seq: int = 1, kind: str = "T1", created: float = 0.0) -> Task:
    t = cfg.task_types[kind]
    return Task(
        TaskId("c01", 0, seq),
        "c01",
        kind,
        t.cls,
        t.deadline["loose"],
        t.l_max["loose"],
        t.input_bits,
        t.result_bits,
        created,
    )


def test_the_anchor_deadline_by_hand(cfg: Config) -> None:
    """6A.3: d_A = t_rx + B_car - Û_up, Û_up = W_radio + S / R_car + δ̂_radio.

    A T1 frame (1.2 Mbit) from a car 300 m away (R_car = 539.1 Mbps): push = 1.2e6 / 539.1e6 = 2.226 ms. It
    waited W_radio = 0.5 ms behind another upload; δ̂_radio = mean of U(1, 5) ms = 3 ms; B_car = 100 ms
    (sent at creation). The whole upload arrived at t_rx = 0.5 + 2.226 + 3.4 (its real delay) = 6.126 ms.
    d_A = 6.126 + 100 - (0.5 + 2.226 + 3) = 100.4 ms: 0.4 ms later than the car's own deadline, because the
    real radio delay (3.4 ms) was 0.4 ms longer than its mean.
    """
    rate = shannon_rate(300.0, cfg.radio)
    push = 1.2e6 / rate
    t_rx = 0.0005 + push + 0.0034
    d_a = anchor_deadline(
        t_rx=t_rx,
        b_car=0.100,
        w_radio=0.0005,
        size_bits=1.2e6,
        rate=rate,
        mean_radio_delay=sum(cfg.radio.uplink_delay) / 2,
    )
    assert d_a == pytest.approx(t_rx + 0.100 - (0.0005 + push + 0.003), rel=1e-12)
    assert d_a == pytest.approx(0.1004, rel=1e-9)


def test_with_the_mean_delay_the_anchor_deadline_is_the_cars_deadline(cfg: Config) -> None:
    # When the radio delay equals its mean, d_A = creation + D on the anchor's clock, for any wait and rate,
    # here with a T2 clip sharing the air (CHANGES C6).
    clock = EventClock()
    up = RadioUplink(clock)
    rate = shannon_rate(120.0, cfg.radio)
    got: list[RadioTransfer] = []
    clock.schedule(0.0, lambda: up.send(72e6, rate, 0.003, lambda tr: None))  # a T2 clip ahead of it
    clock.schedule(0.001, lambda: up.send(1.2e6, rate, 0.003, got.append))
    clock.run_until(1.0)
    (tr,) = got
    assert tr.wait > 0.001  # it shared the air with the T2
    d_a = anchor_deadline(
        t_rx=tr.delivery, b_car=0.100, w_radio=tr.wait, size_bits=1.2e6, rate=rate, mean_radio_delay=0.003
    )
    assert d_a == pytest.approx(0.001 + 0.100, rel=1e-12)


def test_the_anchor_table_opens_and_closes_each_task_exactly_once(cfg: Config) -> None:
    table = AnchorTable("z1")
    a = table.open(task(cfg, 1), t_rx=0.006, d_a=0.1004)
    assert table.is_open(a.task.task_id) and a.d_a == 0.1004 and a.anchor_id == "z1"
    table.close(a.task.task_id, outcome="on_time", now=0.05)
    assert not table.is_open(a.task.task_id)
    assert table.closed[a.task.task_id] == ("on_time", 0.05)
    with pytest.raises(ValueError, match="already closed"):
        table.close(a.task.task_id, outcome="drop", now=0.06)
    with pytest.raises(ValueError, match="already"):
        table.open(task(cfg, 1), t_rx=0.2, d_a=0.3)  # a second row for the same task id


def test_attempt_numbers_count_from_one(cfg: Config) -> None:
    table = AnchorTable("z1")
    row = table.open(task(cfg, 7), t_rx=0.0, d_a=0.1)
    assert table.next_attempt(row.task.task_id) == 1
    assert table.next_attempt(row.task.task_id) == 2
