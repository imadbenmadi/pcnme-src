"""Identifiers (PLAN M2.1; final-design 5.5)."""

from __future__ import annotations

import pytest

from pcnme.ids import AttemptKey, SnapshotCounter, SnapshotVersion, TaskId, TaskIds
from pcnme.node.executor import AttemptKey as ExecutorAttemptKey


def test_a_task_id_is_car_epoch_seq_and_prints_as_in_the_design() -> None:
    t = TaskId("c03-047", 2, 1187)
    assert (t.car_id, t.epoch, t.seq) == ("c03-047", 2, 1187)
    assert str(t) == "c03-047.2.1187"  # 5.5


def test_task_ids_order_as_the_tuple_car_epoch_seq() -> None:
    # 8.2's last tie rule compares task ids as (car_id, epoch, seq).
    ids = [TaskId("c02", 0, 10), TaskId("c01", 1, 1), TaskId("c01", 0, 7), TaskId("c01", 0, 3)]
    assert sorted(ids) == [
        TaskId("c01", 0, 3),
        TaskId("c01", 0, 7),
        TaskId("c01", 1, 1),
        TaskId("c02", 0, 10),
    ]


def test_a_car_numbers_its_tasks_and_a_restart_raises_the_epoch() -> None:
    ids = TaskIds("c03-047")  # fast mode: epoch 0
    assert [ids.next() for _ in range(3)] == [TaskId("c03-047", 0, s) for s in (1, 2, 3)]
    after_restart = TaskIds("c03-047", epoch=1)
    assert after_restart.next() == TaskId("c03-047", 1, 1)
    assert after_restart.next() not in {TaskId("c03-047", 0, s) for s in (1, 2, 3)}  # never reused (5.5)


def test_a_task_id_is_a_valid_random_stream_key() -> None:
    from pcnme.sim.rng import RandomStreams

    s = RandomStreams(2000, "training", {"training": range(2000, 3000)})
    assert s.uniform("run_times", *TaskId("c01", 0, 5), 1) == s.uniform("run_times", "c01", 0, 5, 1)


def test_the_attempt_key_carries_the_anchor_and_is_one_type_everywhere() -> None:
    t = TaskId("c01", 0, 9)
    assert AttemptKey(t, "z1", 1) != AttemptKey(t, "z2", 1)  # two anchors of one task (9.7)
    assert AttemptKey(t, "z1", 1) != AttemptKey(t, "z1", 2)
    assert ExecutorAttemptKey is AttemptKey  # no second definition


@pytest.mark.parametrize("bad", [0, -1])
def test_attempts_count_from_one(bad: int) -> None:
    with pytest.raises(ValueError, match="attempt"):
        AttemptKey(TaskId("c01", 0, 1), "z1", bad)


def test_snapshot_versions_compare_epoch_first() -> None:
    old_epoch_high = SnapshotVersion(epoch=0, counter=9_999)
    new_epoch_low = SnapshotVersion(epoch=1, counter=0)
    assert new_epoch_low > old_epoch_high  # a restarted zone's new snapshots beat its old ones (5.5)
    assert SnapshotVersion(1, 5) > SnapshotVersion(1, 4)


def test_a_snapshot_counter_grows_on_every_snapshot() -> None:
    c = SnapshotCounter(epoch=0)
    assert [c.next() for _ in range(3)] == [
        SnapshotVersion(0, 1),
        SnapshotVersion(0, 2),
        SnapshotVersion(0, 3),
    ]
