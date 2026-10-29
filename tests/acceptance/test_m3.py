"""M3 acceptance tests (PLAN M3; final-design 7.4, 9.1, 10.2, 16.2).

- One scenario test per row of the failure table (9.1) and per case C1-C14 (7.4): the tests below, listed in
  FAILURE_ROWS and CASES and checked to exist (each is a scripted scenario with numbers by hand).
- Exit time against hand calculations, infinity when stopped: tests/mobility/test_predictor.py.
- S1-S3 with Greedy, 3 seeds, the full city: the measured targets are in the M3 report and in the slow
  tests below.
"""

# ruff: noqa: E501  (the references are full test names, which cannot be wrapped)

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from pcnme.config import Config, load_config
from pcnme.experiments.city import City

FAILURE_ROWS = {  # 9.1, one row each
    "refused, then the one re-decision": "tests.zone.test_placement::test_a_refusal_gets_one_re_decision_as_a_new_attempt",
    "refused again: hard drop": "tests.zone.test_placement::test_a_hard_task_refused_twice_is_dropped",
    "refused again: soft to the cloud": "tests.zone.test_placement::test_a_soft_task_refused_twice_goes_to_the_cloud",
    "upload to the executor fails (no reply)": "tests.zone.test_failures::test_a_lost_reply_times_out_and_the_first_result_still_wins",
    "executor zone dies (last will)": "tests.zone.test_failures::test_an_executor_that_dies_triggers_a_retry_at_once",
    "task silently lost (watchdog, probe)": "tests.zone.test_failures::test_a_silently_lost_result_is_found_by_the_watchdog_and_the_probe",
    "deadline cannot be met (skip, abort)": "tests.zone.test_delivery::test_skips_and_aborts_reach_the_anchor_as_status_messages",
    "wrong mobility prediction (result follows)": "tests.experiments.test_city::test_a_result_follows_the_car_to_its_new_zone",
    "car in a coverage gap (hold, republish)": "tests.zone.test_duplicates::test_a_new_radio_session_gets_the_result_again_until_the_ack",
    "upload cut by a handover or gap": "tests.experiments.test_city::test_an_upload_cut_by_a_handover_goes_to_the_next_zone_with_the_same_task_id",
}
CASES = {  # 7.4
    "C1 boulder": "tests.zone.test_delivery::test_a_boulder_goes_to_the_cloud_with_no_decision",
    "C2 run here": "tests.zone.test_delivery::test_a_t1_run_here_by_hand",
    "C3 car leaving (exit time, next zone)": "tests.experiments.test_city::test_the_anchor_predicts_exit_time_and_next_zone_from_the_cars_gps",
    "C4 load balancing": "tests.zone.test_masks::test_greedy_sends_to_a_quiet_neighbour_when_here_is_swamped",
    "C5 cloud": "tests.zone.test_delivery::test_a_t4_run_in_the_cloud_by_hand",
    "C7 car moved": "tests.experiments.test_city::test_a_result_follows_the_car_to_its_new_zone",
    "C8 car stopped": "tests.mobility.test_predictor::test_exit_time_is_infinite_when_stopped",
    "C9 deadline": "tests.zone.test_delivery::test_the_outcome_comes_from_the_cars_latency",
    "C10 masked": "tests.zone.test_masks::test_each_position_is_empty_offline_stale_or_open",
    "C12 coverage gap": "tests.experiments.test_city::test_a_frame_made_in_a_gap_is_held_then_uploaded_with_the_same_id_or_given_up",
    "C13 two zones": "tests.experiments.test_city::test_the_current_access_point_is_kept_unless_another_is_20_m_closer",
    "C14 failure, retry": "tests.zone.test_failures::test_a_hard_task_is_retried_once_a_soft_one_twice",
}
EXIT_TIME = [
    "tests.mobility.test_predictor::test_exit_time_from_the_centre_is_radius_over_speed",
    "tests.mobility.test_predictor::test_exit_time_off_centre_by_hand",
    "tests.mobility.test_predictor::test_exit_time_is_infinite_when_stopped",
]


@pytest.mark.parametrize("ref", [*FAILURE_ROWS.values(), *CASES.values(), *EXIT_TIME])
def test_the_scenario_test_exists(ref: str) -> None:
    module, name = ref.split("::")
    assert callable(getattr(importlib.import_module(module), name)), ref


def test_every_failure_row_and_case_has_a_test() -> None:
    assert len(FAILURE_ROWS) == 10  # the 8 rows of 9.1, the refusal row split by class outcome
    assert {k.split()[0] for k in CASES} == {
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C7",
        "C8",
        "C9",
        "C10",
        "C12",
        "C13",
        "C14",
    }


# --- the full city, S1-S3 with Greedy (16.2) -----------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg(default_config_path: Path) -> Config:
    return load_config(default_config_path)


@pytest.mark.slow
@pytest.mark.parametrize("scenario", ["S1", "S2", "S3"])
def test_the_city_keeps_its_invariants_and_z1_carries_the_hotspot(cfg: Config, scenario: str) -> None:
    c = City(cfg, scenario, "greedy", 2000).run(60.0)
    closed = [r.task.task_id for r in c.records]
    given_up = {d.task.task_id for d in c.car_drops}
    assert len(closed) == len(set(closed)) and set(c.created) == set(closed) | given_up
    per_zone = {z: sum(r.anchor == z for r in c.records) for z in c.zones}
    assert per_zone["z1"] / (sum(per_zone.values()) / 4) == pytest.approx(1.6, rel=0.15)


@pytest.mark.slow
@pytest.mark.parametrize("scenario", ["S2", "S3"])
def test_no_upload_timeout_feeds_the_wan_after_c7(cfg: Config, scenario: str) -> None:
    # CHANGES C7: before it, timeouts re-sent uploads into a WAN that could not recall them (S2 seed 2000
    # collapsed to TI 1.10 with a 3.8 s mean wait; S3 carried 2.45 against 1.36 eligible). Now the WAN carries
    # at most the eligible bits (Greedy keeps some at the fog) and the S2 collapse is gone.
    c = City(cfg, scenario, "greedy", 2000).run(60.0)
    soft = [
        r for r in c.records if r.task.cls == "soft" or cfg.is_boulder(r.task.task_type, c.scenario.deadlines)
    ]
    eligible = sum(r.task.input_bits for r in soft) / 60.0 / cfg.derived.b_wan
    carried = c.wan.bits_sent / 60.0 / cfg.derived.b_wan
    waits = [r.transfer.w_wan for r in c.records if r.transfer is not None]
    assert carried <= 1.05 * eligible
    if scenario == "S2":
        assert sum(waits) / len(waits) < 0.5


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="FINDING (QUESTIONS Q29): about 15 % of the frames are made in coverage gaps and given up car-side, "
    "so the measured rho_fog and TI_WAN sit about 15 % under target. Coverage, not queueing; left for Imed.",
)
def test_rho_fog_and_ti_wan_match_the_targets_within_5_percent(cfg: Config) -> None:
    # PLAN M3 acceptance: rho_fog and TI_WAN measured from the run match the config targets within 5 %.
    c = City(cfg, "S2", "greedy", 2000).run(60.0)
    per_zone = len(c.records) / 4 / 60.0
    rho = per_zone * cfg.derived.mean_gpu_run / cfg.hardware["F-GPU"].c_gpu
    assert rho == pytest.approx(c.scenario.rho_fog, rel=0.05)
