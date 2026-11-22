"""Exact-state metric tests for the MuJoCo manipulation runner."""

from __future__ import annotations

import pytest

import policyci.mujoco_metrics as mujoco_metrics
from policyci.mujoco_metrics import (
    derive_lift_metrics,
    derive_placement_metrics,
    failure_window,
)
from policyci.schemas import SimulationSample, SimulationTrace


def make_trace(
    *,
    heights: list[float],
    contacts: list[bool],
    eef_x: list[float] | None = None,
) -> SimulationTrace:
    if len(heights) != len(contacts):
        raise ValueError("test fixture series must align")
    eef_x = eef_x or [0.10] * len(heights)
    samples = [
        SimulationSample(
            time_s=index * 0.05,
            object_position_m=(0.0, 0.0, 0.02 + height),
            object_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            eef_position_m=(eef_x[index], 0.0, 0.02 + height),
            eef_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            gripper_width_m=0.02,
            left_finger_contact=contact,
            right_finger_contact=contact,
        )
        for index, (height, contact) in enumerate(zip(heights, contacts))
    ]
    return SimulationTrace(
        sample_rate_hz=20.0,
        seed=1000,
        initial_state_sha256="a" * 64,
        model_sha256="b" * 64,
        samples=samples,
    )


def test_lift_requires_threshold_contact_and_continuous_hold():
    trace = make_trace(
        heights=[0.0] * 4 + [0.09] * 11,
        contacts=[False] * 4 + [True] * 11,
    )

    metrics = derive_lift_metrics(
        trace,
        lift_threshold_m=0.08,
        hold_duration_s=0.5,
    )

    assert metrics["lift_success"] is True
    assert metrics["max_object_lift_m"] == pytest.approx(0.09)
    assert metrics["bilateral_contact_duration_s"] == pytest.approx(0.5)
    assert metrics["min_eef_object_distance_m"] == pytest.approx(0.10)


def test_lift_rejects_height_without_bilateral_contact():
    trace = make_trace(
        heights=[0.0] + [0.12] * 12,
        contacts=[False] * 13,
    )

    metrics = derive_lift_metrics(
        trace,
        lift_threshold_m=0.08,
        hold_duration_s=0.5,
    )

    assert metrics["lift_success"] is False
    assert metrics["bilateral_contact_duration_s"] == 0.0


def test_lift_rejects_discontinuous_hold():
    trace = make_trace(
        heights=[0.0] + [0.09] * 6 + [0.0] + [0.09] * 6,
        contacts=[False] + [True] * 13,
    )

    metrics = derive_lift_metrics(
        trace,
        lift_threshold_m=0.08,
        hold_duration_s=0.5,
    )

    assert metrics["lift_success"] is False


def test_distance_uses_exact_three_dimensional_state():
    trace = make_trace(
        heights=[0.0, 0.0],
        contacts=[False, False],
        eef_x=[0.30, 0.04],
    )

    metrics = derive_lift_metrics(
        trace,
        lift_threshold_m=0.08,
        hold_duration_s=0.5,
    )

    assert metrics["min_eef_object_distance_m"] == pytest.approx(0.04)


@pytest.mark.parametrize(
    ("lift_threshold_m", "hold_duration_s"),
    [(0.0, 0.5), (-0.1, 0.5), (0.08, 0.0)],
)
def test_lift_metrics_reject_nonpositive_configuration(
    lift_threshold_m: float,
    hold_duration_s: float,
):
    trace = make_trace(heights=[0.0], contacts=[False])

    with pytest.raises(ValueError, match="positive"):
        derive_lift_metrics(
            trace,
            lift_threshold_m=lift_threshold_m,
            hold_duration_s=hold_duration_s,
        )


def test_failure_window_uses_first_failed_contact_attempt():
    trace = make_trace(
        heights=[0.0] * 9,
        contacts=[False, False, True, True, False, False, False, False, False],
    )

    assert failure_window(
        trace,
        lift_threshold_m=0.08,
        window_duration_s=0.2,
    ) == pytest.approx((0.1, 0.3))


def test_failure_window_falls_back_to_tail_without_contact():
    trace = make_trace(heights=[0.0] * 9, contacts=[False] * 9)

    assert failure_window(
        trace,
        lift_threshold_m=0.08,
        window_duration_s=0.2,
    ) == pytest.approx((0.2, 0.4))


def _placement_trace(*, success_indices: set[int]) -> SimulationTrace:
    samples = [
        SimulationSample(
            time_s=index * 0.05,
            object_position_m=(0.20, 0.40 - index * 0.02, 0.84),
            object_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            eef_position_m=(0.20, 0.40 - index * 0.02, 0.94),
            eef_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            gripper_width_m=0.04,
            left_finger_contact=index < 4,
            right_finger_contact=index < 4,
            target_position_m=(0.20, 0.20, 0.80),
            task_success=index in success_indices,
        )
        for index in range(15)
    ]
    return SimulationTrace(
        sample_rate_hz=20.0,
        seed=1000,
        initial_state_sha256="a" * 64,
        model_sha256="b" * 64,
        samples=samples,
    )


def test_placement_requires_sustained_exact_task_success():
    metrics = derive_placement_metrics(
        _placement_trace(success_indices=set(range(4, 15))),
        hold_duration_s=0.5,
    )

    assert metrics["placement_success"] is True
    assert metrics["placement_success_duration_s"] == pytest.approx(0.5)
    assert metrics["final_object_target_xy_distance_m"] == pytest.approx(0.08)


def test_placement_rejects_a_transient_target_entry():
    metrics = derive_placement_metrics(
        _placement_trace(success_indices=set(range(4, 9))),
        hold_duration_s=0.5,
    )

    assert metrics["placement_success"] is False


def test_pick_place_progress_advances_once_per_verified_milestone_and_never_regresses():
    states = [
        # object xyz, eef xyz, bilateral contact, task success
        ((0.0, 0.0, 0.80), (0.0, 0.0, 1.00), False, False),
        ((0.0, 0.0, 0.80), (0.03, 0.0, 0.82), False, False),
        ((0.0, 0.0, 0.80), (0.03, 0.0, 0.82), True, False),
        ((0.0, 0.0, 0.86), (0.03, 0.0, 0.88), True, False),
        ((0.95, 0.0, 0.86), (0.95, 0.0, 0.88), True, False),
        ((1.0, 0.0, 0.80), (1.0, 0.0, 0.90), False, True),
        ((1.0, 0.0, 0.80), (1.0, 0.0, 0.90), False, True),
        ((1.0, 0.0, 0.80), (1.0, 0.0, 0.90), False, True),
        ((0.0, 0.0, 0.80), (0.0, 0.0, 1.00), False, False),
    ]
    samples = [
        SimulationSample(
            time_s=index * 0.1,
            object_position_m=object_position,
            object_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            eef_position_m=eef_position,
            eef_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            gripper_width_m=0.04,
            left_finger_contact=bilateral_contact,
            right_finger_contact=bilateral_contact,
            target_position_m=(1.0, 0.0, 0.80),
            task_success=task_success,
        )
        for index, (
            object_position,
            eef_position,
            bilateral_contact,
            task_success,
        ) in enumerate(states)
    ]
    trace = SimulationTrace(
        sample_rate_hz=10.0,
        seed=1000,
        initial_state_sha256="a" * 64,
        model_sha256="b" * 64,
        samples=samples,
    )

    assert mujoco_metrics.derive_pick_place_progress(
        trace,
        placement_hold_duration_s=0.2,
    ) == [0.0, 20.0, 40.0, 60.0, 80.0, 80.0, 80.0, 100.0, 100.0]
