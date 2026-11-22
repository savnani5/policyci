"""Dependency-free tests for the deterministic Panda pick-and-place controller."""

from __future__ import annotations

import importlib
import sys

import pytest

from policyci.examples.mujoco_grasp import pick_place
from policyci.examples.mujoco_grasp.adapter import (
    phase_action,
    position_delta,
    single_component,
)
from policyci.examples.mujoco_grasp.pick_place import pick_place_phase_specs


def test_offset_is_applied_as_a_faulty_measured_action_frame():
    delta = position_delta(
        actual_eef=(0.00, 0.00, 0.10),
        target=(0.10, 0.00, 0.10),
        action_frame_offset=(0.06, 0.00, 0.00),
        gain=10.0,
    )

    assert delta == pytest.approx((0.4, 0.0, 0.0))


def test_position_delta_clips_each_axis():
    assert position_delta(
        actual_eef=(0.0, 0.0, 0.0),
        target=(1.0, -1.0, 0.0),
        action_frame_offset=(0.0, 0.0, 0.0),
        gain=10.0,
    ) == (1.0, -1.0, 0.0)


def test_phase_action_uses_translation_zero_rotation_and_gripper_command():
    assert phase_action((0.2, -0.3, 0.4), gripper_command=1.0) == (
        0.2,
        -0.3,
        0.4,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    assert phase_action((0.2, -0.3, 0.4), gripper_command=-1.0)[-1] == -1.0
    assert phase_action((0.2, -0.3, 0.4), gripper_command=0.0)[-1] == 0.0


def test_pick_place_story_grasps_transports_releases_and_retreats():
    phases = pick_place_phase_specs(
        object_position=(0.10, -0.20, 0.84),
        target_position=(0.20, 0.40, 0.80),
    )

    assert sum(steps for _, steps, _, _ in phases) == 240
    assert phases == (
        ("approach", 35, (0.10, -0.20, 1.00), -1.0),
        ("align", 30, (0.10, -0.20, 0.86), -1.0),
        ("grasp", 25, (0.10, -0.20, 0.86), 1.0),
        ("lift", 30, (0.10, -0.20, 1.08), 1.0),
        ("transport", 50, (0.20, 0.40, 1.08), 1.0),
        ("place", 30, (0.20, 0.40, 0.90), 1.0),
        ("release", 15, (0.20, 0.40, 0.90), -1.0),
        ("retreat", 25, (0.20, 0.40, 1.08), -1.0),
    )


def test_object_pose_scenario_shifts_ground_truth_and_controller_reference():
    offset_object_qpos = getattr(pick_place, "offset_object_qpos", None)

    assert callable(offset_object_qpos)
    nominal_qpos = (0.10, -0.20, 0.84, 1.0, 0.0, 0.0, 0.0)
    shifted_qpos = offset_object_qpos(nominal_qpos, (0.05, 0.0, 0.0))
    phases = pick_place_phase_specs(
        object_position=shifted_qpos[:3],
        target_position=(0.20, 0.40, 0.80),
    )

    assert shifted_qpos == pytest.approx(
        (0.15, -0.20, 0.84, 1.0, 0.0, 0.0, 0.0)
    )
    assert phases[1][2] == pytest.approx((0.15, -0.20, 0.86))


def test_perception_bias_changes_controller_target_not_object_state():
    perceived_object_position = getattr(
        pick_place,
        "perceived_object_position",
        None,
    )

    assert callable(perceived_object_position)
    actual = (0.10, -0.20, 0.84)
    perceived = perceived_object_position(actual, (0.05, 0.0, 0.0))
    phases = pick_place_phase_specs(
        object_position=perceived,
        target_position=(0.20, 0.40, 0.80),
    )
    assert actual == (0.10, -0.20, 0.84)
    assert phases[1][2] == pytest.approx((0.15, -0.20, 0.86))

def test_model_identity_hashes_the_deterministic_task_specification():
    model_identity_sha256 = getattr(pick_place, "model_identity_sha256", None)

    assert callable(model_identity_sha256)
    baseline = model_identity_sha256({"body_parts": {"right": {"kp": 150}}, "damping": 1.0})
    reordered = model_identity_sha256({"damping": 1.0, "body_parts": {"right": {"kp": 150}}})
    changed = model_identity_sha256({"body_parts": {"right": {"kp": 120}}, "damping": 1.0})

    assert baseline == reordered
    assert baseline != changed
    assert len(baseline) == 64


def test_single_component_unwraps_panda_per_arm_mapping():
    right_gripper = object()

    assert single_component({"right": right_gripper}, name="gripper") is right_gripper

    with pytest.raises(ValueError, match="exactly one"):
        single_component({}, name="gripper")


def test_adapter_module_imports_without_simulation_dependencies(monkeypatch):
    monkeypatch.setitem(sys.modules, "robosuite", None)
    monkeypatch.setitem(sys.modules, "mujoco", None)

    module = importlib.reload(importlib.import_module("policyci.examples.mujoco_grasp.adapter"))

    assert module.RobosuiteLiftAdapter.__name__ == "RobosuiteLiftAdapter"
