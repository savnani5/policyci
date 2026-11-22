"""Pure loading and evidence validation for robustness dashboards."""

from __future__ import annotations

from pathlib import Path

import pytest

from policyci.robustness_visualization import load_robustness_visualization
from policyci.schemas import (
    ArtifactRef,
    ExecutionValidity,
    GateResult,
    GateVerdict,
    RobustnessCaseResult,
    RobustnessCheckResult,
    SimulationSample,
    SimulationTrace,
    SubjectRef,
)


def _trace(
    *,
    offset_x: float,
    succeeds: bool,
    initial_hash: str,
    slips: bool = False,
) -> SimulationTrace:
    samples = []
    subtasks = (
        "approach",
        "align",
        "grasp",
        "lift",
        "transport",
        "place",
        "release",
        "retreat",
    )
    for index in range(13):
        success = succeeds and index >= 2
        if succeeds and index >= 5:
            object_position = (0.20, 0.40, 0.80)
        elif succeeds and index == 4:
            object_position = (0.20, 0.40, 0.90)
        elif succeeds and index == 3:
            object_position = (offset_x, 0.0, 0.90)
        elif slips and 4 <= index <= 8:
            object_position = (0.0, 0.0, 0.84 + 0.04 * (index - 3))
        else:
            object_position = (offset_x, 0.0, 0.84)
        eef_position = (
            (offset_x + 0.03, 0.0, object_position[2] + 0.02)
            if index >= 1
            else (0.10, 0.0, 0.95)
        )
        bilateral_contact = succeeds and index in {2, 3, 4}
        samples.append(
            SimulationSample(
                time_s=index * 0.05,
                object_position_m=object_position,
                object_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                eef_position_m=eef_position,
                eef_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                gripper_width_m=0.05,
                left_finger_contact=bilateral_contact,
                right_finger_contact=bilateral_contact,
                target_position_m=(0.20, 0.40, 0.80),
                task_success=success,
                subtask=subtasks[min(index * len(subtasks) // 13, 7)],
            )
        )
    return SimulationTrace(
        sample_rate_hz=20.0,
        seed=1000,
        initial_state_sha256=initial_hash * 64,
        model_sha256="c" * 64,
        samples=samples,
    )


def _case(
    root: Path,
    *,
    case_id: str,
    offset_x: float,
    succeeds: bool,
    initial_hash: str,
    observation_bias_x: float = 0.0,
    slips: bool = False,
) -> RobustnessCaseResult:
    repeat = root / "cases" / case_id / "repeats" / "1000"
    repeat.mkdir(parents=True)
    (repeat / "episode.mp4").write_bytes(b"video")
    (repeat / "trajectory.json").write_text(
        _trace(
            offset_x=offset_x,
            succeeds=succeeds,
            initial_hash=initial_hash,
            slips=slips,
        ).model_dump_json(indent=2)
    )
    return RobustnessCaseResult(
        case_id=case_id,
        object_position_offset_m=(offset_x, 0.0, 0.0),
        object_observation_bias_m=(observation_bias_x, 0.0, 0.0),
        validity=ExecutionValidity.VALID,
        metric="placement_success",
        metric_value=succeeds,
        passed=succeeds,
        metrics={
            "placement_success": succeeds,
            "placement_success_duration_s": 0.5 if succeeds else 0.0,
            "min_object_target_xy_distance_m": 0.0 if succeeds else 0.4,
            "final_object_target_xy_distance_m": 0.0 if succeeds else 0.4,
            "max_object_lift_m": 0.0,
            "bilateral_contact_duration_s": 0.0,
        },
        artifacts=[
            ArtifactRef(
                kind="video",
                path=f"cases/{case_id}/repeats/1000/episode.mp4",
            ),
            ArtifactRef(
                kind="trajectory",
                path=f"cases/{case_id}/repeats/1000/trajectory.json",
            ),
        ],
        environment={
            "controller_sha256": "b" * 64,
            "model_sha256": "c" * 64,
        },
    )


def _result(root: Path) -> RobustnessCheckResult:
    cases = [
        _case(
            root,
            case_id="nominal_center",
            offset_x=0.0,
            succeeds=True,
            initial_hash="a",
        ),
        _case(
            root,
            case_id="object_right_5cm",
            offset_x=0.05,
            succeeds=True,
            initial_hash="d",
        ),
        _case(
            root,
            case_id="perception_bias_5cm",
            offset_x=0.0,
            succeeds=False,
            initial_hash="e",
            observation_bias_x=0.05,
        ),
    ]
    return RobustnessCheckResult(
        check_id="check_object_pose",
        suite_name="mujoco-pick-place-robustness",
        runner="mujoco",
        subject=SubjectRef(
            id="fixed-fixture-controller",
            code_sha="mujoco-pick-place-demo-v2",
            config={"action_frame_offset_m": [0.0, 0.0, 0.0]},
        ),
        metric="placement_success",
        cases=cases,
        gate=GateResult(
            verdict=GateVerdict.BLOCK,
            exit_code=1,
            summary="BLOCKED: 2/3 required scenario(s) failed",
        ),
        gate_config={"mode": "require_value", "expect": True},
        case_judges={
            case.case_id: {"placement_hold_duration_s": 0.5}
            for case in cases
        },
    )


def _write_result(root: Path) -> RobustnessCheckResult:
    result = _result(root)
    (root / "result.json").write_text(result.model_dump_json(indent=2))
    return result


def test_loader_selects_nominal_and_failed_perception_case_with_full_timeline(
    tmp_path: Path,
    monkeypatch,
):
    _write_result(tmp_path)
    monkeypatch.setattr(
        "policyci.robustness_visualization._probe_video_duration",
        lambda path: 0.6,
    )

    model = load_robustness_visualization(tmp_path)

    assert model.base_case_id == "nominal_center"
    assert model.perturbed_case_id == "perception_bias_5cm"
    assert model.base_passed is True
    assert model.perturbed_passed is False
    assert model.base_offset_m == (0.0, 0.0, 0.0)
    assert model.perturbed_offset_m == (0.0, 0.0, 0.0)
    assert model.perturbed_observation_bias_m == (0.05, 0.0, 0.0)
    assert model.times_s == pytest.approx([index * 0.05 for index in range(13)])
    assert len(model.base_target_distances_m) == 13
    assert len(model.perturbed_target_distances_m) == 13
    assert model.scenarios[2].observation_bias_m == (0.05, 0.0, 0.0)
    assert model.base_subtasks[0] == "approach"
    assert model.base_subtasks[-1] == "retreat"
    assert model.base_task_progress_pct[-1] == 100.0
    assert model.perturbed_task_progress_pct[-1] == 20.0
    assert "ROBUSTNESS GATE BLOCKED" in model.summary_markdown
    assert "fixed-fixture-controller" in model.summary_markdown


def test_loader_rejects_metric_that_contradicts_exact_state(
    tmp_path: Path,
    monkeypatch,
):
    result = _result(tmp_path)
    result.cases[0].metric_value = False
    result.cases[0].passed = False
    (tmp_path / "result.json").write_text(result.model_dump_json(indent=2))
    monkeypatch.setattr(
        "policyci.robustness_visualization._probe_video_duration",
        lambda path: 0.6,
    )

    with pytest.raises(ValueError, match="exact simulator state"):
        load_robustness_visualization(tmp_path)


def test_loader_allows_declared_initial_state_difference_but_not_model_mismatch(
    tmp_path: Path,
    monkeypatch,
):
    result = _result(tmp_path)
    trajectory = (
        tmp_path
        / "cases/perception_bias_5cm/repeats/1000/trajectory.json"
    )
    trace = SimulationTrace.model_validate_json(trajectory.read_text())
    trajectory.write_text(
        trace.model_copy(update={"model_sha256": "f" * 64}).model_dump_json(indent=2)
    )
    (tmp_path / "result.json").write_text(result.model_dump_json(indent=2))
    monkeypatch.setattr(
        "policyci.robustness_visualization._probe_video_duration",
        lambda path: 0.6,
    )

    with pytest.raises(ValueError, match="model hash"):
        load_robustness_visualization(tmp_path)
