"""Rerun dashboard contract for absolute object-pose robustness gates."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import policyci.robustness_rerun as robustness_rerun
from policyci.robustness_rerun import render_robustness_rerun
from policyci.robustness_visualization import (
    RobustnessScenarioDisplay,
    RobustnessVisualization,
)
from policyci.schemas import ExecutionValidity, GateVerdict


def _write_video(path: Path, *, color: str) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for Rerun video tests")
    completed = subprocess.run(
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=64x64:r=20:d=0.25",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def _model(tmp_path: Path) -> RobustnessVisualization:
    base_video = tmp_path / "base.mp4"
    perturbed_video = tmp_path / "perturbed.mp4"
    _write_video(base_video, color="green")
    _write_video(perturbed_video, color="red")
    return RobustnessVisualization(
        check_dir=tmp_path,
        check_id="check_object_pose",
        suite_name="mujoco-pick-place-robustness",
        subject_id="fixed-fixture-controller",
        controller_sha256="b" * 64,
        verdict=GateVerdict.BLOCK,
        gate_summary="BLOCKED: 2/3 required scenarios failed",
        metric="placement_success",
        times_s=[0.0, 0.05, 0.10, 0.15, 0.20],
        scenarios=[
            RobustnessScenarioDisplay(
                case_id="nominal_center",
                offset_m=(0.0, 0.0, 0.0),
                observation_bias_m=(0.0, 0.0, 0.0),
                validity=ExecutionValidity.VALID,
                passed=True,
                metric_value=True,
            ),
            RobustnessScenarioDisplay(
                case_id="perception_bias_5cm",
                offset_m=(0.0, 0.0, 0.0),
                observation_bias_m=(0.05, 0.0, 0.0),
                validity=ExecutionValidity.VALID,
                passed=False,
                metric_value=False,
            ),
        ],
        base_case_id="nominal_center",
        perturbed_case_id="perception_bias_5cm",
        base_offset_m=(0.0, 0.0, 0.0),
        perturbed_offset_m=(0.0, 0.0, 0.0),
        base_observation_bias_m=(0.0, 0.0, 0.0),
        perturbed_observation_bias_m=(0.05, 0.0, 0.0),
        base_passed=True,
        perturbed_passed=False,
        base_video=base_video,
        perturbed_video=perturbed_video,
        base_target_distances_m=[0.45, 0.35, 0.20, 0.04, 0.02],
        perturbed_target_distances_m=[0.45, 0.36, 0.28, 0.20, 0.18],
        base_object_heights_m=[0.84, 0.84, 0.90, 1.02, 0.80],
        perturbed_object_heights_m=[0.84, 0.84, 0.92, 1.01, 0.84],
        base_task_success=[False, False, False, True, True],
        perturbed_task_success=[False, False, False, False, False],
        base_subtasks=["approach", "align", "grasp", "transport", "place"],
        perturbed_subtasks=["approach", "align", "grasp", "transport", "place"],
        base_task_progress_pct=[0.0, 20.0, 40.0, 80.0, 100.0],
        perturbed_task_progress_pct=[0.0, 20.0, 20.0, 20.0, 20.0],
        hold_duration_s=0.5,
        summary_markdown="# ROBUSTNESS GATE BLOCKED",
    )


def test_dashboard_contains_full_base_perturbation_and_gate_evidence(tmp_path: Path):
    output = render_robustness_rerun(_model(tmp_path), tmp_path / "robustness.rrd")

    rerun = shutil.which("rerun")
    assert rerun is not None
    verified = subprocess.run(
        [rerun, "rrd", "verify", str(output)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert verified.returncode == 0, verified.stderr or verified.stdout
    printed = subprocess.run(
        [rerun, "rrd", "print", "-vv", str(output)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert printed.returncode == 0, printed.stderr or printed.stdout
    for entity in (
        "/base_case/video",
        "/perturbed/video",
        "/verdict",
        "/task_progress/base_case",
        "/task_progress/perturbed",
        "/subtask_timeline/segments",
        "/subtask_timeline/playhead",
        "/exact_state/target_distance/base_case",
        "/exact_state/target_distance/perturbed",
        "/exact_state/target_distance/threshold",
        "/exact_state/object_height/base_case",
        "/exact_state/object_height/perturbed",
        "/exact_state/task_success/base_case",
        "/exact_state/task_success/perturbed",
        "/time_panel",
    ):
        assert entity in printed.stdout
    assert "/summary" not in printed.stdout
    assert "/subtask_timeline/row_labels" not in printed.stdout
    assert "/baseline/" not in printed.stdout
    assert "/candidate/" not in printed.stdout


def test_dashboard_uses_one_shared_copy_of_identical_subtask_segments(tmp_path: Path):
    model = _model(tmp_path)

    segments = robustness_rerun._shared_subtask_segments(model)

    assert segments == [
        ("approach", 0.0, 0.05),
        ("align", 0.05, 0.10),
        ("grasp", 0.10, 0.15),
        ("transport", 0.15, 0.20),
        ("place", 0.20, 0.25),
    ]


def test_robustness_renderer_rejects_output_outside_result_directory(tmp_path: Path):
    check_dir = tmp_path / "check"
    check_dir.mkdir()
    model = _model(check_dir)

    with pytest.raises(ValueError, match="confined"):
        render_robustness_rerun(model, tmp_path / "outside.rrd")
