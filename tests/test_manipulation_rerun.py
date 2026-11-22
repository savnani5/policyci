"""Video-first Rerun dashboard generation for manipulation checks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from policyci.manipulation_rerun import render_manipulation_rerun
from policyci.manipulation_visualization import (
    ManipulationSensorSeries,
    ManipulationVisualization,
)
from policyci.rerun_visualizer import render_visualization
from policyci.schemas import GateVerdict, TaskStatus


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


def _model(tmp_path: Path) -> ManipulationVisualization:
    baseline_video = tmp_path / "baseline.mp4"
    candidate_video = tmp_path / "candidate.mp4"
    _write_video(baseline_video, color="red")
    _write_video(candidate_video, color="green")
    return ManipulationVisualization(
        check_dir=tmp_path,
        check_id="check_grasp",
        suite_name="mujoco-grasp-regression",
        case_id="grasp_offset",
        unit_id="repeat-000",
        baseline_id="grasp-offset-bad",
        candidate_id="grasp-offset-fixed",
        verdict=GateVerdict.PASS,
        gate_summary="PASS: grasp regression fixed",
        comparison_status=TaskStatus.FIXED,
        metric="lift_success",
        baseline_value=False,
        candidate_value=True,
        timeline="sim_time_s",
        times_s=[0.0, 0.05, 0.10, 0.15, 0.20],
        baseline_video=baseline_video,
        candidate_video=candidate_video,
        baseline_heights_m=[0.0, 0.0, 0.0, 0.0, 0.0],
        candidate_heights_m=[0.0, 0.0, 0.03, 0.08, 0.10],
        baseline_bilateral_contact=[False, False, False, False, False],
        candidate_bilateral_contact=[False, True, True, True, True],
        baseline_lift_success=False,
        candidate_lift_success=True,
        lift_threshold_m=0.08,
        hold_duration_s=0.5,
        failure_start_s=0.10,
        failure_end_s=0.20,
        baseline_sensors=ManipulationSensorSeries(
            gripper_command=[-1.0, -1.0, 1.0, 1.0, 1.0],
            gripper_width_m=[0.08, 0.08, 0.06, 0.03, 0.02],
            left_normal_force_n=[0.0, 0.0, 0.0, 0.0, 0.0],
            right_normal_force_n=[0.0, 0.0, 0.0, 0.0, 0.0],
            wrist_force_magnitude_n=[1.0, 1.2, 1.1, 1.0, 1.0],
        ),
        candidate_sensors=ManipulationSensorSeries(
            gripper_command=[-1.0, -1.0, 1.0, 1.0, 1.0],
            gripper_width_m=[0.08, 0.08, 0.06, 0.05, 0.05],
            left_normal_force_n=[0.0, 0.0, 0.0, 3.0, 4.0],
            right_normal_force_n=[0.0, 0.0, 0.0, 3.5, 4.5],
            wrist_force_magnitude_n=[1.0, 1.2, 1.1, 6.0, 7.0],
        ),
        summary_markdown=("# PASS — FIXED\n\nVerdict computed from exact simulator state."),
    )


def _placement_model(tmp_path: Path) -> ManipulationVisualization:
    return _model(tmp_path).model_copy(
        update={
            "task_kind": "placement",
            "metric": "placement_success",
            "lift_threshold_m": None,
            "baseline_target_distances_m": [0.45, 0.42, 0.40, 0.40, 0.40],
            "candidate_target_distances_m": [0.45, 0.35, 0.20, 0.05, 0.01],
            "baseline_task_success": [False, False, False, False, False],
            "candidate_task_success": [False, False, False, True, True],
            "baseline_placement_success": False,
            "candidate_placement_success": True,
            "failure_start_s": 0.0,
            "failure_end_s": 0.20,
        }
    )


def test_rendered_dashboard_contains_both_videos_and_exact_state_signals(
    tmp_path: Path,
):
    output = render_manipulation_rerun(_model(tmp_path), tmp_path / "view.rrd")

    assert output == tmp_path / "view.rrd"
    assert output.stat().st_size > 0
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
        "/baseline/video",
        "/candidate/video",
        "/exact_state/object_height/baseline",
        "/exact_state/object_height/candidate",
        "/exact_state/bilateral_contact/baseline",
        "/exact_state/bilateral_contact/candidate",
        "/failure_window",
        "/time_panel",
    ):
        assert entity in printed.stdout
    assert "TimePanelBlueprint:time_selection" in printed.stdout


def test_pick_place_dashboard_contains_full_story_placement_signals(tmp_path: Path):
    output = render_manipulation_rerun(
        _placement_model(tmp_path),
        tmp_path / "pick-place.rrd",
    )

    rerun = shutil.which("rerun")
    assert rerun is not None
    printed = subprocess.run(
        [rerun, "rrd", "print", "-vv", str(output)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert printed.returncode == 0, printed.stderr or printed.stdout
    for entity in (
        "/exact_state/target_distance/baseline",
        "/exact_state/target_distance/candidate",
        "/exact_state/task_success/baseline",
        "/exact_state/task_success/candidate",
        "/sensors/force/left_normal/baseline",
        "/sensors/force/left_normal/candidate",
        "/sensors/force/right_normal/baseline",
        "/sensors/force/right_normal/candidate",
        "/sensors/force/wrist/baseline",
        "/sensors/force/wrist/candidate",
        "/sensors/gripper/width_cm/baseline",
        "/sensors/gripper/width_cm/candidate",
        "/sensors/gripper/command/baseline",
        "/sensors/gripper/command/candidate",
    ):
        assert entity in printed.stdout


def test_renderer_rejects_output_outside_check_directory(tmp_path: Path):
    check_dir = tmp_path / "check"
    check_dir.mkdir()
    model = _model(check_dir)

    with pytest.raises(ValueError, match="confined"):
        render_manipulation_rerun(model, tmp_path / "outside.rrd")


def test_renderer_rejects_symlinked_visualization_directory(tmp_path: Path):
    check_dir = tmp_path / "check"
    check_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (check_dir / "visualizations").symlink_to(outside, target_is_directory=True)
    model = _model(check_dir)

    with pytest.raises(ValueError, match="confined"):
        render_manipulation_rerun(
            model,
            check_dir / "visualizations/grasp_offset/repeat-000.rrd",
        )


def test_generic_renderer_dispatches_manipulation_model(tmp_path: Path):
    output = render_visualization(_model(tmp_path), tmp_path / "generic.rrd")

    assert output == tmp_path / "generic.rrd"
    assert output.stat().st_size > 0
