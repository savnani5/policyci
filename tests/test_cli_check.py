"""Public CLI contract for `robolens check`."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import policyci.cli as cli
from policyci.runner.base import PreflightResult
from policyci.checkspec import CaseSpec, CheckSuite, GateSpec, RunnerSpec
from policyci.manipulation_visualization import ManipulationVisualization
from policyci.robustness_visualization import (
    RobustnessScenarioDisplay,
    RobustnessVisualization,
)
from policyci.runner.mujoco import MujocoRunner
from policyci.runner.ros2_replay import Ros2ReplayRunner
from policyci.schemas import (
    ArtifactRef,
    CaseExecution,
    ExecutionValidity,
    GateVerdict,
    TaskStatus,
)
from policyci.visualization import CheckVisualization


runner = CliRunner()


class CliFakeRunner:
    def __init__(self, baseline_value: float, candidate_value: float, *, invalid=False):
        self.values = {"baseline": baseline_value, "candidate": candidate_value}
        self.invalid = invalid

    def preflight(self, subject, case):
        if self.invalid and subject.id == "candidate":
            return PreflightResult.no_verdict("candidate fixture unavailable")
        return PreflightResult.valid()

    def run_case(self, subject, case, output_dir):
        artifacts = []
        for unit_id in ("1000", "1001"):
            relative = f"artifacts/{unit_id}.mp4"
            path = output_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{subject.id}:{unit_id}".encode())
            artifacts.append(ArtifactRef(kind="video", path=relative))
        return CaseExecution(
            subject=subject,
            case_id=case.id,
            runner="fake",
            validity=ExecutionValidity.VALID,
            metrics={"success_rate": self.values[subject.id]},
            valid_units=2,
            unit_ids=["1000", "1001"],
            artifacts=artifacts,
            synthetic=True,
        )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """
version: 1
name: release
runner: {type: libero, suite: libero_object}
execution: {episodes: 2, seeds: [1000, 1001]}
gate:
  metric: success_rate
  mode: max_drop
  max_drop_pp: 10
  min_valid_episodes: 2
cases:
  - {id: soup, task: 0}
"""
    )
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text("id: baseline\ncheckpoint: fixtures/baseline\n")
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text("id: candidate\ncheckpoint: fixtures/candidate\n")
    return suite, baseline, candidate


@pytest.mark.parametrize(
    ("baseline_value", "candidate_value", "invalid", "exit_code", "verdict"),
    [
        (90.0, 90.0, False, 0, "PASS"),
        (90.0, 20.0, False, 1, "BLOCK"),
        (90.0, 90.0, True, 2, "NO_VERDICT"),
    ],
)
def test_check_command_returns_stable_gate_exit_codes(
    tmp_path: Path,
    monkeypatch,
    baseline_value: float,
    candidate_value: float,
    invalid: bool,
    exit_code: int,
    verdict: str,
):
    suite, baseline, candidate = _inputs(tmp_path)
    fake = CliFakeRunner(baseline_value, candidate_value, invalid=invalid)
    monkeypatch.setattr(cli, "_build_check_runner", lambda loaded_suite: fake)

    result = runner.invoke(
        cli.app,
        [
            "check",
            "--suite",
            str(suite),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == exit_code
    assert verdict in result.stdout
    assert "result=" in result.stdout
    result_paths = list((tmp_path / "runs").glob("check_*/result.json"))
    assert len(result_paths) == 1


class RobustnessCliFakeRunner:
    def preflight(self, subject, case):
        return PreflightResult.valid()

    def run_case(self, subject, case, output_dir):
        repeat = output_dir / "repeats" / "1000"
        repeat.mkdir(parents=True)
        (repeat / "episode.mp4").write_bytes(b"video")
        (repeat / "trajectory.json").write_text("{}")
        offset = case.resolved_object_position_offset_m()
        return CaseExecution(
            subject=subject,
            case_id=case.id,
            runner="mujoco",
            validity=ExecutionValidity.VALID,
            metrics={"placement_success": case.id == "nominal_center"},
            valid_units=1,
            unit_ids=["1000"],
            artifacts=[
                ArtifactRef(kind="video", path="repeats/1000/episode.mp4"),
                ArtifactRef(kind="trajectory", path="repeats/1000/trajectory.json"),
            ],
            environment={
                "model_sha256": "a" * 64,
                "controller_sha256": "b" * 64,
                "adapter_sha256": "c" * 64,
                "camera": "agentview:1280x720",
                "camera_fovy_deg": 34.0,
                "control_hz": 20,
                "horizon": 240,
                "task": "PickPlaceCan",
                "robosuite_version": "1.5.2",
                "mujoco_version": "3.3.7",
                "object_offset_x_m": offset[0],
                "object_offset_y_m": offset[1],
                "object_offset_z_m": offset[2],
                "object_observation_bias_x_m": (
                    case.resolved_object_observation_bias_m()[0]
                ),
                "object_observation_bias_y_m": (
                    case.resolved_object_observation_bias_m()[1]
                ),
                "object_observation_bias_z_m": (
                    case.resolved_object_observation_bias_m()[2]
                ),
            },
        )


def test_robustness_check_command_prints_cases_and_preserves_block_exit(
    tmp_path: Path,
    monkeypatch,
):
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """
version: 1
name: object-pose-robustness
runner: {type: mujoco}
gate: {metric: placement_success, mode: require_value, expect: true}
cases:
  - id: nominal_center
    object_position_offset_m: [0.0, 0.0, 0.0]
    judge: {placement_hold_duration_s: 0.5}
  - id: object_right_5cm
    object_position_offset_m: [0.05, 0.0, 0.0]
    judge: {placement_hold_duration_s: 0.5}
"""
    )
    subject = tmp_path / "subject.yaml"
    subject.write_text(
        """
id: fixed-fixture-controller
code_sha: mujoco-pick-place-demo-v2
config: {action_frame_offset_m: [0.0, 0.0, 0.0]}
"""
    )
    monkeypatch.setattr(
        cli,
        "_build_check_runner",
        lambda loaded_suite: RobustnessCliFakeRunner(),
    )

    result = runner.invoke(
        cli.app,
        [
            "robustness-check",
            "--suite",
            str(suite),
            "--subject",
            str(subject),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 1
    assert "nominal_center" in result.stdout and "PASS" in result.stdout
    assert "object_right_5cm" in result.stdout and "FAIL" in result.stdout
    assert "ROBUSTNESS GATE BLOCKED" in result.stdout
    assert "result=" in result.stdout


def test_robustness_visualize_command_builds_rrd_without_opening(
    tmp_path: Path,
    monkeypatch,
):
    base_video = tmp_path / "base.mp4"
    perturbed_video = tmp_path / "perturbed.mp4"
    base_video.write_bytes(b"base")
    perturbed_video.write_bytes(b"perturbed")
    model = RobustnessVisualization(
        check_dir=tmp_path,
        check_id="check_object_pose",
        suite_name="object-pose-robustness",
        subject_id="fixed-fixture-controller",
        controller_sha256="b" * 64,
        verdict=GateVerdict.BLOCK,
        gate_summary="BLOCKED: one protected scenario failed",
        metric="placement_success",
        times_s=[0.0, 0.05],
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
                case_id="object_right_5cm",
                offset_m=(0.05, 0.0, 0.0),
                observation_bias_m=(0.0, 0.0, 0.0),
                validity=ExecutionValidity.VALID,
                passed=False,
                metric_value=False,
            ),
        ],
        base_case_id="nominal_center",
        perturbed_case_id="object_right_5cm",
        base_offset_m=(0.0, 0.0, 0.0),
        perturbed_offset_m=(0.05, 0.0, 0.0),
        base_observation_bias_m=(0.0, 0.0, 0.0),
        perturbed_observation_bias_m=(0.0, 0.0, 0.0),
        base_passed=True,
        perturbed_passed=False,
        base_video=base_video,
        perturbed_video=perturbed_video,
        base_target_distances_m=[0.4, 0.02],
        perturbed_target_distances_m=[0.4, 0.18],
        base_object_heights_m=[0.84, 0.80],
        perturbed_object_heights_m=[0.84, 0.84],
        base_task_success=[False, True],
        perturbed_task_success=[False, False],
        base_subtasks=["approach", "place"],
        perturbed_subtasks=["approach", "grasp"],
        base_task_progress_pct=[20.0, 100.0],
        perturbed_task_progress_pct=[20.0, 20.0],
        hold_duration_s=0.5,
        summary_markdown="# ROBUSTNESS GATE BLOCKED",
    )
    observed: dict = {}

    def fake_load(check_dir, *, failed_case_id):
        observed.update(check_dir=check_dir, failed_case_id=failed_case_id)
        return model

    def fake_render(loaded, output):
        assert loaded == model
        output.parent.mkdir(parents=True)
        output.write_bytes(b"rrd")
        return output

    monkeypatch.setattr(
        "policyci.robustness_visualization.load_robustness_visualization",
        fake_load,
    )
    monkeypatch.setattr(
        "policyci.robustness_rerun.render_robustness_rerun",
        fake_render,
    )
    monkeypatch.setattr(
        "policyci.rerun_visualizer.open_rerun_visualization",
        lambda *args, **kwargs: pytest.fail("headless visualization launched a GUI"),
    )

    result = runner.invoke(
        cli.app,
        [
            "robustness-visualize",
            str(tmp_path),
            "--case",
            "object_right_5cm",
            "--no-open",
        ],
    )

    assert result.exit_code == 0
    assert observed == {
        "check_dir": tmp_path,
        "failed_case_id": "object_right_5cm",
    }
    assert "ROBUSTNESS GATE BLOCKED" in result.stdout
    assert "nominal_center" in result.stdout
    assert "object_right_5cm" in result.stdout
    assert (tmp_path / "visualizations/robustness/object_right_5cm.rrd").is_file()


def test_malformed_suite_exits_three_with_actionable_error(tmp_path: Path):
    suite, baseline, candidate = _inputs(tmp_path)
    suite.write_text("version: 1\nname: missing-everything\n")

    result = runner.invoke(
        cli.app,
        [
            "check",
            "--suite",
            str(suite),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
        ],
    )

    assert result.exit_code == 3
    assert "configuration error" in result.stdout.lower()
    assert "runner" in result.stdout.lower()


def test_help_exposes_check_as_the_primary_workflow():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "check" in result.stdout
    assert "baseline" in result.stdout.lower()


def _visualization_model(tmp_path: Path) -> CheckVisualization:
    baseline_mcap = tmp_path / "baseline.mcap"
    candidate_mcap = tmp_path / "candidate.mcap"
    baseline_mcap.write_bytes(b"baseline")
    candidate_mcap.write_bytes(b"candidate")
    return CheckVisualization(
        check_dir=tmp_path,
        check_id="check_demo",
        suite_name="twist-replay",
        case_id="zero_command_drift",
        unit_id="repeat-000",
        baseline_id="known-bad",
        candidate_id="fixed",
        verdict=GateVerdict.PASS,
        gate_summary="PASS: all required cases satisfy the configured gate",
        comparison_status=TaskStatus.FIXED,
        metric="within_tolerance",
        baseline_value=False,
        candidate_value=True,
        judge={"tolerance": 0.05},
        tolerance=0.05,
        timeline="sample_index",
        expected=[0.0],
        baseline_observed=[0.4],
        candidate_observed=[0.0],
        baseline_error=[0.4],
        candidate_error=[0.0],
        baseline_unit_within_tolerance=False,
        candidate_unit_within_tolerance=True,
        unit="m/s",
        baseline_mcaps=[baseline_mcap],
        candidate_mcaps=[candidate_mcap],
        summary_markdown="# PASS — FIXED",
    )


def test_visualize_command_builds_rrd_without_opening_in_headless_mode(
    tmp_path: Path,
    monkeypatch,
):
    model = _visualization_model(tmp_path)
    observed: dict = {}

    def fake_load(check_dir, *, case_id, unit_id):
        observed.update(check_dir=check_dir, case_id=case_id, unit_id=unit_id)
        return model

    def fake_render(loaded, output):
        assert loaded == model
        output.parent.mkdir(parents=True)
        output.write_bytes(b"rrd")
        return output

    monkeypatch.setattr(
        "policyci.visualization.load_visualization",
        fake_load,
    )
    monkeypatch.setattr(
        "policyci.rerun_visualizer.render_visualization",
        fake_render,
    )
    monkeypatch.setattr(
        "policyci.rerun_visualizer.open_visualization",
        lambda *args, **kwargs: pytest.fail("headless visualization launched a GUI"),
    )

    result = runner.invoke(cli.app, ["visualize", str(tmp_path), "--no-open"])

    assert result.exit_code == 0
    assert observed == {"check_dir": tmp_path, "case_id": None, "unit_id": None}
    assert "PASS" in result.stdout
    assert "FIXED" in result.stdout
    assert "within_tolerance" in result.stdout
    assert "aggregate_metric=" in result.stdout
    assert "selected_repeat=baseline_FAIL candidate_PASS" in result.stdout
    assert "False" in result.stdout and "True" in result.stdout
    assert "zero_command_drift / repeat-000" in result.stdout
    assert (tmp_path / "visualizations/zero_command_drift/repeat-000.rrd").is_file()


def test_visualize_command_returns_three_for_missing_check(tmp_path: Path):
    result = runner.invoke(cli.app, ["visualize", str(tmp_path), "--no-open"])

    assert result.exit_code == 3
    assert "visualization error" in result.stdout.lower()
    assert "result" in result.stdout.lower()


def test_visualize_command_explains_missing_optional_dependency(
    tmp_path: Path,
    monkeypatch,
):
    model = _visualization_model(tmp_path)
    monkeypatch.setattr(
        "policyci.visualization.load_visualization",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        "policyci.rerun_visualizer.render_visualization",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError(
                "Rerun visualization is not installed; run `uv sync --extra visualize`"
            )
        ),
    )

    result = runner.invoke(cli.app, ["visualize", str(tmp_path), "--no-open"])

    assert result.exit_code == 3
    assert "uv sync" in result.stdout
    assert "--extra" in result.stdout
    assert "visualize" in result.stdout


def test_runner_factory_dispatches_ros2_replay(monkeypatch):
    class Adapter:
        def applied_subject_fields(self, subject, case):
            return set()

        def subject_command(self, subject, case):
            return ["subject"]

        def readiness_command(self, subject, case):
            return ["ready"]

        def analyze(self, subject, case, capture):
            return {"ok": True}

    monkeypatch.setattr(
        "policyci.runner.ros2_replay._load_adapter",
        lambda reference: Adapter(),
    )
    suite = CheckSuite(
        version=1,
        name="replay",
        runner=RunnerSpec(type="ros2_replay", adapter="fake.module:Adapter"),
        gate=GateSpec(metric="ok", mode="require_value", expect=True),
        cases=[
            {
                "id": "case",
                "bag": "input.mcap",
                "required_topics": ["/input"],
                "capture_topics": ["/output"],
            }
        ],
    )
    built = cli._build_check_runner(suite)
    assert isinstance(built, Ros2ReplayRunner)
    assert built.adapter.__class__.__name__ == "Adapter"


def test_runner_factory_dispatches_mujoco_without_loading_simulation_dependencies():
    suite = CheckSuite(
        version=1,
        name="grasp",
        runner=RunnerSpec(type="mujoco"),
        gate=GateSpec(
            metric="lift_success",
            mode="require_value",
            expect=True,
        ),
        cases=[
            CaseSpec(
                id="grasp_offset",
                judge={"lift_threshold_m": 0.08, "hold_duration_s": 0.5},
            )
        ],
    )

    built = cli._build_check_runner(suite)

    assert isinstance(built, MujocoRunner)
    assert built._adapter is None


def _manipulation_visualization_model(tmp_path: Path) -> ManipulationVisualization:
    baseline_video = tmp_path / "baseline.mp4"
    candidate_video = tmp_path / "candidate.mp4"
    baseline_video.write_bytes(b"baseline-video")
    candidate_video.write_bytes(b"candidate-video")
    return ManipulationVisualization(
        check_dir=tmp_path,
        check_id="check_grasp",
        suite_name="mujoco-grasp-regression",
        case_id="grasp_offset",
        unit_id="1000",
        baseline_id="grasp-offset-bad",
        candidate_id="grasp-offset-fixed",
        verdict=GateVerdict.PASS,
        gate_summary="PASS: grasp regression fixed",
        comparison_status=TaskStatus.FIXED,
        metric="lift_success",
        baseline_value=False,
        candidate_value=True,
        timeline="sim_time_s",
        times_s=[0.0, 0.05],
        baseline_video=baseline_video,
        candidate_video=candidate_video,
        baseline_heights_m=[0.0, 0.0],
        candidate_heights_m=[0.0, 0.09],
        baseline_bilateral_contact=[False, False],
        candidate_bilateral_contact=[False, True],
        baseline_lift_success=False,
        candidate_lift_success=True,
        lift_threshold_m=0.08,
        hold_duration_s=0.5,
        failure_start_s=0.0,
        failure_end_s=0.05,
        summary_markdown="# PASS — FIXED",
    )


def test_visualize_command_dispatches_manipulation_model(
    tmp_path: Path,
    monkeypatch,
):
    model = _manipulation_visualization_model(tmp_path)

    def fake_render(loaded, output):
        assert loaded == model
        output.parent.mkdir(parents=True)
        output.write_bytes(b"rrd")
        return output

    monkeypatch.setattr(
        "policyci.visualization.load_visualization",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        "policyci.rerun_visualizer.render_visualization",
        fake_render,
    )

    result = runner.invoke(cli.app, ["visualize", str(tmp_path), "--no-open"])

    assert result.exit_code == 0
    assert "grasp_offset / 1000" in result.stdout
    assert "lift_success" in result.stdout
    assert "selected_repeat=baseline_FAIL candidate_PASS" in result.stdout


def test_bad_replay_adapter_is_configuration_error(tmp_path: Path):
    suite, baseline, candidate = _inputs(tmp_path)
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    suite.write_text(
        f"""
version: 1
name: replay
runner: {{type: ros2_replay, adapter: missing_adapter.module:Adapter}}
gate: {{metric: ok, mode: require_value, expect: true}}
cases:
  - id: case
    bag: {bag.name}
    required_topics: [/input]
    capture_topics: [/output]
"""
    )
    result = runner.invoke(
        cli.app,
        [
            "check",
            "--suite",
            str(suite),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
        ],
    )
    assert result.exit_code == 3
    assert "configuration error" in result.stdout.lower()
    assert "missing_adapter" in result.stdout
