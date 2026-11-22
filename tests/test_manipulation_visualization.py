"""Sound loading for synchronized manipulation evidence."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from policyci.manipulation_visualization import load_manipulation_visualization
from policyci.schemas import (
    ArtifactRef,
    CheckResult,
    Comparison,
    EvidencePair,
    ExecutionValidity,
    GateResult,
    GateVerdict,
    SimulationSample,
    SimulationTrace,
    SubjectRef,
    TaskStatus,
)
from policyci.visualization import load_visualization


def _write_video(path: Path, *, duration_s: float = 0.75, color: str) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for synchronized video evidence tests")
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=64x64:r=20:d={duration_s}",
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


def _trace(
    *,
    fixed: bool,
    placement: bool = False,
    sensor_evidence: bool = True,
) -> SimulationTrace:
    samples: list[SimulationSample] = []
    for index in range(15):
        time_s = index / 20
        lifted = fixed and index >= 3
        sensor_fields = (
            {
                "gripper_command": 1.0 if index >= 2 else -1.0,
                "left_finger_normal_force_n": 4.0 if lifted else 0.0,
                "right_finger_normal_force_n": 5.0 if lifted else 0.0,
                "wrist_force_xyz_n": (3.0, 4.0, 0.0) if lifted else (0.0, 0.0, 1.0),
            }
            if sensor_evidence
            else {}
        )
        samples.append(
            SimulationSample(
                time_s=time_s,
                object_position_m=(
                    0.20 if placement and lifted else 0.0,
                    0.20 if placement and lifted else 0.0,
                    0.84 if placement else (0.89 if lifted else 0.80),
                ),
                object_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                eef_position_m=(0.0, 0.0, 0.94 if lifted else 0.85),
                eef_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                gripper_width_m=0.02 if lifted else 0.08,
                left_finger_contact=lifted,
                right_finger_contact=lifted,
                target_position_m=(0.20, 0.20, 0.80) if placement else None,
                task_success=lifted if placement else None,
                **sensor_fields,
            )
        )
    return SimulationTrace(
        sample_rate_hz=20,
        seed=7,
        initial_state_sha256="a" * 64,
        model_sha256="b" * 64,
        samples=samples,
    )


def _write_check(
    tmp_path: Path,
    *,
    video_duration_s: float = 0.75,
    placement: bool = False,
    sensor_evidence: bool = True,
) -> Path:
    root = tmp_path / "check_grasp"
    case_id = "grasp_offset"
    unit_id = "repeat-000"
    side_artifacts: dict[str, list[ArtifactRef]] = {}
    for side, fixed, color in (
        ("baseline", False, "red"),
        ("candidate", True, "green"),
    ):
        prefix = f"{side}/cases/{case_id}/repeats/{unit_id}"
        video_relative = f"{prefix}/episode.mp4"
        trace_relative = f"{prefix}/trajectory.json"
        _write_video(
            root / video_relative,
            duration_s=video_duration_s,
            color=color,
        )
        trace_path = root / trace_relative
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            _trace(
                fixed=fixed,
                placement=placement,
                sensor_evidence=sensor_evidence,
            ).model_dump_json(indent=2)
        )
        side_artifacts[side] = [
            ArtifactRef(kind="video", path=video_relative),
            ArtifactRef(kind="trajectory", path=trace_relative),
        ]

    result = CheckResult(
        check_id="check_grasp",
        suite_name="mujoco-grasp-regression",
        runner="mujoco",
        baseline=SubjectRef(
            id="grasp-offset-bad",
            config={"action_frame_offset_m": [0.06, 0.0, 0.0]},
        ),
        candidate=SubjectRef(
            id="grasp-offset-fixed",
            config={"action_frame_offset_m": [0.0, 0.0, 0.0]},
        ),
        comparisons=[
            Comparison(
                case_id=case_id,
                status=TaskStatus.FIXED,
                baseline_validity=ExecutionValidity.VALID,
                candidate_validity=ExecutionValidity.VALID,
                metric="placement_success" if placement else "lift_success",
                baseline_value=False,
                candidate_value=True,
                evidence_pairs=[
                    EvidencePair(
                        case_id=case_id,
                        unit_id=unit_id,
                        baseline_artifacts=side_artifacts["baseline"],
                        candidate_artifacts=side_artifacts["candidate"],
                    )
                ],
            )
        ],
        gate=GateResult(
            verdict=GateVerdict.PASS,
            exit_code=0,
            summary="PASS: grasp regression fixed",
        ),
        gate_config={
            "metric": "placement_success" if placement else "lift_success",
            "mode": "require_value",
        },
        case_judges={
            case_id: (
                {"placement_hold_duration_s": 0.5}
                if placement
                else {"lift_threshold_m": 0.08, "hold_duration_s": 0.5}
            )
        },
    )
    root.mkdir(exist_ok=True)
    (root / "result.json").write_text(result.model_dump_json(indent=2))
    return root


def _result_payload(root: Path) -> dict:
    return json.loads((root / "result.json").read_text())


def _write_result_payload(root: Path, payload: dict) -> None:
    (root / "result.json").write_text(json.dumps(payload, indent=2))


def test_loader_builds_video_first_model_from_exact_simulator_state(tmp_path: Path):
    root = _write_check(tmp_path)

    model = load_manipulation_visualization(root)

    assert model.case_id == "grasp_offset"
    assert model.unit_id == "repeat-000"
    assert model.timeline == "sim_time_s"
    assert model.times_s == pytest.approx([index / 20 for index in range(15)])
    assert model.baseline_heights_m == pytest.approx([0.0] * 15)
    assert model.candidate_heights_m[-1] == pytest.approx(0.09)
    assert model.baseline_bilateral_contact == [False] * 15
    assert model.candidate_bilateral_contact[3:] == [True] * 12
    assert model.baseline_lift_success is False
    assert model.candidate_lift_success is True
    assert model.baseline_unit_pass is False
    assert model.candidate_unit_pass is True
    assert model.lift_threshold_m == 0.08
    assert model.hold_duration_s == 0.5
    assert model.failure_start_s == 0.0
    assert model.failure_end_s == pytest.approx(0.7)
    assert model.baseline_video.stat().st_size > 0
    assert model.candidate_video.stat().st_size > 0
    assert "exact simulator state" in model.summary_markdown


def test_loader_builds_full_story_pick_place_model(tmp_path: Path):
    root = _write_check(tmp_path, placement=True)

    model = load_manipulation_visualization(root)

    assert model.task_kind == "placement"
    assert model.baseline_lift_success is False
    assert model.candidate_lift_success is False
    assert model.baseline_placement_success is False
    assert model.candidate_placement_success is True
    assert model.baseline_unit_pass is False
    assert model.candidate_unit_pass is True
    assert model.baseline_target_distances_m == pytest.approx([2**0.5 * 0.2] * 15)
    assert model.candidate_target_distances_m[-1] == pytest.approx(0.0)
    assert model.baseline_task_success == [False] * 15
    assert model.candidate_task_success[3:] == [True] * 12
    assert model.baseline_sensors is not None
    assert model.candidate_sensors is not None
    assert model.baseline_sensors.gripper_width_m == pytest.approx([0.08] * 15)
    assert model.candidate_sensors.gripper_width_m[3:] == pytest.approx([0.02] * 12)
    assert model.candidate_sensors.left_normal_force_n[3:] == pytest.approx([4.0] * 12)
    assert model.candidate_sensors.right_normal_force_n[3:] == pytest.approx([5.0] * 12)
    assert model.candidate_sensors.wrist_force_magnitude_n[3:] == pytest.approx([5.0] * 12)
    assert model.story_start_s == 0.0
    assert model.story_end_s == pytest.approx(0.7)
    assert "placed in target bin" in model.summary_markdown


def test_loader_preserves_legacy_traces_without_sensor_evidence(tmp_path: Path):
    root = _write_check(tmp_path, placement=True, sensor_evidence=False)

    model = load_manipulation_visualization(root)

    assert model.baseline_sensors is None
    assert model.candidate_sensors is None


def test_loader_rejects_sensor_evidence_on_only_one_side(tmp_path: Path):
    root = _write_check(tmp_path, placement=True)
    trace_path = next(root.rglob("candidate/**/trajectory.json"))
    payload = json.loads(trace_path.read_text())
    sensor_fields = (
        "gripper_command",
        "left_finger_normal_force_n",
        "right_finger_normal_force_n",
        "wrist_force_xyz_n",
    )
    for sample in payload["samples"]:
        for field in sensor_fields:
            sample.pop(field)
    trace_path.write_text(json.dumps(payload, indent=2))

    with pytest.raises(ValueError, match="sensor evidence availability differs"):
        load_manipulation_visualization(root)


def test_generic_loader_dispatches_mujoco(tmp_path: Path):
    root = _write_check(tmp_path)

    model = load_visualization(root)

    assert model.timeline == "sim_time_s"
    assert model.candidate_unit_pass is True


def test_loader_rejects_aggregate_metric_that_contradicts_exact_state(tmp_path: Path):
    root = _write_check(tmp_path)
    payload = _result_payload(root)
    payload["comparisons"][0]["candidate_value"] = False
    _write_result_payload(root, payload)

    with pytest.raises(ValueError, match="automatic metric"):
        load_manipulation_visualization(root)


def test_loader_rejects_different_initial_state_hashes(tmp_path: Path):
    root = _write_check(tmp_path)
    trace_path = next(root.rglob("candidate/**/trajectory.json"))
    payload = json.loads(trace_path.read_text())
    payload["initial_state_sha256"] = "c" * 64
    trace_path.write_text(json.dumps(payload, indent=2))

    with pytest.raises(ValueError, match="initial state"):
        load_manipulation_visualization(root)


def test_loader_rejects_mismatched_sample_times(tmp_path: Path):
    root = _write_check(tmp_path)
    trace_path = next(root.rglob("candidate/**/trajectory.json"))
    payload = json.loads(trace_path.read_text())
    payload["samples"][-1]["time_s"] = 0.71
    trace_path.write_text(json.dumps(payload, indent=2))

    with pytest.raises(ValueError, match="sample times"):
        load_manipulation_visualization(root)


@pytest.mark.parametrize("artifact_kind", ["video", "trajectory"])
def test_loader_requires_exactly_one_artifact_per_kind(
    tmp_path: Path,
    artifact_kind: str,
):
    root = _write_check(tmp_path)
    payload = _result_payload(root)
    artifacts = payload["comparisons"][0]["evidence_pairs"][0]["candidate_artifacts"]
    artifact = next(item for item in artifacts if item["kind"] == artifact_kind)
    artifacts.append(dict(artifact))
    _write_result_payload(root, payload)

    with pytest.raises(ValueError, match="exactly one"):
        load_manipulation_visualization(root)


def test_loader_rejects_artifact_symlink_outside_check(tmp_path: Path):
    root = _write_check(tmp_path)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    candidate_video = next(root.rglob("candidate/**/episode.mp4"))
    candidate_video.unlink()
    candidate_video.symlink_to(outside)

    with pytest.raises(ValueError, match="confined"):
        load_manipulation_visualization(root)


def test_loader_rejects_video_that_is_not_synchronized_to_trace(tmp_path: Path):
    root = _write_check(tmp_path, video_duration_s=3.0)

    with pytest.raises(ValueError, match="duration"):
        load_manipulation_visualization(root)
