"""Absolute single-subject gate behavior for protected simulation scenarios."""

from __future__ import annotations

from pathlib import Path

from policyci.checkspec import CaseSpec, CheckSuite, GateSpec, RunnerSpec
from policyci.robustness_check import execute_robustness_check
from policyci.runner.base import PreflightResult
from policyci.schemas import (
    ArtifactRef,
    CaseExecution,
    ExecutionValidity,
    GateVerdict,
    SubjectRef,
)


def three_case_suite() -> CheckSuite:
    return CheckSuite(
        version=1,
        name="mujoco-pick-place-robustness",
        runner=RunnerSpec(type="mujoco"),
        gate=GateSpec(
            metric="placement_success",
            mode="require_value",
            expect=True,
            min_valid_repeats=1,
        ),
        cases=[
            CaseSpec(
                id="nominal_center",
                object_position_offset_m=(0.0, 0.0, 0.0),
                judge={"placement_hold_duration_s": 0.5},
            ),
            CaseSpec(
                id="object_right_5cm",
                object_position_offset_m=(0.05, 0.0, 0.0),
                judge={"placement_hold_duration_s": 0.5},
            ),
            CaseSpec(
                id="perception_bias_5cm",
                object_observation_bias_m=(0.05, 0.0, 0.0),
                judge={"placement_hold_duration_s": 0.5},
            ),
        ],
    )


def fixed_subject() -> SubjectRef:
    return SubjectRef(
        id="fixed-fixture-controller",
        code_sha="mujoco-pick-place-demo-v2",
        config={"action_frame_offset_m": [0.0, 0.0, 0.0]},
    )


class FakeRunner:
    def __init__(
        self,
        values: dict[str, bool | ExecutionValidity],
        *,
        controller_by_case: dict[str, str] | None = None,
        observation_bias_by_case: dict[str, tuple[float, float, float]] | None = None,
    ) -> None:
        self.values = values
        self.controller_by_case = controller_by_case or {}
        self.observation_bias_by_case = observation_bias_by_case or {}

    def preflight(self, subject: SubjectRef, case: CaseSpec) -> PreflightResult:
        del subject
        if self.values[case.id] is ExecutionValidity.NO_VERDICT:
            return PreflightResult.no_verdict("simulator unavailable")
        return PreflightResult.valid()

    def run_case(
        self,
        subject: SubjectRef,
        case: CaseSpec,
        output_dir: Path,
    ) -> CaseExecution:
        repeat = output_dir / "repeats" / "1000"
        repeat.mkdir(parents=True)
        (repeat / "episode.mp4").write_bytes(b"video")
        (repeat / "trajectory.json").write_text("{}")
        offset = case.resolved_object_position_offset_m()
        observation_bias = self.observation_bias_by_case.get(
            case.id,
            case.resolved_object_observation_bias_m(),
        )
        return CaseExecution(
            subject=subject,
            case_id=case.id,
            runner="mujoco",
            validity=ExecutionValidity.VALID,
            metrics={"placement_success": bool(self.values[case.id])},
            valid_units=1,
            unit_ids=["1000"],
            artifacts=[
                ArtifactRef(kind="video", path="repeats/1000/episode.mp4"),
                ArtifactRef(kind="trajectory", path="repeats/1000/trajectory.json"),
            ],
            environment={
                "runner": "mujoco",
                "model_sha256": "a" * 64,
                "controller_sha256": self.controller_by_case.get(case.id, "b" * 64),
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
                "object_observation_bias_x_m": observation_bias[0],
                "object_observation_bias_y_m": observation_bias[1],
                "object_observation_bias_z_m": observation_bias[2],
            },
        )


def test_absolute_gate_blocks_when_valid_required_scenario_fails(tmp_path: Path):
    result, root = execute_robustness_check(
        three_case_suite(),
        fixed_subject(),
        FakeRunner(
            {
                "nominal_center": True,
                "object_right_5cm": True,
                "perception_bias_5cm": False,
            }
        ),
        runs_root=tmp_path,
    )

    assert result.gate.verdict is GateVerdict.BLOCK
    assert result.gate.exit_code == 1
    assert [case.passed for case in result.cases] == [True, True, False]
    assert [case.object_position_offset_m[0] for case in result.cases] == [
        0.0,
        0.05,
        0.0,
    ]
    assert [case.object_observation_bias_m[0] for case in result.cases] == [
        0.0,
        0.0,
        0.05,
    ]
    assert result.cases[0].artifacts[0].path.startswith("cases/nominal_center/")
    assert (root / "result.json").is_file()
    assert (root / "cases/nominal_center/execution.json").is_file()


def test_absolute_gate_passes_only_when_every_required_scenario_passes(tmp_path: Path):
    result, _ = execute_robustness_check(
        three_case_suite(),
        fixed_subject(),
        FakeRunner(
            {
                "nominal_center": True,
                "object_right_5cm": True,
                "perception_bias_5cm": True,
            }
        ),
        runs_root=tmp_path,
    )

    assert result.gate.verdict is GateVerdict.PASS
    assert result.gate.exit_code == 0


def test_absolute_gate_returns_no_verdict_for_invalid_required_case(tmp_path: Path):
    result, _ = execute_robustness_check(
        three_case_suite(),
        fixed_subject(),
        FakeRunner(
            {
                "nominal_center": True,
                "object_right_5cm": ExecutionValidity.NO_VERDICT,
                "perception_bias_5cm": False,
            }
        ),
        runs_root=tmp_path,
    )

    assert result.gate.verdict is GateVerdict.NO_VERDICT
    assert result.gate.exit_code == 2
    assert result.cases[1].passed is None
    assert "simulator unavailable" in " ".join(result.cases[1].reasons)


def test_absolute_gate_rejects_cross_case_controller_identity_mismatch(tmp_path: Path):
    result, _ = execute_robustness_check(
        three_case_suite(),
        fixed_subject(),
        FakeRunner(
            {
                "nominal_center": True,
                "object_right_5cm": True,
                "perception_bias_5cm": False,
            },
            controller_by_case={"perception_bias_5cm": "d" * 64},
        ),
        runs_root=tmp_path,
    )

    assert result.gate.verdict is GateVerdict.NO_VERDICT
    assert result.gate.exit_code == 2
    assert "controller_sha256" in result.gate.summary


def test_absolute_gate_rejects_unapplied_perception_scenario(tmp_path: Path):
    result, _ = execute_robustness_check(
        three_case_suite(),
        fixed_subject(),
        FakeRunner(
            {
                "nominal_center": True,
                "object_right_5cm": True,
                "perception_bias_5cm": False,
            },
            observation_bias_by_case={
                "perception_bias_5cm": (0.0, 0.0, 0.0),
            },
        ),
        runs_root=tmp_path,
    )

    assert result.gate.verdict is GateVerdict.NO_VERDICT
    assert result.gate.exit_code == 2
    assert "object_observation_bias_x_m" in result.gate.summary
