"""Strict loading for the ALOHA open-loop replay evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from policyci.aloha_visualization import AlohaReplayVisualization
from policyci.schemas import (
    ArtifactRef,
    CheckResult,
    Comparison,
    EvidencePair,
    ExecutionValidity,
    GateResult,
    GateVerdict,
    ReplayTrace,
    SignalSeries,
    SubjectRef,
    TaskStatus,
)
from policyci.visualization import load_visualization


def _trace(*, candidate: bool) -> ReplayTrace:
    time_s = [0.0, 0.02, 0.04, 0.06]
    left_expected = [0.0, 0.1, 0.2, 0.3]
    right_expected = [0.0, 0.2, 0.4, 0.6]
    left_observed = [0.0, 0.0, 0.1, 0.2] if candidate else left_expected
    right_observed = [0.0, 0.0, 0.2, 0.4] if candidate else right_expected
    left_error = [
        abs(actual - expected)
        for expected, actual in zip(left_expected, left_observed, strict=True)
    ]
    right_error = [
        abs(actual - expected)
        for expected, actual in zip(right_expected, right_observed, strict=True)
    ]
    return ReplayTrace(
        trace_type="aloha_action",
        series={
            "time_s": SignalSeries(values=time_s, unit="s"),
            "left_gripper_expected": SignalSeries(values=left_expected),
            "left_gripper_observed": SignalSeries(values=left_observed),
            "right_gripper_expected": SignalSeries(values=right_expected),
            "right_gripper_observed": SignalSeries(values=right_observed),
            "left_gripper_error": SignalSeries(values=left_error),
            "right_gripper_error": SignalSeries(values=right_error),
            "max_gripper_error": SignalSeries(
                values=[
                    max(left, right)
                    for left, right in zip(left_error, right_error, strict=True)
                ]
            ),
            "max_arm_error": SignalSeries(values=[0.0] * len(time_s)),
        },
    )


def _write_check(
    tmp_path: Path,
    *,
    baseline_input: bytes = b"same-real-episode",
    candidate_input: bytes = b"same-real-episode",
) -> Path:
    root = tmp_path / "check_aloha"
    case_id = "aloha_coffee_replay"
    unit_id = "repeat-000"
    artifacts: dict[str, list[ArtifactRef]] = {}
    for side, input_bytes, candidate in (
        ("baseline", baseline_input, False),
        ("candidate", candidate_input, True),
    ):
        prefix = f"{side}/cases/{case_id}/repeats/{unit_id}"
        input_relative = f"{prefix}/input.mcap"
        output_relative = f"{prefix}/capture/capture_0.mcap"
        trace_relative = f"{prefix}/trace.json"
        for relative, contents in (
            (input_relative, input_bytes),
            (output_relative, f"{side}-output".encode()),
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        trace_path = root / trace_relative
        trace_path.write_text(_trace(candidate=candidate).model_dump_json(indent=2))
        artifacts[side] = [
            ArtifactRef(kind="input_mcap", path=input_relative),
            ArtifactRef(kind="mcap", path=output_relative),
            ArtifactRef(kind="trace", path=trace_relative),
        ]

    result = CheckResult(
        check_id="check_aloha",
        suite_name="aloha-coffee-replay",
        runner="ros2_replay",
        baseline=SubjectRef(
            id="aloha-action-baseline",
            config={"gripper_delay_messages": 0},
        ),
        candidate=SubjectRef(
            id="aloha-gripper-delay-300ms",
            config={"gripper_delay_messages": 15},
        ),
        comparisons=[
            Comparison(
                case_id=case_id,
                status=TaskStatus.REGRESSED,
                baseline_validity=ExecutionValidity.VALID,
                candidate_validity=ExecutionValidity.VALID,
                metric="within_tolerance",
                baseline_value=True,
                candidate_value=False,
                evidence_pairs=[
                    EvidencePair(
                        case_id=case_id,
                        unit_id=unit_id,
                        baseline_artifacts=artifacts["baseline"],
                        candidate_artifacts=artifacts["candidate"],
                    )
                ],
            )
        ],
        gate=GateResult(
            verdict=GateVerdict.BLOCK,
            exit_code=1,
            summary="BLOCK: candidate regressed on aloha_coffee_replay",
        ),
        gate_config={
            "metric": "within_tolerance",
            "mode": "require_value",
            "expect": True,
        },
        case_judges={
            case_id: {
                "sample_rate_hz": 50.0,
                "gripper_tolerance": 0.05,
                "arm_tolerance": 1e-9,
            }
        },
    )
    root.mkdir(exist_ok=True)
    (root / "result.json").write_text(result.model_dump_json(indent=2))
    return root


def test_load_aloha_visualization_reports_honest_open_loop_fault(
    tmp_path: Path,
) -> None:
    model = load_visualization(_write_check(tmp_path))

    assert isinstance(model, AlohaReplayVisualization)
    assert model.first_candidate_violation_index == 1
    assert model.first_candidate_violation_time_s == pytest.approx(0.02)
    assert model.baseline_unit_pass is True
    assert model.candidate_unit_pass is False
    assert model.mcap_paths == []
    assert "OPEN-LOOP COMPONENT REGRESSION" in model.summary_markdown
    assert "300 ms GRIPPER COMMAND DELAY" in model.summary_markdown
    assert "does not show a counterfactual physical outcome" in model.summary_markdown


def test_load_aloha_visualization_requires_identical_input_mcaps(
    tmp_path: Path,
) -> None:
    root = _write_check(
        tmp_path,
        baseline_input=b"episode-a",
        candidate_input=b"episode-b",
    )

    with pytest.raises(ValueError, match="input MCAP hashes differ"):
        load_visualization(root)


def test_load_aloha_visualization_rejects_error_not_derived_from_commands(
    tmp_path: Path,
) -> None:
    root = _write_check(tmp_path)
    trace_path = next(root.rglob("candidate/**/trace.json"))
    payload = json.loads(trace_path.read_text())
    payload["series"]["left_gripper_error"]["values"][1] = 0.0
    trace_path.write_text(json.dumps(payload, indent=2))

    with pytest.raises(ValueError, match="left_gripper_error"):
        load_visualization(root)
