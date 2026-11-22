"""Single-subject absolute gates over declared robot simulation scenarios."""

from __future__ import annotations

import math
from pathlib import Path

from policyci.checkspec import CaseSpec, CheckSuite
from policyci.runner.base import Runner
from policyci.schemas import (
    ArtifactRef,
    CaseExecution,
    ExecutionValidity,
    GateResult,
    GateVerdict,
    MetricValue,
    RobustnessCaseResult,
    RobustnessCheckResult,
    SubjectRef,
)
from policyci.store import check_dir, new_check_id


_IDENTITY_KEYS = (
    "model_sha256",
    "controller_sha256",
    "adapter_sha256",
    "camera",
    "camera_fovy_deg",
    "control_hz",
    "horizon",
    "task",
    "robosuite_version",
    "mujoco_version",
)


def _invalid_execution(
    subject: SubjectRef,
    case: CaseSpec,
    reasons: list[str],
) -> CaseExecution:
    return CaseExecution(
        subject=subject,
        case_id=case.id,
        runner="mujoco",
        validity=ExecutionValidity.NO_VERDICT,
        reasons=reasons,
    )


def _run_case(
    runner: Runner,
    subject: SubjectRef,
    case: CaseSpec,
    output_dir: Path,
) -> CaseExecution:
    try:
        preflight = runner.preflight(subject, case)
    except Exception as exc:
        return _invalid_execution(subject, case, [f"preflight error: {str(exc)[:1000]}"])
    if preflight.validity is not ExecutionValidity.VALID:
        return _invalid_execution(subject, case, list(preflight.reasons))
    try:
        execution = runner.run_case(subject, case, output_dir)
    except Exception as exc:
        return _invalid_execution(subject, case, [f"runner error: {str(exc)[:1000]}"])
    reasons: list[str] = []
    if execution.subject != subject:
        reasons.append("runner returned a different subject identity")
    if execution.case_id != case.id:
        reasons.append("runner returned a different case identity")
    if execution.runner != "mujoco":
        reasons.append(f"runner returned unexpected identity {execution.runner!r}")
    for artifact in execution.artifacts:
        resolved = (output_dir / artifact.path).resolve()
        if not resolved.is_relative_to(output_dir.resolve()):
            reasons.append(f"artifact escapes case directory: {artifact.path}")
        elif not resolved.is_file() or resolved.stat().st_size == 0:
            reasons.append(f"artifact is missing or empty: {artifact.path}")
    if reasons:
        return execution.model_copy(
            update={
                "validity": ExecutionValidity.NO_VERDICT,
                "reasons": [*execution.reasons, *reasons],
            }
        )
    return execution


def _metric_matches(value: MetricValue, expected: MetricValue) -> bool:
    if isinstance(expected, bool):
        return isinstance(value, bool) and value is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and math.isclose(float(value), float(expected), rel_tol=1e-9, abs_tol=1e-12)
        )
    return type(value) is type(expected) and value == expected


def _scenario_runtime_reasons(
    case: CaseSpec,
    execution: CaseExecution,
) -> list[str]:
    expected = case.resolved_object_position_offset_m()
    names = ("object_offset_x_m", "object_offset_y_m", "object_offset_z_m")
    reasons: list[str] = []
    for name, wanted in zip(names, expected):
        observed = execution.environment.get(name)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or not math.isclose(float(observed), wanted, rel_tol=0.0, abs_tol=1e-9)
        ):
            reasons.append(
                f"{case.id}: {name} does not match declared scenario "
                f"(expected={wanted}, observed={observed!r})"
            )
    expected_bias = case.resolved_object_observation_bias_m()
    bias_names = (
        "object_observation_bias_x_m",
        "object_observation_bias_y_m",
        "object_observation_bias_z_m",
    )
    for name, wanted in zip(bias_names, expected_bias):
        observed = execution.environment.get(name)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or not math.isclose(float(observed), wanted, rel_tol=0.0, abs_tol=1e-9)
        ):
            reasons.append(
                f"{case.id}: {name} does not match declared scenario "
                f"(expected={wanted}, observed={observed!r})"
            )
    return reasons


def _identity_reasons(cases: list[RobustnessCaseResult]) -> list[str]:
    valid = [case for case in cases if case.validity is ExecutionValidity.VALID]
    if not valid:
        return []
    reasons: list[str] = []
    reference = valid[0]
    for key in _IDENTITY_KEYS:
        expected = reference.environment.get(key)
        if expected is None:
            reasons.append(f"{reference.case_id}: missing runtime identity {key}")
            continue
        for case in valid[1:]:
            observed = case.environment.get(key)
            if observed != expected:
                reasons.append(
                    f"runtime identity mismatch for {key}: "
                    f"{reference.case_id}={expected!r}, {case.case_id}={observed!r}"
                )
    return reasons


def _gate(
    cases: list[RobustnessCaseResult],
    *,
    identity_reasons: list[str],
) -> GateResult:
    required = [case for case in cases if case.required]
    invalid = [case for case in required if case.validity is not ExecutionValidity.VALID]
    if invalid or identity_reasons:
        details = [case.case_id for case in invalid]
        details.extend(identity_reasons)
        return GateResult(
            verdict=GateVerdict.NO_VERDICT,
            exit_code=2,
            summary="NO_VERDICT: " + "; ".join(details),
        )
    failed = [case for case in required if case.passed is False]
    if failed:
        names = ", ".join(case.case_id for case in failed)
        return GateResult(
            verdict=GateVerdict.BLOCK,
            exit_code=1,
            summary=(
                f"BLOCKED: {len(failed)}/{len(required)} required scenario(s) failed: "
                f"{names}"
            ),
        )
    return GateResult(
        verdict=GateVerdict.PASS,
        exit_code=0,
        summary=f"PASS: all {len(required)} required scenarios passed",
    )


def execute_robustness_check(
    suite: CheckSuite,
    subject: SubjectRef,
    runner: Runner,
    *,
    runs_root: Path,
) -> tuple[RobustnessCheckResult, Path]:
    """Run one subject over every case and emit an absolute suite verdict."""

    if suite.runner.type != "mujoco":
        raise ValueError("robustness-check currently requires runner.type=mujoco")
    if suite.gate.mode != "require_value" or suite.gate.expect is None:
        raise ValueError("robustness-check requires an absolute require_value gate")

    result_root = check_dir(new_check_id(), runs_root)
    result_root.mkdir(parents=True, exist_ok=False)
    cases: list[RobustnessCaseResult] = []
    scenario_reasons: list[str] = []
    for case in suite.cases:
        case_dir = result_root / "cases" / case.id
        case_dir.mkdir(parents=True, exist_ok=True)
        execution = _run_case(runner, subject, case, case_dir)
        (case_dir / "execution.json").write_text(execution.model_dump_json(indent=2))
        reasons = list(execution.reasons)
        if execution.validity is ExecutionValidity.VALID:
            applied_reasons = _scenario_runtime_reasons(case, execution)
            if applied_reasons:
                scenario_reasons.extend(applied_reasons)
                reasons.extend(applied_reasons)
                execution = execution.model_copy(
                    update={
                        "validity": ExecutionValidity.NO_VERDICT,
                        "reasons": reasons,
                    }
                )
                (case_dir / "execution.json").write_text(
                    execution.model_dump_json(indent=2)
                )
        value = execution.metrics.get(suite.gate.metric)
        if execution.validity is ExecutionValidity.VALID and value is None:
            reasons.append(f"missing gate metric {suite.gate.metric!r}")
            execution = execution.model_copy(
                update={
                    "validity": ExecutionValidity.NO_VERDICT,
                    "reasons": reasons,
                }
            )
            (case_dir / "execution.json").write_text(execution.model_dump_json(indent=2))
        passed = (
            _metric_matches(value, suite.gate.expect)
            if execution.validity is ExecutionValidity.VALID and value is not None
            else None
        )
        artifacts = [
            ArtifactRef(
                kind=artifact.kind,
                path=f"cases/{case.id}/{artifact.path}",
            )
            for artifact in execution.artifacts
        ]
        cases.append(
            RobustnessCaseResult(
                case_id=case.id,
                required=case.required,
                object_position_offset_m=case.resolved_object_position_offset_m(),
                object_observation_bias_m=(
                    case.resolved_object_observation_bias_m()
                ),
                validity=execution.validity,
                metric=suite.gate.metric,
                metric_value=value,
                passed=passed,
                reasons=list(execution.reasons),
                metrics=dict(execution.metrics),
                artifacts=artifacts,
                environment=dict(execution.environment),
            )
        )

    identity_reasons = [*scenario_reasons, *_identity_reasons(cases)]
    gate = _gate(cases, identity_reasons=identity_reasons)
    result = RobustnessCheckResult(
        check_id=result_root.name,
        suite_name=suite.name,
        runner="mujoco",
        subject=subject,
        metric=suite.gate.metric,
        cases=cases,
        gate=gate,
        gate_config=suite.gate.model_dump(mode="json"),
        case_judges={case.id: dict(case.judge) for case in suite.cases},
    )
    (result_root / "result.json").write_text(result.model_dump_json(indent=2))
    return result, result_root
