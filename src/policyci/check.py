"""Runner-independent baseline/candidate execution, pairing, and release gate."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

from policyci.checkspec import CaseSpec, CheckSuite
from policyci.runner.base import Runner
from policyci.schemas import (
    ArtifactRef,
    CaseExecution,
    CheckResult,
    Comparison,
    EvidencePair,
    ExecutionValidity,
    GateResult,
    GateVerdict,
    MetricValue,
    SubjectRef,
    TaskStatus,
)
from policyci.store import (
    DEFAULT_RUNS_DIR,
    check_dir,
    new_check_id,
    save_check_result,
    write_json,
)


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "unit"


def _probe_video(path: Path) -> tuple[bool, str]:
    """Require a readable video stream and at least one frame."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return False, "ffprobe is unavailable"
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type,nb_read_frames",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if completed.returncode != 0:
        return False, (completed.stderr or f"ffprobe exit {completed.returncode}").strip()
    try:
        streams = json.loads(completed.stdout).get("streams", [])
        frames = int(streams[0].get("nb_read_frames", 0)) if streams else 0
    except (AttributeError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        frames = 0
    if frames < 1:
        return False, "no decodable video frames"
    return True, ""


def _portable_suite_payload(suite: CheckSuite) -> dict:
    payload = suite.model_dump(mode="json")
    for case, serialized in zip(suite.cases, payload["cases"]):
        if case.bag is not None:
            serialized["bag"] = case.bag.name
    return payload


def _no_verdict_execution(
    subject: SubjectRef,
    case: CaseSpec,
    runner_name: str,
    reasons: list[str],
) -> CaseExecution:
    return CaseExecution(
        subject=subject,
        case_id=case.id,
        runner=runner_name,
        validity=ExecutionValidity.NO_VERDICT,
        reasons=reasons,
    )


def _run_subject_case(
    runner: Runner,
    runner_name: str,
    subject: SubjectRef,
    case: CaseSpec,
    output_dir: Path,
) -> CaseExecution:
    try:
        preflight = runner.preflight(subject, case)
    except Exception as exc:
        return _no_verdict_execution(
            subject,
            case,
            runner_name,
            [f"preflight error: {str(exc)[:1000]}"],
        )

    if preflight.validity is not ExecutionValidity.VALID:
        return _no_verdict_execution(subject, case, runner_name, preflight.reasons)

    try:
        execution = runner.run_case(subject, case, output_dir)
    except Exception as exc:
        return _no_verdict_execution(
            subject,
            case,
            runner_name,
            [f"runner error: {str(exc)[:1000]}"],
        )

    contract_reasons = []
    if execution.subject != subject:
        contract_reasons.append("runner returned a different subject identity")
    if execution.case_id != case.id:
        contract_reasons.append("runner returned a different case identity")
    if contract_reasons:
        return _no_verdict_execution(subject, case, runner_name, contract_reasons)
    return _validate_execution_artifacts(execution, output_dir)


def _validate_execution_artifacts(
    execution: CaseExecution,
    output_dir: Path,
) -> CaseExecution:
    """A runner cannot claim VALID while pointing at absent evidence."""
    declared = {(artifact.kind, artifact.path) for artifact in execution.artifacts}
    for episode in execution.episodes:
        if episode.video_path:
            declared.add(("video", episode.video_path))

    missing = []
    empty = []
    corrupt = []
    for kind, relative in sorted(declared):
        try:
            portable = ArtifactRef(kind=kind, path=relative)
        except ValueError:
            missing.append(f"invalid artifact path {relative!r}")
            continue
        artifact_path = output_dir / portable.path
        if not artifact_path.is_file():
            missing.append(portable.path)
        elif artifact_path.stat().st_size == 0:
            empty.append(portable.path)
        elif kind == "video" and execution.runner == "libero":
            valid, detail = _probe_video(artifact_path)
            if not valid:
                corrupt.append(f"{portable.path} ({detail[:300]})")

    units_without_evidence = []
    if execution.validity is ExecutionValidity.VALID:
        for unit_id in execution.unit_ids:
            episode_evidence = any(
                str(episode.seed) == unit_id and episode.video_path
                for episode in execution.episodes
            )
            artifact_evidence = any(
                unit_id in Path(artifact.path).parts
                or Path(artifact.path).stem == unit_id
                for artifact in execution.artifacts
            )
            if not episode_evidence and not artifact_evidence:
                units_without_evidence.append(unit_id)

    if not missing and not empty and not corrupt and not units_without_evidence:
        return execution
    reasons = list(execution.reasons)
    if missing:
        reasons.append(f"missing declared artifacts: {', '.join(missing)}")
    if empty:
        reasons.append(f"empty declared artifacts: {', '.join(empty)}")
    if corrupt:
        reasons.append(f"corrupt declared videos: {', '.join(corrupt)}")
    if units_without_evidence:
        reasons.append(
            "missing evidence for valid units: " + ", ".join(units_without_evidence)
        )
    return execution.model_copy(
        update={
            "validity": ExecutionValidity.NO_VERDICT,
            "reasons": reasons,
        }
    )


def _save_execution(execution: CaseExecution, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "execution.json").write_text(execution.model_dump_json(indent=2))
    episodes_dir = directory / "episodes"
    episodes_dir.mkdir(exist_ok=True)
    for episode in execution.episodes:
        name = _safe_component(str(episode.seed))
        (episodes_dir / f"{name}.json").write_text(episode.model_dump_json(indent=2))


def _artifact_refs_for_unit(
    execution: CaseExecution,
    unit_id: str,
    side: str,
    case_id: str,
) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    for episode in execution.episodes:
        if str(episode.seed) != unit_id or not episode.video_path:
            continue
        refs.append(
            ArtifactRef(
                kind="video",
                path=f"{side}/cases/{case_id}/{episode.video_path}",
            )
        )
    seen = {(ref.kind, ref.path) for ref in refs}
    for artifact in execution.artifacts:
        if unit_id not in Path(artifact.path).parts:
            continue
        ref = ArtifactRef(
            kind=artifact.kind,
            path=f"{side}/cases/{case_id}/{artifact.path}",
        )
        if (ref.kind, ref.path) not in seen:
            refs.append(ref)
            seen.add((ref.kind, ref.path))
    return refs


def _make_pairs(
    case: CaseSpec,
    baseline: CaseExecution,
    candidate: CaseExecution,
) -> list[EvidencePair]:
    candidate_units = set(candidate.unit_ids)
    common_units = [unit_id for unit_id in baseline.unit_ids if unit_id in candidate_units]
    return [
        EvidencePair(
            case_id=case.id,
            unit_id=unit_id,
            baseline_artifacts=_artifact_refs_for_unit(
                baseline, unit_id, "baseline", case.id
            ),
            candidate_artifacts=_artifact_refs_for_unit(
                candidate, unit_id, "candidate", case.id
            ),
        )
        for unit_id in common_units
    ]


def _not_evaluated(
    case: CaseSpec,
    suite: CheckSuite,
    baseline: CaseExecution,
    candidate: CaseExecution,
    reason: str,
) -> Comparison:
    return Comparison(
        case_id=case.id,
        required=case.required,
        status=TaskStatus.NOT_EVALUATED,
        baseline_validity=baseline.validity,
        candidate_validity=candidate.validity,
        metric=suite.gate.metric,
        reason=reason,
        evidence_pairs=_make_pairs(case, baseline, candidate),
    )


def _as_number(value: MetricValue) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _compare_case(
    case: CaseSpec,
    suite: CheckSuite,
    baseline: CaseExecution,
    candidate: CaseExecution,
) -> Comparison:
    if baseline.validity is not ExecutionValidity.VALID:
        reason = "; ".join(baseline.reasons) or "baseline execution is not valid"
        return _not_evaluated(case, suite, baseline, candidate, reason)
    if candidate.validity is not ExecutionValidity.VALID:
        reason = "; ".join(candidate.reasons) or "candidate execution is not valid"
        return _not_evaluated(case, suite, baseline, candidate, reason)

    required_units = (
        suite.gate.min_valid_episodes
        if suite.runner.type == "libero"
        else suite.gate.min_valid_repeats
    )
    if baseline.valid_units < required_units or candidate.valid_units < required_units:
        return _not_evaluated(
            case,
            suite,
            baseline,
            candidate,
            (
                f"insufficient valid units: baseline={baseline.valid_units}, "
                f"candidate={candidate.valid_units}, required={required_units}"
            ),
        )

    if baseline.unit_ids != candidate.unit_ids:
        return _not_evaluated(
            case,
            suite,
            baseline,
            candidate,
            "baseline and candidate unit identities differ",
        )

    if baseline.environment != candidate.environment:
        return _not_evaluated(
            case,
            suite,
            baseline,
            candidate,
            "baseline and candidate environments differ",
        )

    metric = suite.gate.metric
    if metric not in baseline.metrics or metric not in candidate.metrics:
        return _not_evaluated(
            case,
            suite,
            baseline,
            candidate,
            f"configured metric {metric!r} is missing",
        )

    baseline_value = baseline.metrics[metric]
    candidate_value = candidate.metrics[metric]
    delta: float | None = None

    if suite.gate.mode == "max_drop":
        baseline_number = _as_number(baseline_value)
        candidate_number = _as_number(candidate_value)
        if baseline_number is None or candidate_number is None:
            return _not_evaluated(
                case,
                suite,
                baseline,
                candidate,
                f"metric {metric!r} must be finite numeric for max_drop",
            )
        delta = candidate_number - baseline_number
        if not math.isfinite(delta):
            return _not_evaluated(
                case,
                suite,
                baseline,
                candidate,
                f"metric {metric!r} produced a non-finite delta",
            )
        if delta < -suite.gate.max_drop_pp:
            status = TaskStatus.REGRESSED
        elif delta > suite.gate.max_drop_pp:
            status = TaskStatus.FIXED
        else:
            status = TaskStatus.UNCHANGED
    else:
        expected = suite.gate.expect
        if type(baseline_value) is not type(expected) or type(candidate_value) is not type(
            expected
        ):
            return _not_evaluated(
                case,
                suite,
                baseline,
                candidate,
                f"metric {metric!r} type must exactly match gate.expect",
            )
        candidate_meets = candidate_value == expected
        baseline_meets = baseline_value == expected
        if not candidate_meets:
            status = TaskStatus.REGRESSED
        elif not baseline_meets:
            status = TaskStatus.FIXED
        else:
            status = TaskStatus.UNCHANGED

    return Comparison(
        case_id=case.id,
        required=case.required,
        status=status,
        baseline_validity=baseline.validity,
        candidate_validity=candidate.validity,
        metric=metric,
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        delta=delta,
        evidence_pairs=_make_pairs(case, baseline, candidate),
    )


def _gate(comparisons: list[Comparison]) -> GateResult:
    required = [comparison for comparison in comparisons if comparison.required]
    regressions = [
        comparison for comparison in required if comparison.status is TaskStatus.REGRESSED
    ]
    if regressions:
        names = ", ".join(comparison.case_id for comparison in regressions)
        return GateResult(
            verdict=GateVerdict.BLOCK,
            exit_code=1,
            summary=f"BLOCK: {len(regressions)} required case(s) regressed: {names}",
        )

    incomplete = [
        comparison
        for comparison in required
        if comparison.status is TaskStatus.NOT_EVALUATED
    ]
    if incomplete:
        names = ", ".join(comparison.case_id for comparison in incomplete)
        return GateResult(
            verdict=GateVerdict.NO_VERDICT,
            exit_code=2,
            summary=f"NO_VERDICT: {len(incomplete)} required case(s) not evaluated: {names}",
        )

    return GateResult(
        verdict=GateVerdict.PASS,
        exit_code=0,
        summary="PASS: all required cases satisfy the configured gate",
    )


def execute_check(
    suite: CheckSuite,
    baseline: SubjectRef,
    candidate: SubjectRef,
    runner: Runner,
    *,
    runs_root: Path = DEFAULT_RUNS_DIR,
    check_id: str | None = None,
) -> tuple[CheckResult, Path]:
    """Execute and compare both subjects, then materialize one portable result tree."""

    resolved_check_id = check_id or new_check_id()
    root = check_dir(resolved_check_id, runs_root)
    root.mkdir(parents=True, exist_ok=False)

    check_payload = {
        "check_id": resolved_check_id,
        "status": "running",
        "suite": _portable_suite_payload(suite),
        "baseline": baseline.model_dump(mode="json"),
        "candidate": candidate.model_dump(mode="json"),
    }
    write_json(root / "check.json", check_payload)

    executions: dict[str, dict[str, CaseExecution]] = {
        "baseline": {},
        "candidate": {},
    }
    subjects = {"baseline": baseline, "candidate": candidate}
    for side, subject in subjects.items():
        manifest_cases = []
        for case in suite.cases:
            case_dir = root / side / "cases" / case.id
            execution = _run_subject_case(
                runner,
                suite.runner.type,
                subject,
                case,
                case_dir,
            )
            executions[side][case.id] = execution
            _save_execution(execution, case_dir)
            manifest_cases.append(
                {"id": case.id, "execution": f"cases/{case.id}/execution.json"}
            )
        write_json(
            root / side / "manifest.json",
            {"subject": subject.model_dump(mode="json"), "cases": manifest_cases},
        )

    comparisons = []
    for case in suite.cases:
        comparison = _compare_case(
            case,
            suite,
            executions["baseline"][case.id],
            executions["candidate"][case.id],
        )
        comparisons.append(comparison)
        for pair in comparison.evidence_pairs:
            pair_dir = root / "pairs" / case.id / _safe_component(pair.unit_id)
            pair_dir.mkdir(parents=True, exist_ok=True)
            (pair_dir / "pair.json").write_text(pair.model_dump_json(indent=2))

    result = CheckResult(
        check_id=resolved_check_id,
        suite_name=suite.name,
        runner=suite.runner.type,
        baseline=baseline,
        candidate=candidate,
        comparisons=comparisons,
        gate=_gate(comparisons),
        gate_config=suite.gate.model_dump(mode="json"),
        suite_version=suite.version,
        case_judges={case.id: case.judge for case in suite.cases},
    )
    save_check_result(result, runs_root)
    check_payload["status"] = result.gate.verdict.value
    write_json(root / "check.json", check_payload)
    return result, root
