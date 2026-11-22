"""Pure selection and validation for paired RoboLens replay visualizations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from policyci.schemas import (
    ArtifactRef,
    CheckResult,
    Comparison,
    EvidencePair,
    GateVerdict,
    MetricValue,
    ReplayTrace,
    TaskStatus,
)

if TYPE_CHECKING:
    from policyci.aloha_visualization import AlohaReplayVisualization
    from policyci.manipulation_visualization import ManipulationVisualization


class CheckVisualization(BaseModel):
    """Validated, Rerun-independent data for one paired replay unit."""

    model_config = ConfigDict(extra="forbid")

    check_dir: Path
    check_id: str
    suite_name: str
    case_id: str
    unit_id: str
    baseline_id: str
    candidate_id: str
    verdict: GateVerdict
    gate_summary: str
    comparison_status: TaskStatus
    metric: str
    baseline_value: MetricValue | None
    candidate_value: MetricValue | None
    judge: dict[str, MetricValue]
    tolerance: float
    timeline: str
    expected: list[float]
    baseline_observed: list[float]
    candidate_observed: list[float]
    baseline_error: list[float]
    candidate_error: list[float]
    baseline_unit_within_tolerance: bool
    candidate_unit_within_tolerance: bool
    first_candidate_violation_index: int | None = None
    unit: str
    baseline_mcaps: list[Path]
    candidate_mcaps: list[Path]
    summary_markdown: str

    @property
    def mcap_paths(self) -> list[Path]:
        return [*self.baseline_mcaps, *self.candidate_mcaps]

    @property
    def baseline_unit_pass(self) -> bool:
        return self.baseline_unit_within_tolerance

    @property
    def candidate_unit_pass(self) -> bool:
        return self.candidate_unit_within_tolerance


def _select_comparison(
    result: CheckResult,
    case_id: str | None,
) -> Comparison:
    if case_id is not None:
        selected = next(
            (comparison for comparison in result.comparisons if comparison.case_id == case_id),
            None,
        )
        if selected is None:
            raise ValueError(f"unknown visualization case: {case_id}")
        return selected
    changed = next(
        (
            comparison
            for comparison in result.comparisons
            if comparison.status in {TaskStatus.REGRESSED, TaskStatus.FIXED}
        ),
        None,
    )
    if changed is not None:
        return changed
    if not result.comparisons:
        raise ValueError("check result contains no cases")
    return result.comparisons[0]


def _select_pair(comparison: Comparison, unit_id: str | None) -> EvidencePair:
    if unit_id is not None:
        selected = next(
            (pair for pair in comparison.evidence_pairs if pair.unit_id == unit_id),
            None,
        )
        if selected is None:
            raise ValueError(
                f"unknown visualization repeat {unit_id!r} for case {comparison.case_id!r}"
            )
        return selected
    if not comparison.evidence_pairs:
        raise ValueError(f"case {comparison.case_id!r} contains no evidence repeats")
    return comparison.evidence_pairs[0]


def _resolve_artifact(check_dir: Path, artifact: ArtifactRef) -> Path:
    root = check_dir.resolve()
    path = (root / artifact.path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"artifact is not confined to the check directory: {artifact.path}")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty {artifact.kind} artifact: {artifact.path}")
    return path


def _load_side_evidence(
    check_dir: Path,
    artifacts: list[ArtifactRef],
    *,
    side: str,
) -> tuple[ReplayTrace, list[Path]]:
    trace_refs = [artifact for artifact in artifacts if artifact.kind == "trace"]
    if len(trace_refs) != 1:
        raise ValueError(f"{side} evidence must contain exactly one trace artifact")
    mcap_refs = [artifact for artifact in artifacts if artifact.kind == "mcap"]
    if not mcap_refs:
        raise ValueError(f"{side} evidence must contain at least one MCAP artifact")

    trace_path = _resolve_artifact(check_dir, trace_refs[0])
    try:
        trace = ReplayTrace.model_validate_json(trace_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid {side} trace artifact: {exc}") from exc
    mcaps = [_resolve_artifact(check_dir, artifact) for artifact in mcap_refs]
    return trace, mcaps


def _required_signal(trace: ReplayTrace, name: str, side: str) -> list[float]:
    signal = trace.series.get(name)
    if signal is None:
        raise ValueError(f"{side} trace is missing required {name!r} signal")
    return list(signal.values)


@dataclass(frozen=True)
class _PairTraceEvidence:
    timeline: str
    expected: list[float]
    baseline_observed: list[float]
    candidate_observed: list[float]
    baseline_error: list[float]
    candidate_error: list[float]
    unit: str
    baseline_mcaps: list[Path]
    candidate_mcaps: list[Path]


def _validate_absolute_error(
    *,
    side: str,
    expected: list[float],
    observed: list[float],
    errors: list[float],
) -> None:
    for index, (target, actual, stored_error) in enumerate(
        zip(expected, observed, errors)
    ):
        computed_error = abs(actual - target)
        if not math.isclose(
            stored_error,
            computed_error,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{side} absolute_error[{index}] does not match the exact samples"
            )


def _load_pair_trace_evidence(
    check_dir: Path,
    pair: EvidencePair,
) -> _PairTraceEvidence:
    baseline_trace, baseline_mcaps = _load_side_evidence(
        check_dir,
        pair.baseline_artifacts,
        side="baseline",
    )
    candidate_trace, candidate_mcaps = _load_side_evidence(
        check_dir,
        pair.candidate_artifacts,
        side="candidate",
    )

    if baseline_trace.timeline != candidate_trace.timeline:
        raise ValueError("baseline and candidate trace timelines differ")
    expected = _required_signal(baseline_trace, "expected", "baseline")
    candidate_expected = _required_signal(candidate_trace, "expected", "candidate")
    if expected != candidate_expected:
        raise ValueError("baseline and candidate expected signals differ")
    baseline_observed = _required_signal(baseline_trace, "observed", "baseline")
    candidate_observed = _required_signal(candidate_trace, "observed", "candidate")
    baseline_error = _required_signal(baseline_trace, "absolute_error", "baseline")
    candidate_error = _required_signal(candidate_trace, "absolute_error", "candidate")
    lengths = {
        len(expected),
        len(baseline_observed),
        len(candidate_observed),
        len(baseline_error),
        len(candidate_error),
    }
    if len(lengths) != 1:
        raise ValueError("baseline and candidate trace sample counts differ")
    _validate_absolute_error(
        side="baseline",
        expected=expected,
        observed=baseline_observed,
        errors=baseline_error,
    )
    _validate_absolute_error(
        side="candidate",
        expected=expected,
        observed=candidate_observed,
        errors=candidate_error,
    )

    return _PairTraceEvidence(
        timeline=baseline_trace.timeline,
        expected=expected,
        baseline_observed=baseline_observed,
        candidate_observed=candidate_observed,
        baseline_error=baseline_error,
        candidate_error=candidate_error,
        unit=baseline_trace.series["observed"].unit,
        baseline_mcaps=baseline_mcaps,
        candidate_mcaps=candidate_mcaps,
    )


def _trace_metric(
    metric: str,
    errors: list[float],
    tolerance: float,
) -> MetricValue | None:
    if metric == "within_tolerance":
        return all(error <= tolerance for error in errors)
    if metric == "mean_abs_error":
        return sum(errors) / len(errors)
    if metric == "max_abs_error":
        return max(errors)
    if metric == "message_count":
        return len(errors)
    return None


def first_violation_index(
    errors: list[float],
    *,
    tolerance: float,
) -> int | None:
    return next(
        (index for index, error in enumerate(errors) if error > tolerance),
        None,
    )


def _aggregate_trace_metrics(values: list[MetricValue]) -> MetricValue:
    first = values[0]
    if isinstance(first, bool):
        return all(bool(value) for value in values)
    return sum(float(value) for value in values) / len(values)


def _validate_automatic_metric(
    *,
    side: str,
    stored: MetricValue | None,
    derived: MetricValue,
) -> None:
    matches = False
    if isinstance(derived, bool):
        matches = isinstance(stored, bool) and stored is derived
    elif isinstance(stored, (int, float)) and not isinstance(stored, bool):
        matches = math.isclose(
            float(stored),
            float(derived),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    if not matches:
        raise ValueError(
            f"{side} automatic metric does not match the exact replay traces"
        )


def _summary_markdown(
    result: CheckResult,
    comparison: Comparison,
    pair: EvidencePair,
    tolerance: float,
    baseline_unit_within_tolerance: bool,
    candidate_unit_within_tolerance: bool,
    first_candidate_violation: int | None,
) -> str:
    baseline_unit = "PASS" if baseline_unit_within_tolerance else "FAIL"
    candidate_unit = "PASS" if candidate_unit_within_tolerance else "FAIL"
    lines = [
            f"# {result.gate.verdict.value.upper()} — {comparison.status.value.upper()}",
            "",
            f"**Case / repeat:** `{comparison.case_id}` / `{pair.unit_id}`  ",
            f"**Subjects:** `{result.baseline.id}` → `{result.candidate.id}`  ",
            f"**Aggregate automatic metric:** `{comparison.metric}` = "
            f"baseline `{comparison.baseline_value}`, candidate `{comparison.candidate_value}`  ",
            f"**Selected repeat from exact samples:** baseline `{baseline_unit}`, "
            f"candidate `{candidate_unit}`  ",
            f"**Immutable tolerance:** `{tolerance}`  ",
    ]
    if first_candidate_violation is not None:
        lines.append(
            f"**First candidate violation:** sample `{first_candidate_violation}`  "
        )
    lines.extend(
        [
            "",
            f"RoboLens automatically evaluated the metric and gate: {result.gate.summary}",
            "Rerun displays the paired evidence; it does not decide the verdict.",
        ]
    )
    return "\n".join(lines)


def load_check_visualization(
    check_dir: Path,
    *,
    case_id: str | None = None,
    unit_id: str | None = None,
) -> CheckVisualization:
    """Load one sound evidence pair without importing Rerun."""
    root = check_dir.resolve()
    result_path = root / "result.json"
    if not result_path.is_file():
        raise ValueError(f"check result is missing: {result_path}")
    try:
        result = CheckResult.model_validate_json(result_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid check result: {exc}") from exc

    comparison = _select_comparison(result, case_id)
    pair = _select_pair(comparison, unit_id)
    judge = result.case_judges.get(comparison.case_id, {})
    raw_tolerance = judge.get("tolerance")
    if (
        isinstance(raw_tolerance, bool)
        or not isinstance(raw_tolerance, (int, float))
        or not math.isfinite(float(raw_tolerance))
        or float(raw_tolerance) < 0
    ):
        raise ValueError(
            f"case {comparison.case_id!r} requires a finite non-negative tolerance"
        )
    tolerance = float(raw_tolerance)

    selected_evidence: _PairTraceEvidence | None = None
    baseline_derived: list[MetricValue] = []
    candidate_derived: list[MetricValue] = []
    for evidence_pair in comparison.evidence_pairs:
        evidence = _load_pair_trace_evidence(root, evidence_pair)
        if evidence_pair is pair:
            selected_evidence = evidence
        baseline_metric = _trace_metric(
            comparison.metric,
            evidence.baseline_error,
            tolerance,
        )
        candidate_metric = _trace_metric(
            comparison.metric,
            evidence.candidate_error,
            tolerance,
        )
        if baseline_metric is not None and candidate_metric is not None:
            baseline_derived.append(baseline_metric)
            candidate_derived.append(candidate_metric)

    if selected_evidence is None:
        raise ValueError(f"selected evidence repeat is unavailable: {pair.unit_id}")
    if baseline_derived and candidate_derived:
        _validate_automatic_metric(
            side="baseline",
            stored=comparison.baseline_value,
            derived=_aggregate_trace_metrics(baseline_derived),
        )
        _validate_automatic_metric(
            side="candidate",
            stored=comparison.candidate_value,
            derived=_aggregate_trace_metrics(candidate_derived),
        )

    baseline_unit_within_tolerance = all(
        error <= tolerance for error in selected_evidence.baseline_error
    )
    candidate_unit_within_tolerance = all(
        error <= tolerance for error in selected_evidence.candidate_error
    )
    first_candidate_violation = first_violation_index(
        selected_evidence.candidate_error,
        tolerance=tolerance,
    )

    return CheckVisualization(
        check_dir=root,
        check_id=result.check_id,
        suite_name=result.suite_name,
        case_id=comparison.case_id,
        unit_id=pair.unit_id,
        baseline_id=result.baseline.id,
        candidate_id=result.candidate.id,
        verdict=result.gate.verdict,
        gate_summary=result.gate.summary,
        comparison_status=comparison.status,
        metric=comparison.metric,
        baseline_value=comparison.baseline_value,
        candidate_value=comparison.candidate_value,
        judge=judge,
        tolerance=tolerance,
        timeline=selected_evidence.timeline,
        expected=selected_evidence.expected,
        baseline_observed=selected_evidence.baseline_observed,
        candidate_observed=selected_evidence.candidate_observed,
        baseline_error=selected_evidence.baseline_error,
        candidate_error=selected_evidence.candidate_error,
        baseline_unit_within_tolerance=baseline_unit_within_tolerance,
        candidate_unit_within_tolerance=candidate_unit_within_tolerance,
        first_candidate_violation_index=first_candidate_violation,
        unit=selected_evidence.unit,
        baseline_mcaps=selected_evidence.baseline_mcaps,
        candidate_mcaps=selected_evidence.candidate_mcaps,
        summary_markdown=_summary_markdown(
            result,
            comparison,
            pair,
            tolerance,
            baseline_unit_within_tolerance,
            candidate_unit_within_tolerance,
            first_candidate_violation,
        ),
    )


def load_visualization(
    check_dir: Path,
    *,
    case_id: str | None = None,
    unit_id: str | None = None,
) -> CheckVisualization | AlohaReplayVisualization | ManipulationVisualization:
    """Dispatch a completed check to its runner-specific evidence loader."""
    root = check_dir.resolve()
    result_path = root / "result.json"
    if not result_path.is_file():
        raise ValueError(f"check result is missing: {result_path}")
    try:
        result = CheckResult.model_validate_json(result_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid check result: {exc}") from exc

    if result.runner == "ros2_replay":
        comparison = _select_comparison(result, case_id)
        pair = _select_pair(comparison, unit_id)
        trace_refs = [
            artifact
            for artifact in pair.baseline_artifacts
            if artifact.kind == "trace"
        ]
        if len(trace_refs) != 1:
            raise ValueError("baseline evidence must contain exactly one trace artifact")
        trace_path = _resolve_artifact(root, trace_refs[0])
        try:
            trace = ReplayTrace.model_validate_json(trace_path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid baseline trace artifact: {exc}") from exc
        if trace.trace_type == "aloha_action":
            from policyci.aloha_visualization import load_aloha_visualization

            return load_aloha_visualization(
                root,
                case_id=case_id,
                unit_id=unit_id,
            )
        return load_check_visualization(root, case_id=case_id, unit_id=unit_id)
    if result.runner == "mujoco":
        from policyci.manipulation_visualization import (
            load_manipulation_visualization,
        )

        return load_manipulation_visualization(
            root,
            case_id=case_id,
            unit_id=unit_id,
        )
    raise ValueError(f"unsupported visualization runner: {result.runner!r}")
