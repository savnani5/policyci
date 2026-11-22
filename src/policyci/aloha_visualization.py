"""Strict evidence loading for the ALOHA open-loop component regression."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

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
from policyci.visualization import (
    _resolve_artifact,
    _select_comparison,
    _select_pair,
    _validate_automatic_metric,
)


class AlohaReplayVisualization(BaseModel):
    """Validated signals and immutable context for one ALOHA replay pair."""

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
    input_mcap: Path
    baseline_output_mcap: Path
    candidate_output_mcap: Path
    input_sha256: str
    time_s: list[float]
    left_gripper_expected: list[float]
    right_gripper_expected: list[float]
    baseline_left_gripper_observed: list[float]
    baseline_right_gripper_observed: list[float]
    candidate_left_gripper_observed: list[float]
    candidate_right_gripper_observed: list[float]
    baseline_left_gripper_error: list[float]
    baseline_right_gripper_error: list[float]
    candidate_left_gripper_error: list[float]
    candidate_right_gripper_error: list[float]
    baseline_max_gripper_error: list[float]
    candidate_max_gripper_error: list[float]
    baseline_max_arm_error: list[float]
    candidate_max_arm_error: list[float]
    gripper_tolerance: float
    arm_tolerance: float
    first_candidate_violation_index: int
    first_candidate_violation_time_s: float
    baseline_unit_within_tolerance: bool
    candidate_unit_within_tolerance: bool
    summary_markdown: str

    @property
    def mcap_paths(self) -> list[Path]:
        """Raw MCAPs stay inside the self-contained RRD instead of opening as tabs."""
        return []

    @property
    def baseline_unit_pass(self) -> bool:
        return self.baseline_unit_within_tolerance

    @property
    def candidate_unit_pass(self) -> bool:
        return self.candidate_unit_within_tolerance

    @property
    def timeline(self) -> str:
        return "episode_time"


@dataclass(frozen=True)
class _SideEvidence:
    trace: ReplayTrace
    input_mcap: Path
    output_mcap: Path
    input_sha256: str


@dataclass(frozen=True)
class _PairEvidence:
    baseline: _SideEvidence
    candidate: _SideEvidence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _single_artifact(
    check_dir: Path,
    artifacts: list[ArtifactRef],
    *,
    kind: str,
    side: str,
) -> Path:
    refs = [artifact for artifact in artifacts if artifact.kind == kind]
    if len(refs) != 1:
        raise ValueError(f"{side} evidence must contain exactly one {kind} artifact")
    return _resolve_artifact(check_dir, refs[0])


def _load_side(
    check_dir: Path,
    artifacts: list[ArtifactRef],
    *,
    side: str,
) -> _SideEvidence:
    trace_path = _single_artifact(check_dir, artifacts, kind="trace", side=side)
    input_mcap = _single_artifact(
        check_dir,
        artifacts,
        kind="input_mcap",
        side=side,
    )
    output_mcap = _single_artifact(check_dir, artifacts, kind="mcap", side=side)
    try:
        trace = ReplayTrace.model_validate_json(trace_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid {side} trace artifact: {exc}") from exc
    if trace.trace_type != "aloha_action":
        raise ValueError(f"{side} trace is not an ALOHA action trace")
    return _SideEvidence(
        trace=trace,
        input_mcap=input_mcap,
        output_mcap=output_mcap,
        input_sha256=_sha256(input_mcap),
    )


def _load_pair(check_dir: Path, pair: EvidencePair) -> _PairEvidence:
    evidence = _PairEvidence(
        baseline=_load_side(
            check_dir,
            pair.baseline_artifacts,
            side="baseline",
        ),
        candidate=_load_side(
            check_dir,
            pair.candidate_artifacts,
            side="candidate",
        ),
    )
    if evidence.baseline.input_sha256 != evidence.candidate.input_sha256:
        raise ValueError("baseline and candidate input MCAP hashes differ")
    return evidence


def _required(trace: ReplayTrace, name: str, *, side: str) -> list[float]:
    signal = trace.series.get(name)
    if signal is None:
        raise ValueError(f"{side} trace is missing required {name!r} signal")
    return [float(value) for value in signal.values]


def _finite_nonnegative_judge(judge: dict[str, MetricValue], name: str) -> float:
    value = judge.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"ALOHA visualization requires finite non-negative {name}")
    return float(value)


def _assert_series_matches(
    *,
    side: str,
    name: str,
    stored: list[float],
    derived: list[float],
) -> None:
    if len(stored) != len(derived):
        raise ValueError(f"{side} {name} sample count differs")
    for index, (actual, expected) in enumerate(zip(stored, derived, strict=True)):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{side} {name}[{index}] does not match command samples")


def _validate_trace_pair(
    evidence: _PairEvidence,
) -> dict[str, list[float]]:
    required_names = (
        "time_s",
        "left_gripper_expected",
        "left_gripper_observed",
        "right_gripper_expected",
        "right_gripper_observed",
        "left_gripper_error",
        "right_gripper_error",
        "max_gripper_error",
        "max_arm_error",
    )
    values: dict[str, list[float]] = {}
    for side, trace in (
        ("baseline", evidence.baseline.trace),
        ("candidate", evidence.candidate.trace),
    ):
        for name in required_names:
            series = _required(trace, name, side=side)
            if not series or any(not math.isfinite(value) for value in series):
                raise ValueError(f"{side} {name} must contain finite samples")
            values[f"{side}_{name}"] = series
        lengths = {len(values[f"{side}_{name}"]) for name in required_names}
        if len(lengths) != 1:
            raise ValueError(f"{side} ALOHA trace signal lengths differ")

    baseline_time = values["baseline_time_s"]
    candidate_time = values["candidate_time_s"]
    if baseline_time != candidate_time:
        raise ValueError("baseline and candidate time_s signals differ")
    if any(
        current <= previous
        for previous, current in zip(baseline_time, baseline_time[1:])
    ):
        raise ValueError("ALOHA trace time_s must be strictly increasing")

    for gripper in ("left", "right"):
        expected_name = f"{gripper}_gripper_expected"
        baseline_expected = values[f"baseline_{expected_name}"]
        candidate_expected = values[f"candidate_{expected_name}"]
        if baseline_expected != candidate_expected:
            raise ValueError(
                f"baseline and candidate {gripper} gripper expected signals differ"
            )
        for side in ("baseline", "candidate"):
            observed = values[f"{side}_{gripper}_gripper_observed"]
            derived_error = [
                abs(actual - target)
                for target, actual in zip(
                    baseline_expected,
                    observed,
                    strict=True,
                )
            ]
            _assert_series_matches(
                side=side,
                name=f"{gripper}_gripper_error",
                stored=values[f"{side}_{gripper}_gripper_error"],
                derived=derived_error,
            )

    for side in ("baseline", "candidate"):
        derived_max = [
            max(left, right)
            for left, right in zip(
                values[f"{side}_left_gripper_error"],
                values[f"{side}_right_gripper_error"],
                strict=True,
            )
        ]
        _assert_series_matches(
            side=side,
            name="max_gripper_error",
            stored=values[f"{side}_max_gripper_error"],
            derived=derived_max,
        )
        if any(value < 0 for value in values[f"{side}_max_arm_error"]):
            raise ValueError(f"{side} max_arm_error must be non-negative")
    return values


def _pair_passes(
    values: dict[str, list[float]],
    *,
    side: str,
    gripper_tolerance: float,
    arm_tolerance: float,
) -> bool:
    return all(
        gripper <= gripper_tolerance and arm <= arm_tolerance
        for gripper, arm in zip(
            values[f"{side}_max_gripper_error"],
            values[f"{side}_max_arm_error"],
            strict=True,
        )
    )


def _summary(
    result: CheckResult,
    comparison: Comparison,
    pair: EvidencePair,
    *,
    gripper_tolerance: float,
    first_violation_index: int,
    first_violation_time_s: float,
) -> str:
    return "\n".join(
        [
            "# BLOCK: OPEN-LOOP COMPONENT REGRESSION",
            "",
            "**Demo fault:** `300 ms GRIPPER COMMAND DELAY`  ",
            f"**Case / repeat:** `{comparison.case_id}` / `{pair.unit_id}`  ",
            f"**Subjects:** `{result.baseline.id}` to `{result.candidate.id}`  ",
            "**Automatic gate:** baseline `PASS`, candidate `FAIL`, gate `BLOCK`  ",
            f"**Immutable gripper tolerance:** `{gripper_tolerance}`  ",
            f"**First candidate violation:** sample `{first_violation_index}` at "
            f"`{first_violation_time_s:.3f} s`  ",
            "",
            "Recorded cameras, joint states, and effort are immutable context. "
            "The gate proves an output regression under identical recorded inputs; "
            "it does not show a counterfactual physical outcome.",
            "Rerun displays the paired evidence. RoboLens computed the verdict before rendering.",
        ]
    )


def load_aloha_visualization(
    check_dir: Path,
    *,
    case_id: str | None = None,
    unit_id: str | None = None,
) -> AlohaReplayVisualization:
    root = check_dir.resolve()
    result_path = root / "result.json"
    try:
        result = CheckResult.model_validate_json(result_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid check result: {exc}") from exc
    comparison = _select_comparison(result, case_id)
    pair = _select_pair(comparison, unit_id)
    if comparison.metric != "within_tolerance":
        raise ValueError("ALOHA visualization requires within_tolerance metric")
    if comparison.status is not TaskStatus.REGRESSED:
        raise ValueError("ALOHA demo visualization requires a regressed comparison")
    if result.gate.verdict is not GateVerdict.BLOCK:
        raise ValueError("ALOHA demo visualization requires a BLOCK gate")
    if result.baseline.config.get("gripper_delay_messages") != 0:
        raise ValueError("ALOHA baseline must use zero gripper delay")
    if result.candidate.config.get("gripper_delay_messages") != 15:
        raise ValueError("ALOHA candidate must use the declared 15-message delay")

    judge = result.case_judges.get(comparison.case_id, {})
    gripper_tolerance = _finite_nonnegative_judge(judge, "gripper_tolerance")
    arm_tolerance = _finite_nonnegative_judge(judge, "arm_tolerance")
    sample_rate_hz = _finite_nonnegative_judge(judge, "sample_rate_hz")
    if not math.isclose(sample_rate_hz, 50.0, abs_tol=1e-9):
        raise ValueError("ALOHA visualization requires a 50 Hz source")

    selected_evidence: _PairEvidence | None = None
    selected_values: dict[str, list[float]] | None = None
    baseline_passes: list[bool] = []
    candidate_passes: list[bool] = []
    for evidence_pair in comparison.evidence_pairs:
        evidence = _load_pair(root, evidence_pair)
        values = _validate_trace_pair(evidence)
        baseline_passes.append(
            _pair_passes(
                values,
                side="baseline",
                gripper_tolerance=gripper_tolerance,
                arm_tolerance=arm_tolerance,
            )
        )
        candidate_passes.append(
            _pair_passes(
                values,
                side="candidate",
                gripper_tolerance=gripper_tolerance,
                arm_tolerance=arm_tolerance,
            )
        )
        if evidence_pair is pair:
            selected_evidence = evidence
            selected_values = values

    if selected_evidence is None or selected_values is None:
        raise ValueError(f"selected evidence repeat is unavailable: {pair.unit_id}")
    baseline_aggregate = all(baseline_passes)
    candidate_aggregate = all(candidate_passes)
    _validate_automatic_metric(
        side="baseline",
        stored=comparison.baseline_value,
        derived=baseline_aggregate,
    )
    _validate_automatic_metric(
        side="candidate",
        stored=comparison.candidate_value,
        derived=candidate_aggregate,
    )
    baseline_unit_pass = _pair_passes(
        selected_values,
        side="baseline",
        gripper_tolerance=gripper_tolerance,
        arm_tolerance=arm_tolerance,
    )
    candidate_unit_pass = _pair_passes(
        selected_values,
        side="candidate",
        gripper_tolerance=gripper_tolerance,
        arm_tolerance=arm_tolerance,
    )
    if not baseline_unit_pass or candidate_unit_pass:
        raise ValueError("selected ALOHA repeat must derive baseline PASS and candidate FAIL")
    first_violation = next(
        (
            index
            for index, (gripper, arm) in enumerate(
                zip(
                    selected_values["candidate_max_gripper_error"],
                    selected_values["candidate_max_arm_error"],
                    strict=True,
                )
            )
            if gripper > gripper_tolerance or arm > arm_tolerance
        ),
        None,
    )
    if first_violation is None:
        raise ValueError("candidate trace contains no violating sample")
    time_s = selected_values["baseline_time_s"]
    violation_time = time_s[first_violation]

    return AlohaReplayVisualization(
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
        input_mcap=selected_evidence.baseline.input_mcap,
        baseline_output_mcap=selected_evidence.baseline.output_mcap,
        candidate_output_mcap=selected_evidence.candidate.output_mcap,
        input_sha256=selected_evidence.baseline.input_sha256,
        time_s=time_s,
        left_gripper_expected=selected_values["baseline_left_gripper_expected"],
        right_gripper_expected=selected_values["baseline_right_gripper_expected"],
        baseline_left_gripper_observed=selected_values[
            "baseline_left_gripper_observed"
        ],
        baseline_right_gripper_observed=selected_values[
            "baseline_right_gripper_observed"
        ],
        candidate_left_gripper_observed=selected_values[
            "candidate_left_gripper_observed"
        ],
        candidate_right_gripper_observed=selected_values[
            "candidate_right_gripper_observed"
        ],
        baseline_left_gripper_error=selected_values["baseline_left_gripper_error"],
        baseline_right_gripper_error=selected_values[
            "baseline_right_gripper_error"
        ],
        candidate_left_gripper_error=selected_values["candidate_left_gripper_error"],
        candidate_right_gripper_error=selected_values[
            "candidate_right_gripper_error"
        ],
        baseline_max_gripper_error=selected_values["baseline_max_gripper_error"],
        candidate_max_gripper_error=selected_values["candidate_max_gripper_error"],
        baseline_max_arm_error=selected_values["baseline_max_arm_error"],
        candidate_max_arm_error=selected_values["candidate_max_arm_error"],
        gripper_tolerance=gripper_tolerance,
        arm_tolerance=arm_tolerance,
        first_candidate_violation_index=first_violation,
        first_candidate_violation_time_s=violation_time,
        baseline_unit_within_tolerance=baseline_unit_pass,
        candidate_unit_within_tolerance=candidate_unit_pass,
        summary_markdown=_summary(
            result,
            comparison,
            pair,
            gripper_tolerance=gripper_tolerance,
            first_violation_index=first_violation,
            first_violation_time_s=violation_time,
        ),
    )
