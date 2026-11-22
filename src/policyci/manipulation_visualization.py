"""Pure selection and validation for paired manipulation visualizations."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from policyci.mujoco_metrics import (
    derive_lift_metrics,
    derive_placement_metrics,
    failure_window,
)
from policyci.schemas import (
    ArtifactRef,
    CheckResult,
    Comparison,
    EvidencePair,
    GateVerdict,
    MetricValue,
    SimulationTrace,
    TaskStatus,
)
from policyci.visualization import _resolve_artifact, _select_comparison, _select_pair


class ManipulationSensorSeries(BaseModel):
    """Frame-aligned physical sensor signals for one side of a comparison."""

    model_config = ConfigDict(extra="forbid")

    gripper_command: list[float]
    gripper_width_m: list[float]
    left_normal_force_n: list[float]
    right_normal_force_n: list[float]
    wrist_force_magnitude_n: list[float]


class ManipulationVisualization(BaseModel):
    """Validated video and exact-state data for one paired simulation repeat."""

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
    timeline: Literal["sim_time_s"]
    times_s: list[float]
    baseline_video: Path
    candidate_video: Path
    baseline_heights_m: list[float]
    candidate_heights_m: list[float]
    baseline_bilateral_contact: list[bool]
    candidate_bilateral_contact: list[bool]
    baseline_lift_success: bool
    candidate_lift_success: bool
    lift_threshold_m: float | None
    hold_duration_s: float
    task_kind: Literal["lift", "placement"] = "lift"
    baseline_target_distances_m: list[float] = Field(default_factory=list)
    candidate_target_distances_m: list[float] = Field(default_factory=list)
    baseline_task_success: list[bool] = Field(default_factory=list)
    candidate_task_success: list[bool] = Field(default_factory=list)
    baseline_placement_success: bool | None = None
    candidate_placement_success: bool | None = None
    baseline_sensors: ManipulationSensorSeries | None = None
    candidate_sensors: ManipulationSensorSeries | None = None
    failure_start_s: float
    failure_end_s: float
    summary_markdown: str

    @property
    def baseline_unit_pass(self) -> bool:
        if self.task_kind == "placement":
            return self.baseline_placement_success is True
        return self.baseline_lift_success

    @property
    def candidate_unit_pass(self) -> bool:
        if self.task_kind == "placement":
            return self.candidate_placement_success is True
        return self.candidate_lift_success

    @property
    def story_start_s(self) -> float:
        return self.failure_start_s

    @property
    def story_end_s(self) -> float:
        return self.failure_end_s


@dataclass(frozen=True)
class _SideEvidence:
    video: Path
    trace: SimulationTrace


@dataclass(frozen=True)
class _PairEvidence:
    baseline: _SideEvidence
    candidate: _SideEvidence


def _positive_judge(judge: dict[str, MetricValue], name: str) -> float:
    value = judge.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"manipulation visualization requires a positive {name}")
    return float(value)


def _probe_video_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to validate synchronized video evidence")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise ValueError(f"invalid video artifact {path}: {completed.stderr.strip()}")
    try:
        duration = float(json.loads(completed.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"video duration is unavailable: {path}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"video duration must be positive and finite: {path}")
    return duration


def _validate_video_sync(video: Path, trace: SimulationTrace, *, side: str) -> None:
    duration = _probe_video_duration(video)
    final_sample_time = trace.samples[-1].time_s
    allowed_skew = 2.0 / trace.sample_rate_hz
    if abs(duration - final_sample_time) > allowed_skew + 1e-9:
        raise ValueError(
            f"{side} video duration {duration:.6f}s is not synchronized with "
            f"the exact-state trace ending at {final_sample_time:.6f}s"
        )


def _load_side_evidence(
    check_dir: Path,
    artifacts: list[ArtifactRef],
    *,
    side: str,
) -> _SideEvidence:
    videos = [artifact for artifact in artifacts if artifact.kind == "video"]
    trajectories = [artifact for artifact in artifacts if artifact.kind == "trajectory"]
    if len(videos) != 1:
        raise ValueError(f"{side} evidence must contain exactly one video artifact")
    if len(trajectories) != 1:
        raise ValueError(f"{side} evidence must contain exactly one trajectory artifact")

    video = _resolve_artifact(check_dir, videos[0])
    trajectory = _resolve_artifact(check_dir, trajectories[0])
    try:
        trace = SimulationTrace.model_validate_json(trajectory.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid {side} trajectory artifact: {exc}") from exc
    _validate_video_sync(video, trace, side=side)
    return _SideEvidence(video=video, trace=trace)


def _load_pair_evidence(check_dir: Path, pair: EvidencePair) -> _PairEvidence:
    baseline = _load_side_evidence(
        check_dir,
        pair.baseline_artifacts,
        side="baseline",
    )
    candidate = _load_side_evidence(
        check_dir,
        pair.candidate_artifacts,
        side="candidate",
    )
    baseline_times = [sample.time_s for sample in baseline.trace.samples]
    candidate_times = [sample.time_s for sample in candidate.trace.samples]
    if baseline_times != candidate_times:
        raise ValueError("baseline and candidate simulation sample times differ")
    if baseline.trace.sample_rate_hz != candidate.trace.sample_rate_hz:
        raise ValueError("baseline and candidate simulation sample rates differ")
    if baseline.trace.seed != candidate.trace.seed:
        raise ValueError("baseline and candidate simulation seeds differ")
    if baseline.trace.initial_state_sha256 != candidate.trace.initial_state_sha256:
        raise ValueError("baseline and candidate initial state hashes differ")
    if baseline.trace.model_sha256 != candidate.trace.model_sha256:
        raise ValueError("baseline and candidate simulation model hashes differ")
    if (
        baseline.trace.samples[0].has_sensor_evidence
        != candidate.trace.samples[0].has_sensor_evidence
    ):
        raise ValueError("baseline and candidate sensor evidence availability differs")
    return _PairEvidence(baseline=baseline, candidate=candidate)


def _sensor_series(trace: SimulationTrace) -> ManipulationSensorSeries | None:
    if not trace.samples[0].has_sensor_evidence:
        return None
    return ManipulationSensorSeries(
        gripper_command=[float(sample.gripper_command) for sample in trace.samples],
        gripper_width_m=[sample.gripper_width_m for sample in trace.samples],
        left_normal_force_n=[float(sample.left_finger_normal_force_n) for sample in trace.samples],
        right_normal_force_n=[
            float(sample.right_finger_normal_force_n) for sample in trace.samples
        ],
        wrist_force_magnitude_n=[
            math.sqrt(sum(component * component for component in sample.wrist_force_xyz_n))
            for sample in trace.samples
        ],
    )


def _aggregate(values: list[MetricValue]) -> MetricValue:
    if not values:
        raise ValueError("comparison contains no evidence repeats")
    first = values[0]
    if isinstance(first, bool):
        if not all(isinstance(value, bool) for value in values):
            raise ValueError("derived automatic metric types differ across repeats")
        return all(values)
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
        ):
            raise ValueError("derived automatic metric types differ across repeats")
        return sum(float(value) for value in values) / len(values)
    raise ValueError("selected manipulation metric must be numeric or boolean")


def _validate_metric(*, side: str, stored: MetricValue | None, derived: MetricValue) -> None:
    if isinstance(derived, bool):
        matches = isinstance(stored, bool) and stored is derived
    else:
        matches = (
            isinstance(stored, (int, float))
            and not isinstance(stored, bool)
            and math.isclose(float(stored), float(derived), rel_tol=1e-9, abs_tol=1e-12)
        )
    if not matches:
        raise ValueError(f"{side} automatic metric does not match the exact simulator state")


def _metric_for_trace(
    trace: SimulationTrace,
    *,
    metric: str,
    judge: dict[str, MetricValue],
) -> tuple[MetricValue, dict[str, MetricValue]]:
    if "placement_hold_duration_s" in judge:
        metrics = derive_placement_metrics(
            trace,
            hold_duration_s=float(judge["placement_hold_duration_s"]),
        )
    else:
        metrics = derive_lift_metrics(
            trace,
            lift_threshold_m=float(judge["lift_threshold_m"]),
            hold_duration_s=float(judge["hold_duration_s"]),
        )
    if metric not in metrics:
        raise ValueError(f"unsupported manipulation visualization metric: {metric}")
    return metrics[metric], metrics


def _paired_failure_window(
    baseline: SimulationTrace,
    candidate: SimulationTrace,
    *,
    lift_threshold_m: float,
    hold_duration_s: float,
) -> tuple[float, float]:
    """Frame the failed baseline beside the candidate's first proven lift."""
    initial_z = candidate.samples[0].object_position_m[2]
    for sample in candidate.samples:
        height = sample.object_position_m[2] - initial_z
        if (
            height >= lift_threshold_m
            and sample.left_finger_contact
            and sample.right_finger_contact
        ):
            return (
                max(candidate.samples[0].time_s, sample.time_s - 0.75),
                min(candidate.samples[-1].time_s, sample.time_s + 0.75),
            )
    return failure_window(
        baseline,
        lift_threshold_m=lift_threshold_m,
        window_duration_s=hold_duration_s,
    )


def _summary_markdown(
    result: CheckResult,
    comparison: Comparison,
    pair: EvidencePair,
    *,
    task_kind: Literal["lift", "placement"],
    baseline_success: bool,
    candidate_success: bool,
    lift_threshold_m: float | None,
    hold_duration_s: float,
) -> str:
    baseline = "PASS" if baseline_success else "FAIL"
    candidate = "PASS" if candidate_success else "FAIL"
    if task_kind == "placement":
        success_rule = (
            f"**Success rule:** object placed in target bin for ≥ `{hold_duration_s:.2f} s`  "
        )
    else:
        success_rule = (
            f"**Success rule:** lift ≥ `{lift_threshold_m:.3f} m` with bilateral "
            f"contact for ≥ `{hold_duration_s:.2f} s`  "
        )
    return "\n".join(
        [
            f"# {result.gate.verdict.value.upper()} — {comparison.status.value.upper()}",
            "",
            f"**Case / repeat:** `{comparison.case_id}` / `{pair.unit_id}`  ",
            f"**Subjects:** `{result.baseline.id}` → `{result.candidate.id}`  ",
            f"**Visible outcome:** baseline `{baseline}`, candidate `{candidate}`  ",
            f"**Automatic gate:** `{comparison.metric}` = baseline "
            f"`{comparison.baseline_value}`, candidate `{comparison.candidate_value}`  ",
            success_rule,
            "",
            "RoboLens computed the verdict from exact simulator state. The synchronized "
            "videos make the physical regression visible; pixels do not decide the gate.",
        ]
    )


def load_manipulation_visualization(
    check_dir: Path,
    *,
    case_id: str | None = None,
    unit_id: str | None = None,
) -> ManipulationVisualization:
    """Load one sound manipulation evidence pair without importing Rerun."""
    root = check_dir.resolve()
    result_path = root / "result.json"
    if not result_path.is_file():
        raise ValueError(f"check result is missing: {result_path}")
    try:
        result = CheckResult.model_validate_json(result_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid check result: {exc}") from exc
    if result.runner != "mujoco":
        raise ValueError(
            f"manipulation visualization requires runner 'mujoco', got {result.runner!r}"
        )

    comparison = _select_comparison(result, case_id)
    pair = _select_pair(comparison, unit_id)
    judge = result.case_judges.get(comparison.case_id, {})
    placement = "placement_hold_duration_s" in judge
    if placement:
        lift_threshold_m = None
        hold_duration_s = _positive_judge(judge, "placement_hold_duration_s")
    else:
        lift_threshold_m = _positive_judge(judge, "lift_threshold_m")
        hold_duration_s = _positive_judge(judge, "hold_duration_s")

    selected: _PairEvidence | None = None
    baseline_values: list[MetricValue] = []
    candidate_values: list[MetricValue] = []
    selected_baseline_metrics: dict[str, MetricValue] | None = None
    selected_candidate_metrics: dict[str, MetricValue] | None = None
    for evidence_pair in comparison.evidence_pairs:
        evidence = _load_pair_evidence(root, evidence_pair)
        baseline_value, baseline_metrics = _metric_for_trace(
            evidence.baseline.trace,
            metric=comparison.metric,
            judge=judge,
        )
        candidate_value, candidate_metrics = _metric_for_trace(
            evidence.candidate.trace,
            metric=comparison.metric,
            judge=judge,
        )
        baseline_values.append(baseline_value)
        candidate_values.append(candidate_value)
        if evidence_pair is pair:
            selected = evidence
            selected_baseline_metrics = baseline_metrics
            selected_candidate_metrics = candidate_metrics

    if selected is None or selected_baseline_metrics is None or selected_candidate_metrics is None:
        raise ValueError(f"selected evidence repeat is unavailable: {pair.unit_id}")
    _validate_metric(
        side="baseline",
        stored=comparison.baseline_value,
        derived=_aggregate(baseline_values),
    )
    _validate_metric(
        side="candidate",
        stored=comparison.candidate_value,
        derived=_aggregate(candidate_values),
    )

    baseline_trace = selected.baseline.trace
    candidate_trace = selected.candidate.trace
    baseline_initial_z = baseline_trace.samples[0].object_position_m[2]
    candidate_initial_z = candidate_trace.samples[0].object_position_m[2]
    baseline_heights = [
        sample.object_position_m[2] - baseline_initial_z for sample in baseline_trace.samples
    ]
    candidate_heights = [
        sample.object_position_m[2] - candidate_initial_z for sample in candidate_trace.samples
    ]
    baseline_contacts = [
        sample.left_finger_contact and sample.right_finger_contact
        for sample in baseline_trace.samples
    ]
    candidate_contacts = [
        sample.left_finger_contact and sample.right_finger_contact
        for sample in candidate_trace.samples
    ]
    if placement:
        start_s = baseline_trace.samples[0].time_s
        end_s = baseline_trace.samples[-1].time_s
        baseline_success = bool(selected_baseline_metrics["placement_success"])
        candidate_success = bool(selected_candidate_metrics["placement_success"])
        baseline_target_distances = [
            math.dist(sample.object_position_m[:2], sample.target_position_m[:2])
            for sample in baseline_trace.samples
            if sample.target_position_m is not None
        ]
        candidate_target_distances = [
            math.dist(sample.object_position_m[:2], sample.target_position_m[:2])
            for sample in candidate_trace.samples
            if sample.target_position_m is not None
        ]
        baseline_task_success = [bool(sample.task_success) for sample in baseline_trace.samples]
        candidate_task_success = [bool(sample.task_success) for sample in candidate_trace.samples]
    else:
        assert lift_threshold_m is not None
        start_s, end_s = _paired_failure_window(
            baseline_trace,
            candidate_trace,
            lift_threshold_m=lift_threshold_m,
            hold_duration_s=hold_duration_s,
        )
        baseline_success = bool(selected_baseline_metrics["lift_success"])
        candidate_success = bool(selected_candidate_metrics["lift_success"])
        baseline_target_distances = []
        candidate_target_distances = []
        baseline_task_success = []
        candidate_task_success = []

    return ManipulationVisualization(
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
        timeline="sim_time_s",
        times_s=[sample.time_s for sample in baseline_trace.samples],
        baseline_video=selected.baseline.video,
        candidate_video=selected.candidate.video,
        baseline_heights_m=baseline_heights,
        candidate_heights_m=candidate_heights,
        baseline_bilateral_contact=baseline_contacts,
        candidate_bilateral_contact=candidate_contacts,
        baseline_lift_success=baseline_success if not placement else False,
        candidate_lift_success=candidate_success if not placement else False,
        lift_threshold_m=lift_threshold_m,
        hold_duration_s=hold_duration_s,
        task_kind="placement" if placement else "lift",
        baseline_target_distances_m=baseline_target_distances,
        candidate_target_distances_m=candidate_target_distances,
        baseline_task_success=baseline_task_success,
        candidate_task_success=candidate_task_success,
        baseline_placement_success=baseline_success if placement else None,
        candidate_placement_success=candidate_success if placement else None,
        baseline_sensors=_sensor_series(baseline_trace),
        candidate_sensors=_sensor_series(candidate_trace),
        failure_start_s=start_s,
        failure_end_s=end_s,
        summary_markdown=_summary_markdown(
            result,
            comparison,
            pair,
            task_kind="placement" if placement else "lift",
            baseline_success=baseline_success,
            candidate_success=candidate_success,
            lift_threshold_m=lift_threshold_m,
            hold_duration_s=hold_duration_s,
        ),
    )
