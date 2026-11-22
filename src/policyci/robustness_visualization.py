"""Pure loading and validation for single-subject robustness evidence."""

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
    derive_pick_place_progress,
    derive_placement_metrics,
)
from policyci.schemas import (
    ExecutionValidity,
    GateVerdict,
    MetricValue,
    RobustnessCaseResult,
    RobustnessCheckResult,
    SimulationTrace,
)
from policyci.visualization import _resolve_artifact

# Frame zero is sampled after one physics step; scenario application itself is
# validated before stepping by the adapter.
_FIRST_RECORDED_STEP_POSE_TOLERANCE_M = 1e-4


class RobustnessScenarioDisplay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    offset_m: tuple[float, float, float]
    observation_bias_m: tuple[float, float, float]
    validity: ExecutionValidity
    passed: bool | None
    metric_value: MetricValue | None


class RobustnessVisualization(BaseModel):
    """Validated full-episode data for one base and one failed scenario."""

    model_config = ConfigDict(extra="forbid")

    check_dir: Path
    check_id: str
    suite_name: str
    subject_id: str
    controller_sha256: str
    verdict: GateVerdict
    gate_summary: str
    metric: str
    timeline: Literal["sim_time_s"] = "sim_time_s"
    times_s: list[float]
    scenarios: list[RobustnessScenarioDisplay] = Field(min_length=1)
    base_case_id: str
    perturbed_case_id: str
    base_offset_m: tuple[float, float, float]
    perturbed_offset_m: tuple[float, float, float]
    base_observation_bias_m: tuple[float, float, float]
    perturbed_observation_bias_m: tuple[float, float, float]
    base_passed: bool
    perturbed_passed: bool
    base_video: Path
    perturbed_video: Path
    base_target_distances_m: list[float]
    perturbed_target_distances_m: list[float]
    base_object_heights_m: list[float]
    perturbed_object_heights_m: list[float]
    base_task_success: list[bool]
    perturbed_task_success: list[bool]
    base_subtasks: list[str]
    perturbed_subtasks: list[str]
    base_task_progress_pct: list[float]
    perturbed_task_progress_pct: list[float]
    target_distance_threshold_m: float = 0.05
    hold_duration_s: float
    summary_markdown: str


@dataclass(frozen=True)
class _Evidence:
    video: Path
    trace: SimulationTrace
    metrics: dict[str, MetricValue]


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


def _single_artifact(
    root: Path,
    case: RobustnessCaseResult,
    kind: str,
) -> Path:
    matches = [artifact for artifact in case.artifacts if artifact.kind == kind]
    if len(matches) != 1:
        raise ValueError(f"{case.case_id} must contain exactly one {kind} artifact")
    return _resolve_artifact(root, matches[0])


def _stored_metric_matches(stored: MetricValue | None, derived: MetricValue) -> bool:
    if isinstance(derived, bool):
        return isinstance(stored, bool) and stored is derived
    if isinstance(derived, (int, float)) and not isinstance(derived, bool):
        return (
            isinstance(stored, (int, float))
            and not isinstance(stored, bool)
            and math.isclose(float(stored), float(derived), rel_tol=1e-9, abs_tol=1e-12)
        )
    return type(stored) is type(derived) and stored == derived


def _load_evidence(
    root: Path,
    result: RobustnessCheckResult,
    case: RobustnessCaseResult,
) -> _Evidence:
    if case.validity is not ExecutionValidity.VALID:
        raise ValueError(f"{case.case_id} has no valid simulation evidence")
    video = _single_artifact(root, case, "video")
    trajectory = _single_artifact(root, case, "trajectory")
    try:
        trace = SimulationTrace.model_validate_json(trajectory.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid {case.case_id} trajectory artifact: {exc}") from exc
    duration = _probe_video_duration(video)
    allowed_skew = 2.0 / trace.sample_rate_hz
    if abs(duration - trace.samples[-1].time_s) > allowed_skew + 1e-9:
        raise ValueError(
            f"{case.case_id} video duration is not synchronized with exact state"
        )
    judge = result.case_judges.get(case.case_id, {})
    hold = judge.get("placement_hold_duration_s")
    if (
        isinstance(hold, bool)
        or not isinstance(hold, (int, float))
        or not math.isfinite(float(hold))
        or float(hold) <= 0
    ):
        raise ValueError(f"{case.case_id} requires a positive placement hold duration")
    metrics = derive_placement_metrics(trace, hold_duration_s=float(hold))
    derived = metrics.get(result.metric)
    if derived is None:
        raise ValueError(f"unsupported robustness metric {result.metric!r}")
    if not _stored_metric_matches(case.metric_value, derived):
        raise ValueError(
            f"{case.case_id} automatic metric does not match the exact simulator state"
        )
    expected = result.gate_config.get("expect")
    derived_pass = _stored_metric_matches(expected, derived)
    if case.passed is not derived_pass:
        raise ValueError(
            f"{case.case_id} pass state does not match the exact simulator state"
        )
    runtime_model = case.environment.get("model_sha256")
    if runtime_model != trace.model_sha256:
        raise ValueError(f"{case.case_id} runtime and trajectory model hash differ")
    return _Evidence(video=video, trace=trace, metrics=metrics)


def _is_nominal(case: RobustnessCaseResult) -> bool:
    return all(
        abs(value) <= 1e-12 for value in case.object_position_offset_m
    ) and all(abs(value) <= 1e-12 for value in case.object_observation_bias_m)


def _select_cases(
    result: RobustnessCheckResult,
    failed_case_id: str | None,
) -> tuple[RobustnessCaseResult, RobustnessCaseResult]:
    nominal = [case for case in result.cases if _is_nominal(case)]
    if len(nominal) != 1:
        raise ValueError("robustness visualization requires exactly one nominal case")
    if failed_case_id is not None:
        failed = [case for case in result.cases if case.case_id == failed_case_id]
        if len(failed) != 1:
            raise ValueError(f"unknown failed scenario: {failed_case_id}")
        perturbed = failed[0]
    else:
        failed = [
            case
            for case in result.cases
            if case.required and not _is_nominal(case) and case.passed is False
        ]
        if not failed:
            raise ValueError("robustness visualization requires a failed perturbed case")
        perturbed = failed[0]
    if _is_nominal(perturbed) or perturbed.passed is not False:
        raise ValueError("selected scenario must be a failed non-nominal perturbation")
    return nominal[0], perturbed


def _validate_pair(
    base_case: RobustnessCaseResult,
    base: _Evidence,
    perturbed_case: RobustnessCaseResult,
    perturbed: _Evidence,
) -> None:
    base_times = [sample.time_s for sample in base.trace.samples]
    perturbed_times = [sample.time_s for sample in perturbed.trace.samples]
    if base_times != perturbed_times:
        raise ValueError("base and perturbed simulation timelines differ")
    if base.trace.seed != perturbed.trace.seed:
        raise ValueError("base and perturbed simulation seeds differ")
    if base.trace.sample_rate_hz != perturbed.trace.sample_rate_hz:
        raise ValueError("base and perturbed simulation sample rates differ")
    if base.trace.model_sha256 != perturbed.trace.model_sha256:
        raise ValueError("base and perturbed simulation model hash differs")
    base_controller = base_case.environment.get("controller_sha256")
    perturbed_controller = perturbed_case.environment.get("controller_sha256")
    if not base_controller or base_controller != perturbed_controller:
        raise ValueError("base and perturbed controller hash differs")
    base_initial = base.trace.samples[0].object_position_m
    perturbed_initial = perturbed.trace.samples[0].object_position_m
    expected_delta = tuple(
        perturbed_value - base_value
        for perturbed_value, base_value in zip(
            perturbed_case.object_position_offset_m,
            base_case.object_position_offset_m,
        )
    )
    observed_delta = tuple(
        perturbed_value - base_value
        for perturbed_value, base_value in zip(perturbed_initial, base_initial)
    )
    if any(
        not math.isclose(
            observed,
            expected,
            rel_tol=0.0,
            abs_tol=_FIRST_RECORDED_STEP_POSE_TOLERANCE_M,
        )
        for observed, expected in zip(observed_delta, expected_delta)
    ):
        raise ValueError("observed initial object pose does not match declared perturbation")


def _target_distances(trace: SimulationTrace) -> list[float]:
    return [
        math.dist(sample.object_position_m[:2], sample.target_position_m[:2])
        for sample in trace.samples
        if sample.target_position_m is not None
    ]


def _subtasks(trace: SimulationTrace) -> list[str]:
    values = [sample.subtask for sample in trace.samples]
    if any(value is None for value in values):
        raise ValueError("robustness visualization requires a subtask label on every frame")
    return [str(value) for value in values]


def _summary(
    result: RobustnessCheckResult,
    base: RobustnessCaseResult,
    perturbed: RobustnessCaseResult,
    controller_sha256: str,
) -> str:
    title = (
        "ROBUSTNESS GATE BLOCKED"
        if result.gate.verdict is GateVerdict.BLOCK
        else f"ROBUSTNESS GATE {result.gate.verdict.value.upper()}"
    )
    rows = [
        f"| `{case.case_id}` | `{case.object_position_offset_m[0]:+.2f} m` | "
        f"`{case.object_observation_bias_m[0]:+.2f} m` | "
        f"`{'PASS' if case.passed else 'FAIL' if case.passed is False else 'NO_VERDICT'}` |"
        for case in result.cases
    ]
    return "\n".join(
        [
            f"# {title}",
            "",
            f"**Subject:** `{result.subject.id}`  ",
            f"**Controller:** `{controller_sha256[:12]}…` unchanged across scenarios  ",
            f"**Comparison:** base `{base.case_id}` vs perturbed `{perturbed.case_id}`  ",
            f"**Gate:** `{result.metric}` must pass for every protected scenario  ",
            "",
            "| Scenario | Object X offset | Perception X bias | Exact-state result |",
            "| --- | ---: | ---: | --- |",
            *rows,
            "",
            "The controller retargets to the shifted object and that case passes. "
            "A biased object observation changes the commanded grasp pose and blocks placement.",
        ]
    )


def load_robustness_visualization(
    check_dir: Path,
    *,
    failed_case_id: str | None = None,
) -> RobustnessVisualization:
    """Load one honest nominal-versus-perturbed evidence view without Rerun imports."""

    root = check_dir.resolve()
    result_path = root / "result.json"
    if not result_path.is_file():
        raise ValueError(f"robustness result is missing: {result_path}")
    try:
        result = RobustnessCheckResult.model_validate_json(result_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid robustness result: {exc}") from exc
    if result.runner != "mujoco" or result.metric != "placement_success":
        raise ValueError("robustness visualization requires MuJoCo placement_success")

    base_case, perturbed_case = _select_cases(result, failed_case_id)
    base = _load_evidence(root, result, base_case)
    perturbed = _load_evidence(root, result, perturbed_case)
    _validate_pair(base_case, base, perturbed_case, perturbed)
    if base_case.passed is not True or perturbed_case.passed is not False:
        raise ValueError("visual story requires a passing base and failed perturbation")
    controller_sha256 = str(base_case.environment["controller_sha256"])
    hold_duration = float(
        result.case_judges[base_case.case_id]["placement_hold_duration_s"]
    )
    return RobustnessVisualization(
        check_dir=root,
        check_id=result.check_id,
        suite_name=result.suite_name,
        subject_id=result.subject.id,
        controller_sha256=controller_sha256,
        verdict=result.gate.verdict,
        gate_summary=result.gate.summary,
        metric=result.metric,
        times_s=[sample.time_s for sample in base.trace.samples],
        scenarios=[
            RobustnessScenarioDisplay(
                case_id=case.case_id,
                offset_m=case.object_position_offset_m,
                observation_bias_m=case.object_observation_bias_m,
                validity=case.validity,
                passed=case.passed,
                metric_value=case.metric_value,
            )
            for case in result.cases
        ],
        base_case_id=base_case.case_id,
        perturbed_case_id=perturbed_case.case_id,
        base_offset_m=base_case.object_position_offset_m,
        perturbed_offset_m=perturbed_case.object_position_offset_m,
        base_observation_bias_m=base_case.object_observation_bias_m,
        perturbed_observation_bias_m=perturbed_case.object_observation_bias_m,
        base_passed=True,
        perturbed_passed=False,
        base_video=base.video,
        perturbed_video=perturbed.video,
        base_target_distances_m=_target_distances(base.trace),
        perturbed_target_distances_m=_target_distances(perturbed.trace),
        base_object_heights_m=[
            sample.object_position_m[2] for sample in base.trace.samples
        ],
        perturbed_object_heights_m=[
            sample.object_position_m[2] for sample in perturbed.trace.samples
        ],
        base_task_success=[bool(sample.task_success) for sample in base.trace.samples],
        perturbed_task_success=[
            bool(sample.task_success) for sample in perturbed.trace.samples
        ],
        base_subtasks=_subtasks(base.trace),
        perturbed_subtasks=_subtasks(perturbed.trace),
        base_task_progress_pct=derive_pick_place_progress(
            base.trace,
            placement_hold_duration_s=hold_duration,
        ),
        perturbed_task_progress_pct=derive_pick_place_progress(
            perturbed.trace,
            placement_hold_duration_s=hold_duration,
        ),
        hold_duration_s=hold_duration,
        summary_markdown=_summary(
            result,
            base_case,
            perturbed_case,
            controller_sha256,
        ),
    )
