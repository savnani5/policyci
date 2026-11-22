"""Pydantic schemas for runs, episodes, and diffs.

Native result format is LeRobot's eval_info.json; we wrap it with a thin manifest.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TaskStatus(str, Enum):
    FIXED = "fixed"
    REGRESSED = "regressed"
    UNCHANGED = "unchanged"
    ADDED = "added"
    REMOVED = "removed"
    NOT_EVALUATED = "not_evaluated"


class GateVerdict(str, Enum):
    PASS = "pass"
    BLOCK = "block"
    NO_VERDICT = "no_verdict"
    # Read compatibility for existing diff.json artifacts. New checks emit BLOCK.
    FAIL = "fail"


class ExecutionValidity(str, Enum):
    VALID = "valid"
    NO_VERDICT = "no_verdict"


class SubjectRef(BaseModel):
    """Complete identity of the code, model, and configuration under test."""

    model_config = ConfigDict(extra="forbid")

    id: str
    checkpoint: str | None = None
    image: str | None = None
    code_sha: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class ArtifactRef(BaseModel):
    """Portable path to evidence stored below a check directory."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    path: str

    @field_validator("path")
    @classmethod
    def path_is_relative_and_confined(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute():
            raise ValueError("artifact path must be relative")
        if ".." in path.parts:
            raise ValueError("artifact path cannot contain traversal segments")
        return value


class EpisodeRecord(BaseModel):
    """Per-episode artifact — rich enough for a future RoboLens PolicyCIParser."""

    episode_id: str
    task: str
    task_id: int | None = None
    suite: str
    instruction: str = ""
    seed: int
    success: bool
    policy_id: str
    video_path: str | None = None
    sum_reward: float | None = None
    max_reward: float | None = None
    # Perturbation metadata (v1; empty for nominal v0 runs)
    perturbation_axis: str | None = None
    perturbation_level: int = 0
    perturbation_params: dict[str, Any] = Field(default_factory=dict)


MetricValue = bool | int | float | str


class SignalSeries(BaseModel):
    """One finite scalar signal aligned to a replay sample index."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    values: list[float] = Field(min_length=1)
    unit: str = ""


class ReplayTrace(BaseModel):
    """Portable signals derived from the exact samples judged by an adapter."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    trace_type: Literal["scalar_command", "aloha_action"] = "scalar_command"
    timeline: Literal["sample_index"] = "sample_index"
    series: dict[str, SignalSeries] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_series_alignment(self) -> "ReplayTrace":
        lengths = set()
        for name, signal in self.series.items():
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
                raise ValueError(f"unsafe replay series name: {name!r}")
            lengths.add(len(signal.values))
        if len(lengths) != 1:
            raise ValueError("replay trace series must have equal lengths")
        return self

    @property
    def sample_count(self) -> int:
        return len(next(iter(self.series.values())).values)


class ReplayAnalysis(BaseModel):
    """Behavioral metrics plus optional human-inspection trace evidence."""

    model_config = ConfigDict(extra="forbid")

    metrics: dict[str, MetricValue] = Field(min_length=1)
    trace: ReplayTrace | None = None


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class SimulationSample(BaseModel):
    """One exact simulator-state sample aligned to a recorded video frame."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    time_s: float = Field(ge=0)
    object_position_m: tuple[float, float, float]
    object_quaternion_xyzw: tuple[float, float, float, float]
    eef_position_m: tuple[float, float, float]
    eef_quaternion_xyzw: tuple[float, float, float, float]
    gripper_width_m: float = Field(ge=0)
    left_finger_contact: bool
    right_finger_contact: bool
    gripper_command: float | None = Field(default=None, ge=-1, le=1)
    left_finger_normal_force_n: float | None = Field(default=None, ge=0)
    right_finger_normal_force_n: float | None = Field(default=None, ge=0)
    wrist_force_xyz_n: tuple[float, float, float] | None = None
    target_position_m: tuple[float, float, float] | None = None
    task_success: bool | None = None
    subtask: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )

    @model_validator(mode="after")
    def validate_sensor_evidence(self) -> "SimulationSample":
        sensor_values = (
            self.gripper_command,
            self.left_finger_normal_force_n,
            self.right_finger_normal_force_n,
            self.wrist_force_xyz_n,
        )
        present = [value is not None for value in sensor_values]
        if any(present) and not all(present):
            raise ValueError("sensor evidence must be all present or all absent")
        return self

    @property
    def has_sensor_evidence(self) -> bool:
        return self.gripper_command is not None


class SimulationTrace(BaseModel):
    """Portable exact-state evidence produced by a physics simulation."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    version: Literal[1] = 1
    timeline: Literal["sim_time_s"] = "sim_time_s"
    sample_rate_hz: float = Field(gt=0)
    seed: int
    initial_state_sha256: Sha256
    model_sha256: Sha256
    samples: list[SimulationSample] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_time(self) -> "SimulationTrace":
        times = [sample.time_s for sample in self.samples]
        if any(current <= previous for previous, current in zip(times, times[1:])):
            raise ValueError("simulation sample times must be strictly increasing")
        sensor_presence = [sample.has_sensor_evidence for sample in self.samples]
        if any(sensor_presence) and not all(sensor_presence):
            raise ValueError("sensor evidence must cover every frame")
        return self

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def duration_s(self) -> float:
        return self.samples[-1].time_s - self.samples[0].time_s


class CaseExecution(BaseModel):
    """One subject's soundness and behavior result for one protected case."""

    subject: SubjectRef
    case_id: str
    runner: str
    validity: ExecutionValidity
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    valid_units: int = 0
    unit_ids: list[str] = Field(default_factory=list)
    episodes: list[EpisodeRecord] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    environment: dict[str, Any] = Field(default_factory=dict)
    synthetic: bool = False


class EvidencePair(BaseModel):
    case_id: str
    unit_id: str
    baseline_artifacts: list[ArtifactRef] = Field(default_factory=list)
    candidate_artifacts: list[ArtifactRef] = Field(default_factory=list)


class Comparison(BaseModel):
    case_id: str
    required: bool = True
    status: TaskStatus
    baseline_validity: ExecutionValidity
    candidate_validity: ExecutionValidity
    metric: str
    baseline_value: MetricValue | None = None
    candidate_value: MetricValue | None = None
    delta: float | None = None
    reason: str = ""
    evidence_pairs: list[EvidencePair] = Field(default_factory=list)


class GateResult(BaseModel):
    verdict: GateVerdict
    exit_code: int
    summary: str


class RobustnessCaseResult(BaseModel):
    """One subject's absolute result for one declared scenario."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    required: bool = True
    object_position_offset_m: tuple[float, float, float]
    object_observation_bias_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    validity: ExecutionValidity
    metric: str
    metric_value: MetricValue | None = None
    passed: bool | None = None
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    environment: dict[str, Any] = Field(default_factory=dict)


class RobustnessCheckResult(BaseModel):
    """Portable single-subject absolute robustness suite result."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    suite_name: str
    suite_version: int = 1
    runner: str
    subject: SubjectRef
    metric: str
    cases: list[RobustnessCaseResult] = Field(min_length=1)
    gate: GateResult
    gate_config: dict[str, Any] = Field(default_factory=dict)
    case_judges: dict[str, dict[str, MetricValue]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class CheckResult(BaseModel):
    check_id: str
    suite_name: str
    suite_version: int = 1
    runner: str = ""
    baseline: SubjectRef
    candidate: SubjectRef
    comparisons: list[Comparison] = Field(default_factory=list)
    gate: GateResult
    gate_config: dict[str, Any] = Field(default_factory=dict)
    case_judges: dict[str, dict[str, MetricValue]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class TaskResult(BaseModel):
    """Aggregated success for one task within a suite."""

    task: str
    task_id: int | None = None
    suite: str
    n_episodes: int
    n_successes: int
    pc_success: float  # 0–100, matching LeRobot
    seeds: list[int] = Field(default_factory=list)
    video_paths: list[str] = Field(default_factory=list)
    perturbation_axis: str | None = None
    perturbation_level: int = 0


class RunManifest(BaseModel):
    """Thin wrapper around one LeRobot eval run."""

    run_id: str
    policy_id: str
    suite: str
    seed: int
    n_episodes: int
    task_ids: list[int] | None = None
    created_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    gpu_seconds: float = 0.0
    gpu_type: str = ""
    status: str = "pending"  # pending | running | succeeded | failed
    error: str | None = None
    # Per-task aggregates (derived from eval_info)
    tasks: list[TaskResult] = Field(default_factory=list)
    overall_pc_success: float | None = None
    # Raw LeRobot eval_info.json (verbatim when available)
    eval_info: dict[str, Any] | None = None
    # Episode-level records for RoboLens seam / clustering
    episodes: list[EpisodeRecord] = Field(default_factory=list)
    # Suite / perturbation context (v1)
    suite_name: str | None = None
    notes: str = ""

    @property
    def gpu_hours(self) -> float:
        return self.gpu_seconds / 3600.0


class TaskDiff(BaseModel):
    task: str
    suite: str
    status: TaskStatus
    v1_pc_success: float | None = None
    v2_pc_success: float | None = None
    delta_pp: float | None = None  # percentage points
    v1_n_episodes: int = 0
    v2_n_episodes: int = 0
    reason: str = ""


class DiffResult(BaseModel):
    """Two-run regression diff + gate verdict."""

    v1_run_id: str
    v2_run_id: str
    v1_policy_id: str
    v2_policy_id: str
    created_at: datetime = Field(default_factory=utcnow)
    threshold_pp: float = 10.0
    min_episodes: int = 5
    tasks: list[TaskDiff] = Field(default_factory=list)
    verdict: GateVerdict = GateVerdict.PASS
    exit_code: int = 0
    summary: str = ""
    n_fixed: int = 0
    n_regressed: int = 0
    n_unchanged: int = 0
    n_not_evaluated: int = 0

    def recount(self) -> None:
        self.n_fixed = sum(1 for t in self.tasks if t.status == TaskStatus.FIXED)
        self.n_regressed = sum(1 for t in self.tasks if t.status == TaskStatus.REGRESSED)
        self.n_unchanged = sum(1 for t in self.tasks if t.status == TaskStatus.UNCHANGED)
        self.n_not_evaluated = sum(1 for t in self.tasks if t.status == TaskStatus.NOT_EVALUATED)


class RobustnessPoint(BaseModel):
    level: int
    pc_success: float
    n_episodes: int
    n_successes: int


class RobustnessProfile(BaseModel):
    """Success-vs-level curve for one axis (optionally scoped to a task)."""

    axis: str
    task: str | None = None  # None = overall
    points: list[RobustnessPoint] = Field(default_factory=list)
    holds_until: int | None = None  # highest level that still passes threshold
    breaks_at: int | None = None  # first level that fails threshold
    threshold_pc: float = 50.0


class FailureCluster(BaseModel):
    axis: str
    task: str
    level: int
    n_failures: int
    episode_ids: list[str] = Field(default_factory=list)
    name: str = ""  # e.g. "instruction_rephrase × libero_object/0 @ L2"
