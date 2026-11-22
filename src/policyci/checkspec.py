"""Strict YAML inputs for the runner-independent RoboLens check workflow."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from policyci.schemas import MetricValue, SubjectRef


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunnerSpec(StrictModel):
    type: Literal["libero", "ros2_replay", "mujoco"]
    suite: str | None = None
    adapter: str | None = None

    @model_validator(mode="after")
    def validate_runner_fields(self) -> "RunnerSpec":
        if self.type == "libero" and not self.suite:
            raise ValueError("libero runner requires runner.suite")
        if self.type == "ros2_replay" and not self.adapter:
            raise ValueError("ros2_replay runner requires runner.adapter")
        return self


class ExecutionSpec(StrictModel):
    episodes: int = Field(default=1, ge=1)
    seed: int = 1000
    seeds: list[int] | None = None
    readiness_timeout_s: float = Field(default=15.0, gt=0)
    playback_discovery_delay_s: float = Field(default=1.0, ge=0)
    playback_timeout_s: float = Field(default=300.0, gt=0)

    @model_validator(mode="after")
    def validate_seed_count(self) -> "ExecutionSpec":
        if self.seeds is not None and len(self.seeds) != self.episodes:
            raise ValueError(
                f"execution episodes ({self.episodes}) must equal seeds count ({len(self.seeds)})"
            )
        if self.seeds is not None and len(set(self.seeds)) != len(self.seeds):
            raise ValueError("execution contains duplicate seeds")
        return self


class GateSpec(StrictModel):
    metric: str
    mode: Literal["max_drop", "require_value"]
    max_drop_pp: float = Field(default=10.0, ge=0)
    expect: MetricValue | None = None
    min_valid_episodes: int = Field(default=1, ge=1)
    min_valid_repeats: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_mode_fields(self) -> "GateSpec":
        if self.mode == "require_value" and self.expect is None:
            raise ValueError("require_value gate requires gate.expect")
        return self


class CaseSpec(StrictModel):
    id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    )
    required: bool = True

    # LIBERO fields
    task: int | None = None
    episodes: int | None = Field(default=None, ge=1)
    seed: int | None = None
    seeds: list[int] | None = None

    # ROS 2 replay fields
    bag: Path | None = None
    repeats: int = Field(default=1, ge=1)
    required_topics: list[str] = Field(default_factory=list)
    playback_topics: list[str] = Field(default_factory=list)
    capture_topics: list[str] = Field(default_factory=list)
    judge: dict[str, MetricValue] = Field(default_factory=dict)

    # MuJoCo scenario fields
    object_position_offset_m: tuple[float, float, float] | None = None
    object_observation_bias_m: tuple[float, float, float] | None = None

    @field_validator("object_position_offset_m", mode="before")
    @classmethod
    def validate_object_position_offset(
        cls,
        value: object,
    ) -> object:
        if value is None:
            return value
        if (
            isinstance(value, (str, bytes, bool))
            or not isinstance(value, (list, tuple))
            or len(value) != 3
        ):
            raise ValueError("object_position_offset_m must contain three numbers")
        if any(
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            or not math.isfinite(float(component))
            or abs(float(component)) > 0.10
            for component in value
        ):
            raise ValueError(
                "object_position_offset_m values must be finite and within +/-0.10 m"
            )
        return tuple(float(component) for component in value)

    @field_validator("object_observation_bias_m", mode="before")
    @classmethod
    def validate_object_observation_bias(cls, value: object) -> object:
        if value is None:
            return value
        if (
            isinstance(value, (str, bytes, bool))
            or not isinstance(value, (list, tuple))
            or len(value) != 3
        ):
            raise ValueError("object_observation_bias_m must contain three numbers")
        if any(
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            or not math.isfinite(float(component))
            or abs(float(component)) > 0.10
            for component in value
        ):
            raise ValueError(
                "object_observation_bias_m values must be finite and within +/-0.10 m"
            )
        return tuple(float(component) for component in value)

    def resolved_seeds(self, execution: ExecutionSpec) -> list[int]:
        episodes = self.episodes if self.episodes is not None else execution.episodes
        seeds = self.seeds if self.seeds is not None else execution.seeds
        first_seed = self.seed if self.seed is not None else execution.seed
        if seeds is not None:
            if len(seeds) != episodes:
                raise ValueError(
                    f"case {self.id!r} episodes ({episodes}) must equal seeds count ({len(seeds)})"
                )
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"case {self.id!r} contains duplicate seeds")
            return list(seeds)
        return list(range(first_seed, first_seed + episodes))

    def resolved_object_position_offset_m(self) -> tuple[float, float, float]:
        return self.object_position_offset_m or (0.0, 0.0, 0.0)

    def resolved_object_observation_bias_m(self) -> tuple[float, float, float]:
        return self.object_observation_bias_m or (0.0, 0.0, 0.0)


class CheckSuite(StrictModel):
    version: Literal[1]
    name: str
    runner: RunnerSpec
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    gate: GateSpec
    cases: list[CaseSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cases_for_runner(self) -> "CheckSuite":
        ids = [case.id for case in self.cases]
        duplicate_ids = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        if duplicate_ids:
            raise ValueError(f"duplicate case id(s): {', '.join(duplicate_ids)}")

        for case in self.cases:
            if self.runner.type != "mujoco" and case.object_position_offset_m is not None:
                raise ValueError(
                    f"{self.runner.type} case {case.id!r} contains MuJoCo-only "
                    "object_position_offset_m"
                )
            if (
                self.runner.type != "mujoco"
                and case.object_observation_bias_m is not None
            ):
                raise ValueError(
                    f"{self.runner.type} case {case.id!r} contains MuJoCo-only "
                    "object_observation_bias_m"
                )
            if self.runner.type == "libero":
                if case.task is None:
                    raise ValueError(f"libero case {case.id!r} requires task")
                if (
                    case.bag is not None
                    or case.required_topics
                    or case.playback_topics
                    or case.capture_topics
                ):
                    raise ValueError(f"libero case {case.id!r} contains replay-only fields")
                if case.judge:
                    raise ValueError(f"libero case {case.id!r} contains replay-only judge fields")
                case.resolved_seeds(self.execution)
            elif self.runner.type == "ros2_replay":
                if case.bag is None:
                    raise ValueError(f"ros2_replay case {case.id!r} requires bag")
                if not case.required_topics:
                    raise ValueError(f"ros2_replay case {case.id!r} requires required_topics")
                if not case.capture_topics:
                    raise ValueError(f"ros2_replay case {case.id!r} requires capture_topics")
                unknown_playback_topics = sorted(
                    set(case.playback_topics) - set(case.required_topics)
                )
                if unknown_playback_topics:
                    raise ValueError(
                        f"case {case.id!r} playback topics must be required topics: "
                        f"{unknown_playback_topics}"
                    )
                overlap = sorted(set(case.required_topics) & set(case.capture_topics))
                if overlap:
                    raise ValueError(
                        f"case {case.id!r} input and capture topics must be disjoint: {overlap}"
                    )
            else:
                if case.task is not None or case.bag is not None:
                    raise ValueError(
                        f"mujoco case {case.id!r} contains another runner's fields"
                    )
                if case.required_topics or case.playback_topics or case.capture_topics:
                    raise ValueError(
                        f"mujoco case {case.id!r} contains replay-only fields"
                    )
                judge_fields = set(case.judge)
                supported_judges = (
                    {"lift_threshold_m", "hold_duration_s"},
                    {"placement_hold_duration_s"},
                )
                if judge_fields not in supported_judges:
                    raise ValueError(
                        f"mujoco case {case.id!r} judge requires either "
                        "hold_duration_s + lift_threshold_m or "
                        "placement_hold_duration_s"
                    )
                for name in sorted(judge_fields):
                    value = case.judge[name]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or value <= 0
                    ):
                        raise ValueError(
                            f"mujoco judge {name!r} must be positive numeric"
                        )
                lift_metrics = {
                    "lift_success",
                    "max_object_lift_m",
                    "bilateral_contact_duration_s",
                    "min_eef_object_distance_m",
                }
                placement_metrics = {
                    "placement_success",
                    "placement_success_duration_s",
                    "min_object_target_xy_distance_m",
                    "final_object_target_xy_distance_m",
                    "max_object_lift_m",
                    "bilateral_contact_duration_s",
                }
                allowed_metrics = (
                    placement_metrics
                    if "placement_hold_duration_s" in judge_fields
                    else lift_metrics
                )
                if self.gate.metric not in allowed_metrics:
                    raise ValueError(
                        f"mujoco gate metric {self.gate.metric!r} does not match "
                        f"case {case.id!r} judge"
                    )
                case.resolved_seeds(self.execution)
        return self


def _load_yaml_mapping(path: Path | str) -> dict:
    source = Path(path)
    data = yaml.safe_load(source.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a mapping: {source}")
    return data


def load_subject(path: Path | str) -> SubjectRef:
    return SubjectRef.model_validate(_load_yaml_mapping(path))


def load_check_suite(path: Path | str) -> CheckSuite:
    source = Path(path)
    suite = CheckSuite.model_validate(_load_yaml_mapping(source))
    if suite.runner.type != "ros2_replay":
        return suite

    resolved_cases = []
    for case in suite.cases:
        assert case.bag is not None
        bag = case.bag if case.bag.is_absolute() else (source.parent / case.bag).resolve()
        resolved_cases.append(case.model_copy(update={"bag": bag}))
    return suite.model_copy(update={"cases": resolved_cases})
