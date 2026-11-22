"""Physics-backed manipulation runner with portable exact-state evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from policyci.checkspec import CaseSpec, CheckSuite
from policyci.mujoco_metrics import derive_lift_metrics, derive_placement_metrics
from policyci.runner.base import PreflightResult
from policyci.schemas import (
    ArtifactRef,
    CaseExecution,
    ExecutionValidity,
    MetricValue,
    SimulationTrace,
    SubjectRef,
)

RuntimeValue = str | int | float


@dataclass(frozen=True)
class SimulationEpisode:
    trace: SimulationTrace
    video_path: Path
    runtime: dict[str, RuntimeValue]


class MujocoAdapter(Protocol):
    def applied_subject_fields(
        self,
        subject: SubjectRef,
        case: CaseSpec,
    ) -> set[str]: ...

    def run_episode(
        self,
        subject: SubjectRef,
        case: CaseSpec,
        *,
        seed: int,
        output_dir: Path,
    ) -> SimulationEpisode: ...


@dataclass
class _EpisodeOutcome:
    unit_id: str
    trace: SimulationTrace | None = None
    runtime: dict[str, RuntimeValue] | None = None
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.reasons and self.trace is not None and self.runtime is not None


class _DefaultMujocoAdapter:
    """Select the built-in task adapter from the immutable case judge."""

    @staticmethod
    def adapter_for(case: CaseSpec) -> MujocoAdapter:
        if "placement_hold_duration_s" in case.judge:
            from policyci.examples.mujoco_grasp.pick_place import (
                RobosuitePickPlaceAdapter,
            )

            return RobosuitePickPlaceAdapter()

        from policyci.examples.mujoco_grasp.adapter import RobosuiteLiftAdapter

        return RobosuiteLiftAdapter()

    def applied_subject_fields(
        self,
        subject: SubjectRef,
        case: CaseSpec,
    ) -> set[str]:
        return self.adapter_for(case).applied_subject_fields(subject, case)

    def run_episode(
        self,
        subject: SubjectRef,
        case: CaseSpec,
        *,
        seed: int,
        output_dir: Path,
    ) -> SimulationEpisode:
        return self.adapter_for(case).run_episode(
            subject,
            case,
            seed=seed,
            output_dir=output_dir,
        )


def _default_adapter() -> MujocoAdapter:
    return _DefaultMujocoAdapter()


def _aggregate_metrics(outcomes: list[_EpisodeOutcome]) -> dict[str, MetricValue]:
    key_sets = [set(outcome.metrics) for outcome in outcomes]
    if not key_sets or any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("repeat metric keys differ")

    aggregated: dict[str, MetricValue] = {}
    for key in sorted(key_sets[0]):
        values = [outcome.metrics[key] for outcome in outcomes]
        if all(isinstance(value, bool) for value in values):
            aggregated[key] = all(values)
        elif all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values
        ):
            aggregated[key] = sum(float(value) for value in values) / len(values)
        elif all(type(value) is type(values[0]) and value == values[0] for value in values):
            aggregated[key] = values[0]
        else:
            raise ValueError(f"repeat metric {key!r} cannot be aggregated soundly")
    return aggregated


class MujocoRunner:
    """Run an exact seed matrix and gate from MuJoCo state, not video pixels."""

    _SUPPORTED_CONFIG = frozenset({"action_frame_offset_m"})

    def __init__(
        self,
        suite: CheckSuite,
        *,
        adapter: MujocoAdapter | None = None,
    ) -> None:
        if suite.runner.type != "mujoco":
            raise ValueError("MujocoRunner requires runner.type=mujoco")
        self.suite = suite
        self._adapter = adapter

    @property
    def adapter(self) -> MujocoAdapter:
        if self._adapter is None:
            self._adapter = _default_adapter()
        return self._adapter

    def preflight(self, subject: SubjectRef, case: CaseSpec) -> PreflightResult:
        reasons: list[str] = []
        if subject.checkpoint is not None or subject.image is not None:
            reasons.append("MuJoCo runner does not apply checkpoint or image fields")

        unknown = sorted(set(subject.config) - self._SUPPORTED_CONFIG)
        if unknown:
            reasons.append("unsupported MuJoCo subject config: " + ", ".join(unknown))

        offset = subject.config.get("action_frame_offset_m")
        if not isinstance(offset, (list, tuple)) or len(offset) != 3:
            reasons.append("subject config 'action_frame_offset_m' must have three numbers")
        elif any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or abs(float(value)) > 0.10
            for value in offset
        ):
            reasons.append(
                "subject config 'action_frame_offset_m' values must be finite and within +/-0.10 m"
            )

        seeds = case.resolved_seeds(self.suite.execution)
        if len(seeds) < self.suite.gate.min_valid_repeats:
            reasons.append(
                f"case has {len(seeds)} seeds but gate requires "
                f"{self.suite.gate.min_valid_repeats} valid repeats"
            )
        if reasons:
            return PreflightResult.no_verdict(*reasons)

        try:
            declared_fields = {
                field
                for field in ("code_sha",)
                if getattr(subject, field) is not None
            }
            declared_fields.update(f"config.{key}" for key in subject.config)
            applied_fields = set(self.adapter.applied_subject_fields(subject, case))
            unapplied = sorted(declared_fields - applied_fields)
            if unapplied:
                raise ValueError(
                    "adapter does not apply subject fields: " + ", ".join(unapplied)
                )
        except Exception as exc:
            return PreflightResult.no_verdict(
                f"adapter configuration failed: {str(exc)[:1000]}"
            )
        return PreflightResult.valid()

    def run_case(
        self,
        subject: SubjectRef,
        case: CaseSpec,
        output_dir: Path,
    ) -> CaseExecution:
        output_dir.mkdir(parents=True, exist_ok=True)
        outcomes = [
            self._run_episode(subject, case, seed=seed, output_dir=output_dir)
            for seed in case.resolved_seeds(self.suite.execution)
        ]
        artifacts = [artifact for outcome in outcomes for artifact in outcome.artifacts]
        invalid_reasons = [
            f"{outcome.unit_id}: {reason}"
            for outcome in outcomes
            for reason in outcome.reasons
        ]
        valid = [outcome for outcome in outcomes if outcome.valid]

        if invalid_reasons or len(valid) < self.suite.gate.min_valid_repeats:
            reasons = list(invalid_reasons)
            if len(valid) < self.suite.gate.min_valid_repeats:
                reasons.insert(
                    0,
                    "insufficient valid repeats: "
                    f"valid={len(valid)}, required={self.suite.gate.min_valid_repeats}",
                )
            return self._execution(
                subject,
                case,
                valid,
                artifacts,
                validity=ExecutionValidity.NO_VERDICT,
                reasons=reasons,
            )

        runtime = valid[0].runtime
        if any(outcome.runtime != runtime for outcome in valid[1:]):
            return self._execution(
                subject,
                case,
                valid,
                artifacts,
                validity=ExecutionValidity.NO_VERDICT,
                reasons=["repeat runtime identities differ"],
            )
        model_hashes = {outcome.trace.model_sha256 for outcome in valid if outcome.trace}
        sample_rates = {outcome.trace.sample_rate_hz for outcome in valid if outcome.trace}
        if len(model_hashes) != 1 or sample_rates != {20.0}:
            return self._execution(
                subject,
                case,
                valid,
                artifacts,
                validity=ExecutionValidity.NO_VERDICT,
                reasons=["repeat model identity or sample rate differs"],
            )

        try:
            metrics = _aggregate_metrics(valid)
        except ValueError as exc:
            return self._execution(
                subject,
                case,
                valid,
                artifacts,
                validity=ExecutionValidity.NO_VERDICT,
                reasons=[str(exc)],
            )
        return self._execution(
            subject,
            case,
            valid,
            artifacts,
            validity=ExecutionValidity.VALID,
            metrics=metrics,
        )

    def _run_episode(
        self,
        subject: SubjectRef,
        case: CaseSpec,
        *,
        seed: int,
        output_dir: Path,
    ) -> _EpisodeOutcome:
        unit_id = str(seed)
        repeat_dir = output_dir / "repeats" / unit_id
        repeat_dir.mkdir(parents=True, exist_ok=True)
        outcome = _EpisodeOutcome(unit_id=unit_id)
        try:
            episode = self.adapter.run_episode(
                subject,
                case,
                seed=seed,
                output_dir=repeat_dir,
            )
            if episode.trace.seed != seed:
                raise ValueError(
                    f"trace seed differs: expected={seed}, actual={episode.trace.seed}"
                )
            video = episode.video_path.resolve()
            confined_root = repeat_dir.resolve()
            if not video.is_relative_to(confined_root):
                raise ValueError("video path escapes the repeat directory")
            if not video.is_file() or video.stat().st_size == 0:
                raise ValueError("video is missing or empty")
            video_relative = video.relative_to(output_dir.resolve()).as_posix()
            outcome.artifacts.append(ArtifactRef(kind="video", path=video_relative))

            trajectory = repeat_dir / "trajectory.json"
            trajectory.write_text(episode.trace.model_dump_json(indent=2))
            outcome.artifacts.append(
                ArtifactRef(
                    kind="trajectory",
                    path=trajectory.relative_to(output_dir).as_posix(),
                )
            )
            if "placement_hold_duration_s" in case.judge:
                outcome.metrics = derive_placement_metrics(
                    episode.trace,
                    hold_duration_s=float(case.judge["placement_hold_duration_s"]),
                )
            else:
                outcome.metrics = derive_lift_metrics(
                    episode.trace,
                    lift_threshold_m=float(case.judge["lift_threshold_m"]),
                    hold_duration_s=float(case.judge["hold_duration_s"]),
                )
            outcome.trace = episode.trace
            outcome.runtime = dict(episode.runtime)
        except Exception as exc:
            outcome.reasons.append(str(exc)[:1000])
        return outcome

    @staticmethod
    def _execution(
        subject: SubjectRef,
        case: CaseSpec,
        outcomes: list[_EpisodeOutcome],
        artifacts: list[ArtifactRef],
        *,
        validity: ExecutionValidity,
        reasons: list[str] | None = None,
        metrics: dict[str, MetricValue] | None = None,
    ) -> CaseExecution:
        traces = [outcome.trace for outcome in outcomes if outcome.trace is not None]
        runtime = outcomes[0].runtime if outcomes else None
        environment: dict[str, object] = {
            **(runtime or {}),
            "runner": "mujoco",
            "sample_rate_hz": 20.0,
            "initial_state_sha256_by_seed": {
                outcome.unit_id: outcome.trace.initial_state_sha256
                for outcome in outcomes
                if outcome.trace is not None
            },
        }
        if traces:
            environment["model_sha256"] = traces[0].model_sha256
        return CaseExecution(
            subject=subject,
            case_id=case.id,
            runner="mujoco",
            validity=validity,
            reasons=reasons or [],
            metrics=metrics or {},
            valid_units=len(outcomes),
            unit_ids=[outcome.unit_id for outcome in outcomes],
            artifacts=artifacts,
            environment=environment,
        )
