"""Adapter from the existing Modal/LIBERO evaluator to the RoboLens runner contract."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from policyci.checkspec import CaseSpec, CheckSuite
from policyci.runner.base import PreflightResult
from policyci.schemas import (
    ArtifactRef,
    CaseExecution,
    ExecutionValidity,
    RunManifest,
    SubjectRef,
)

EvalFanout = Callable[..., RunManifest]


def _default_eval_fanout(policy_path: str, shards: list[dict], **kwargs) -> RunManifest:
    from policyci.runner.client import run_eval_fanout

    return run_eval_fanout(policy_path, shards, **kwargs)


class LiberoRunner:
    """Run one protected LIBERO case over an exact seed matrix."""

    _SUPPORTED_CONFIG = frozenset({"n_action_steps", "batch_size"})

    def __init__(
        self,
        suite: CheckSuite,
        *,
        eval_fanout: EvalFanout | None = None,
    ) -> None:
        if suite.runner.type != "libero":
            raise ValueError("LiberoRunner requires runner.type=libero")
        self.suite = suite
        self._eval_fanout = eval_fanout or _default_eval_fanout

    def preflight(self, subject: SubjectRef, case: CaseSpec) -> PreflightResult:
        reasons = []
        if not subject.checkpoint or not subject.checkpoint.strip():
            reasons.append("LIBERO subject requires a checkpoint")
        unsupported_identity = [
            field
            for field in ("image", "code_sha")
            if getattr(subject, field) is not None
        ]
        if unsupported_identity:
            reasons.append(
                "LIBERO runner does not apply subject fields: "
                + ", ".join(unsupported_identity)
            )
        if case.task is None:
            reasons.append(f"LIBERO case {case.id!r} requires a task")
        unknown = sorted(set(subject.config) - self._SUPPORTED_CONFIG)
        if unknown:
            reasons.append(f"unsupported LIBERO subject config: {', '.join(unknown)}")
        for key in self._SUPPORTED_CONFIG & set(subject.config):
            value = subject.config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                reasons.append(f"subject config {key!r} must be a positive integer")

        seeds = case.resolved_seeds(self.suite.execution)
        if len(seeds) < self.suite.gate.min_valid_episodes:
            reasons.append(
                f"case has {len(seeds)} seeds but gate requires "
                f"{self.suite.gate.min_valid_episodes} valid episodes"
            )
        if reasons:
            return PreflightResult.no_verdict(*reasons)
        return PreflightResult.valid()

    def run_case(
        self,
        subject: SubjectRef,
        case: CaseSpec,
        output_dir: Path,
    ) -> CaseExecution:
        assert subject.checkpoint is not None
        assert case.task is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        seeds = case.resolved_seeds(self.suite.execution)
        n_action_steps = int(subject.config.get("n_action_steps", 10))
        batch_size = int(subject.config.get("batch_size", 1))
        shards = [
            {
                "suite": self.suite.runner.suite,
                "task_ids": [case.task],
                "n_episodes": 1,
                "seed": seed,
                "n_action_steps": n_action_steps,
                "batch_size": batch_size,
            }
            for seed in seeds
        ]

        runner_root = output_dir / "runner"
        manifest = self._eval_fanout(
            subject.checkpoint,
            shards,
            aggregate_run_id="execution",
            runs_root=runner_root,
            pull_videos=True,
        )
        artifacts, missing_videos = self._materialize_evidence(
            manifest,
            runner_root=runner_root,
            output_dir=output_dir,
        )
        actual_seeds = [episode.seed for episode in manifest.episodes]
        if manifest.status != "succeeded":
            reason = manifest.error or f"LIBERO aggregate status={manifest.status}"
            reasons = [reason[:1000]]
            if missing_videos:
                reasons.append(
                    f"missing episode videos for seeds: {', '.join(missing_videos)}"
                )
            return CaseExecution(
                subject=subject,
                case_id=case.id,
                runner="libero",
                validity=ExecutionValidity.NO_VERDICT,
                reasons=reasons,
                valid_units=len(actual_seeds),
                unit_ids=[str(seed) for seed in actual_seeds],
                episodes=manifest.episodes,
                artifacts=artifacts,
                environment=self._environment(manifest),
            )

        if actual_seeds != seeds:
            return CaseExecution(
                subject=subject,
                case_id=case.id,
                runner="libero",
                validity=ExecutionValidity.NO_VERDICT,
                reasons=[f"episode seeds differ: expected={seeds}, actual={actual_seeds}"],
                valid_units=len(actual_seeds),
                unit_ids=[str(seed) for seed in actual_seeds],
                episodes=manifest.episodes,
                artifacts=artifacts,
                environment=self._environment(manifest),
            )
        if manifest.overall_pc_success is None:
            return CaseExecution(
                subject=subject,
                case_id=case.id,
                runner="libero",
                validity=ExecutionValidity.NO_VERDICT,
                reasons=["LIBERO result is missing overall success rate"],
                valid_units=len(actual_seeds),
                unit_ids=[str(seed) for seed in actual_seeds],
                episodes=manifest.episodes,
                artifacts=artifacts,
                environment=self._environment(manifest),
            )

        if missing_videos:
            return CaseExecution(
                subject=subject,
                case_id=case.id,
                runner="libero",
                validity=ExecutionValidity.NO_VERDICT,
                reasons=[f"missing episode videos for seeds: {', '.join(missing_videos)}"],
                metrics={"success_rate": manifest.overall_pc_success},
                valid_units=len(actual_seeds),
                unit_ids=[str(seed) for seed in actual_seeds],
                episodes=manifest.episodes,
                artifacts=artifacts,
                environment=self._environment(manifest),
            )

        return CaseExecution(
            subject=subject,
            case_id=case.id,
            runner="libero",
            validity=ExecutionValidity.VALID,
            metrics={"success_rate": manifest.overall_pc_success},
            valid_units=len(actual_seeds),
            unit_ids=[str(seed) for seed in actual_seeds],
            episodes=manifest.episodes,
            artifacts=artifacts,
            environment=self._environment(manifest),
        )

    @staticmethod
    def _materialize_evidence(
        manifest: RunManifest,
        *,
        runner_root: Path,
        output_dir: Path,
    ) -> tuple[list[ArtifactRef], list[str]]:
        """Keep every available episode video, including on partial failures."""
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        artifacts: list[ArtifactRef] = []
        missing_videos: list[str] = []
        source_run = runner_root / manifest.run_id
        for episode in manifest.episodes:
            if not episode.video_path:
                missing_videos.append(str(episode.seed))
                continue
            source = source_run / episode.video_path
            if not source.is_file():
                missing_videos.append(str(episode.seed))
                continue
            relative = f"artifacts/{episode.seed}{source.suffix or '.mp4'}"
            destination = output_dir / relative
            shutil.copy2(source, destination)
            episode.video_path = relative
            artifacts.append(ArtifactRef(kind="video", path=relative))
        return artifacts, missing_videos

    def _environment(self, manifest: RunManifest) -> dict[str, str]:
        return {
            "runner": "libero",
            "suite": manifest.suite,
            "gpu_type": manifest.gpu_type,
        }
