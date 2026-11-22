"""Bounded ROS 2 input replay runner with explicit soundness checks."""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import yaml

from policyci.checkspec import CaseSpec, CheckSuite
from policyci.runner.base import PreflightResult
from policyci.schemas import (
    ArtifactRef,
    CaseExecution,
    ExecutionValidity,
    MetricValue,
    ReplayAnalysis,
    SubjectRef,
)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class ManagedProcess(Protocol):
    def poll(self) -> int | None: ...

    def stop(self, timeout_s: float) -> CommandResult: ...


class ProcessBackend(Protocol):
    def run(
        self,
        argv: list[str],
        *,
        timeout_s: float,
        env: dict[str, str] | None = None,
    ) -> CommandResult: ...

    def start(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
    ) -> ManagedProcess: ...


class Ros2ReplayAdapter(Protocol):
    def applied_subject_fields(
        self,
        subject: SubjectRef,
        case: CaseSpec,
    ) -> set[str]: ...

    def subject_command(self, subject: SubjectRef, case: CaseSpec) -> list[str]: ...

    def readiness_command(self, subject: SubjectRef, case: CaseSpec) -> list[str]: ...

    def analyze(
        self,
        subject: SubjectRef,
        case: CaseSpec,
        capture: Path,
    ) -> dict[str, MetricValue] | ReplayAnalysis: ...


def _normalize_analysis(
    value: dict[str, MetricValue] | ReplayAnalysis,
) -> ReplayAnalysis:
    if isinstance(value, ReplayAnalysis):
        return value
    if isinstance(value, dict) and value:
        return ReplayAnalysis(metrics=value)
    raise ValueError("adapter returned no metrics")


def _metadata_topic_message_counts(metadata: Path) -> dict[str, int]:
    payload = yaml.safe_load(metadata.read_text())
    if not isinstance(payload, dict):
        raise ValueError("metadata root is not a mapping")
    information = payload.get("rosbag2_bagfile_information")
    if not isinstance(information, dict):
        raise ValueError("rosbag2_bagfile_information is missing")
    topics = information.get("topics_with_message_count")
    if not isinstance(topics, list):
        raise ValueError("topics_with_message_count is missing")

    counts: dict[str, int] = {}
    for entry in topics:
        if not isinstance(entry, dict):
            continue
        topic_metadata = entry.get("topic_metadata")
        if not isinstance(topic_metadata, dict):
            continue
        name = topic_metadata.get("name")
        count = entry.get("message_count")
        if isinstance(name, str) and isinstance(count, int) and not isinstance(count, bool):
            counts[name] = count
    return counts


class _PopenProcess:
    def __init__(self, process: subprocess.Popen[str]) -> None:
        self.process = process

    def poll(self) -> int | None:
        return self.process.poll()

    def stop(self, timeout_s: float) -> CommandResult:
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            stdout, stderr = self.process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = self.process.communicate()
        return CommandResult(
            returncode=self.process.returncode or 0,
            stdout=(stdout or "")[-4000:],
            stderr=(stderr or "")[-4000:],
        )


class SubprocessBackend:
    """Argument-vector subprocess boundary; never invokes a shell."""

    _RUNTIME_ENV_KEYS = frozenset(
        {
            "AMENT_PREFIX_PATH",
            "CMAKE_PREFIX_PATH",
            "COLCON_PREFIX_PATH",
            "CYCLONEDDS_URI",
            "DYLD_LIBRARY_PATH",
            "FASTRTPS_DEFAULT_PROFILES_FILE",
            "HOME",
            "LANG",
            "LANGUAGE",
            "LC_ALL",
            "LD_LIBRARY_PATH",
            "LOGNAME",
            "PATH",
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONUNBUFFERED",
            "RCUTILS_COLORIZED_OUTPUT",
            "RMW_IMPLEMENTATION",
            "ROS_DISTRO",
            "ROS_DOMAIN_ID",
            "ROS_LOCALHOST_ONLY",
            "ROS_PYTHON_VERSION",
            "ROS_VERSION",
            "SHELL",
            "TEMP",
            "TMP",
            "TMPDIR",
            "USER",
        }
    )

    @classmethod
    def _environment(cls, extra: dict[str, str] | None) -> dict[str, str]:
        allowed = {
            key: value
            for key, value in os.environ.items()
            if key in cls._RUNTIME_ENV_KEYS
        }
        return {**allowed, **(extra or {})}

    def run(
        self,
        argv: list[str],
        *,
        timeout_s: float,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=self._environment(env),
            )
            return CommandResult(
                returncode=completed.returncode,
                stdout=(completed.stdout or "")[-4000:],
                stderr=(completed.stderr or "")[-4000:],
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                returncode=124,
                stdout=str(exc.stdout or "")[-4000:],
                stderr=f"command timed out after {timeout_s:g}s",
            )

    def start(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
    ) -> ManagedProcess:
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._environment(env),
            start_new_session=True,
        )
        return _PopenProcess(process)


@dataclass
class RepeatOutcome:
    unit_id: str
    validity: ExecutionValidity
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    artifacts: list[ArtifactRef] = field(default_factory=list)


def _load_adapter(reference: str) -> Ros2ReplayAdapter:
    if ":" not in reference:
        raise ValueError("adapter must use module:attribute syntax")
    module_name, attribute = reference.split(":", 1)
    module = importlib.import_module(module_name)
    adapter_type = getattr(module, attribute)
    adapter = adapter_type() if isinstance(adapter_type, type) else adapter_type
    for method in (
        "applied_subject_fields",
        "subject_command",
        "readiness_command",
        "analyze",
    ):
        if not callable(getattr(adapter, method, None)):
            raise TypeError(f"adapter {reference!r} is missing {method}()")
    return adapter


def _bag_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class Ros2ReplayRunner:
    def __init__(
        self,
        suite: CheckSuite,
        *,
        adapter: Ros2ReplayAdapter | None = None,
        backend: ProcessBackend | None = None,
        which: Callable[[str], str | None] = shutil.which,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
    ) -> None:
        if suite.runner.type != "ros2_replay":
            raise ValueError("Ros2ReplayRunner requires runner.type=ros2_replay")
        self.suite = suite
        self._adapter = adapter
        self._backend = backend or SubprocessBackend()
        self._which = which
        self._sleep = sleep
        self._monotonic = monotonic
        self._wall_time = wall_time

    @property
    def adapter(self) -> Ros2ReplayAdapter:
        if self._adapter is None:
            assert self.suite.runner.adapter is not None
            self._adapter = _load_adapter(self.suite.runner.adapter)
        return self._adapter

    def preflight(self, subject: SubjectRef, case: CaseSpec) -> PreflightResult:
        reasons = []
        if self._which("ros2") is None:
            reasons.append("ros2 executable is unavailable")
        if case.bag is None or not case.bag.is_file():
            reasons.append(f"input bag does not exist: {case.bag}")
        if case.repeats < self.suite.gate.min_valid_repeats:
            reasons.append(
                f"case has {case.repeats} repeats but gate requires "
                f"{self.suite.gate.min_valid_repeats}"
            )
        if reasons:
            return PreflightResult.no_verdict(*reasons)

        try:
            declared_fields = {
                field
                for field in ("checkpoint", "image", "code_sha")
                if getattr(subject, field) is not None
            }
            declared_fields.update(f"config.{key}" for key in subject.config)
            applied_fields = set(self.adapter.applied_subject_fields(subject, case))
            unapplied_fields = sorted(declared_fields - applied_fields)
            if unapplied_fields:
                raise ValueError(
                    "adapter does not apply subject fields: "
                    + ", ".join(unapplied_fields)
                )
            subject_command = self.adapter.subject_command(subject, case)
            readiness_command = self.adapter.readiness_command(subject, case)
            if not subject_command:
                raise ValueError("adapter returned an empty subject command")
            if not readiness_command:
                raise ValueError("adapter returned an empty readiness command")
        except Exception as exc:
            return PreflightResult.no_verdict(f"adapter configuration failed: {str(exc)[:1000]}")

        assert case.bag is not None
        info = self._backend.run(
            ["ros2", "bag", "info", str(case.bag)],
            timeout_s=30,
        )
        if info.returncode != 0:
            detail = info.stderr or info.stdout or f"exit {info.returncode}"
            return PreflightResult.no_verdict(f"could not inspect input bag: {detail[:1000]}")
        present_topics: set[str] = set()
        for line in info.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("/"):
                present_topics.add(stripped)
                continue
            match = re.search(r"(?:^|\s)Topic:\s*([^\s|]+)", line)
            if match:
                present_topics.add(match.group(1))
        missing_topics = sorted(set(case.required_topics) - present_topics)
        if missing_topics:
            return PreflightResult.no_verdict(
                f"input bag is missing required topics: {', '.join(missing_topics)}"
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
            self._run_repeat(
                subject,
                case,
                output_dir,
                unit_id=f"repeat-{index:03d}",
            )
            for index in range(case.repeats)
        ]
        valid = [outcome for outcome in outcomes if outcome.validity is ExecutionValidity.VALID]
        artifacts = [artifact for outcome in outcomes for artifact in outcome.artifacts]
        invalid_reasons = [
            f"{outcome.unit_id}: {reason}"
            for outcome in outcomes
            for reason in outcome.reasons
        ]
        required = self.suite.gate.min_valid_repeats
        if len(valid) < required:
            return CaseExecution(
                subject=subject,
                case_id=case.id,
                runner="ros2_replay",
                validity=ExecutionValidity.NO_VERDICT,
                reasons=[
                    f"insufficient valid repeats: valid={len(valid)}, required={required}",
                    *invalid_reasons,
                ],
                valid_units=len(valid),
                unit_ids=[outcome.unit_id for outcome in valid],
                artifacts=artifacts,
                environment=self._environment(case),
            )

        try:
            metrics = self._aggregate_metrics(valid)
        except ValueError as exc:
            return CaseExecution(
                subject=subject,
                case_id=case.id,
                runner="ros2_replay",
                validity=ExecutionValidity.NO_VERDICT,
                reasons=[str(exc)],
                valid_units=len(valid),
                unit_ids=[outcome.unit_id for outcome in valid],
                artifacts=artifacts,
                environment=self._environment(case),
            )

        return CaseExecution(
            subject=subject,
            case_id=case.id,
            runner="ros2_replay",
            validity=ExecutionValidity.VALID,
            reasons=invalid_reasons,
            metrics=metrics,
            valid_units=len(valid),
            unit_ids=[outcome.unit_id for outcome in valid],
            artifacts=artifacts,
            environment=self._environment(case),
        )

    def _run_repeat(
        self,
        subject: SubjectRef,
        case: CaseSpec,
        output_dir: Path,
        *,
        unit_id: str,
    ) -> RepeatOutcome:
        repeat_dir = output_dir / "repeats" / unit_id
        repeat_dir.mkdir(parents=True, exist_ok=True)
        capture_dir = repeat_dir / "capture"
        subject_process: ManagedProcess | None = None
        recorder_process: ManagedProcess | None = None
        recorder_stop_result: CommandResult | None = None
        start_time = self._wall_time()
        outcome = RepeatOutcome(unit_id=unit_id, validity=ExecutionValidity.NO_VERDICT)

        try:
            subject_process = self._backend.start(
                self.adapter.subject_command(subject, case)
            )
            deadline = self._monotonic() + self.suite.execution.readiness_timeout_s
            while True:
                if subject_process.poll() is not None:
                    outcome.reasons.append("subject exited before readiness")
                    return outcome
                probe = self._backend.run(
                    self.adapter.readiness_command(subject, case),
                    timeout_s=2,
                )
                if probe.returncode == 0:
                    break
                if self._monotonic() >= deadline:
                    outcome.reasons.append("readiness timeout before playback")
                    return outcome
                self._sleep(0.1)

            recorder_process = self._backend.start(
                [
                    "ros2",
                    "bag",
                    "record",
                    "-s",
                    "mcap",
                    "-o",
                    str(capture_dir),
                    "--topics",
                    *case.capture_topics,
                ]
            )
            recorder_deadline = (
                self._monotonic() + self.suite.execution.readiness_timeout_s
            )
            while True:
                if subject_process.poll() is not None:
                    outcome.reasons.append(
                        "subject exited while waiting for recorder readiness"
                    )
                    return outcome
                if recorder_process.poll() is not None:
                    outcome.reasons.append("recorder exited before readiness")
                    return outcome
                recorder_ready = True
                for topic in case.capture_topics:
                    probe = self._backend.run(
                        ["ros2", "topic", "info", topic],
                        timeout_s=2,
                    )
                    subscriptions = re.search(
                        r"Subscription count:\s*([0-9]+)", probe.stdout
                    )
                    if (
                        probe.returncode != 0
                        or subscriptions is None
                        or int(subscriptions.group(1)) < 1
                    ):
                        recorder_ready = False
                        break
                if recorder_ready:
                    break
                if self._monotonic() >= recorder_deadline:
                    outcome.reasons.append(
                        "recorder readiness timeout before playback"
                    )
                    return outcome
                self._sleep(0.1)
            assert case.bag is not None
            playback = self._backend.run(
                [
                    "ros2",
                    "bag",
                    "play",
                    str(case.bag),
                    "--topics",
                    *(case.playback_topics or case.required_topics),
                    "--clock",
                    "1000",
                    "--delay",
                    str(self.suite.execution.playback_discovery_delay_s),
                    "--wait-for-all-acked",
                    "5000",
                ],
                timeout_s=self.suite.execution.playback_timeout_s,
            )
            if playback.returncode != 0:
                detail = playback.stderr or playback.stdout or f"exit {playback.returncode}"
                outcome.reasons.append(f"playback failed: {detail[:1000]}")
                return outcome
            # Let messages already queued in the ROS graph reach the recorder.
            self._sleep(0.5)
            subject_exit = subject_process.poll()
            if subject_exit is not None:
                outcome.reasons.append(
                    f"subject exited during playback with code {subject_exit}"
                )
                return outcome
            recorder_exit = recorder_process.poll()
            if recorder_exit is not None:
                outcome.reasons.append(
                    f"recorder exited during playback with code {recorder_exit}"
                )
                return outcome
        finally:
            if recorder_process is not None:
                recorder_stop_result = recorder_process.stop(timeout_s=5)
            if subject_process is not None:
                subject_process.stop(timeout_s=5)

        if recorder_stop_result is None or recorder_stop_result.returncode not in {
            0,
            -signal.SIGINT,
            -signal.SIGTERM,
        }:
            code = (
                recorder_stop_result.returncode
                if recorder_stop_result is not None
                else "missing"
            )
            outcome.reasons.append(f"recorder finalization failed with code {code}")
            return outcome

        metadata = capture_dir / "metadata.yaml"
        if not metadata.is_file() or metadata.stat().st_size == 0:
            outcome.reasons.append("capture recorder metadata is missing or empty")
            return outcome
        try:
            captured_counts = _metadata_topic_message_counts(metadata)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            outcome.reasons.append(f"capture recorder metadata is invalid: {str(exc)[:1000]}")
            return outcome
        empty_topics = [
            topic for topic in case.capture_topics if captured_counts.get(topic, 0) <= 0
        ]
        if empty_topics:
            outcome.reasons.append(
                "capture topics contain no messages: " + ", ".join(empty_topics)
            )
            return outcome

        captures = sorted(capture_dir.rglob("*.mcap")) if capture_dir.exists() else []
        if not captures:
            outcome.reasons.append("capture output is missing")
            return outcome
        fresh = [path for path in captures if path.stat().st_mtime >= start_time]
        if not fresh:
            outcome.reasons.append("capture output is not fresh")
            return outcome

        try:
            analysis = _normalize_analysis(
                self.adapter.analyze(subject, case, capture_dir)
            )
            trace_path = repeat_dir / "trace.json"
            if analysis.trace is not None:
                trace_path.write_text(analysis.trace.model_dump_json(indent=2))
        except Exception as exc:
            outcome.reasons.append(f"capture analysis failed: {str(exc)[:1000]}")
            return outcome

        assert case.bag is not None
        input_copy = repeat_dir / "input.mcap"
        try:
            source_hash = _bag_sha256(case.bag)
            shutil.copyfile(case.bag, input_copy)
            copied_hash = _bag_sha256(input_copy)
        except OSError as exc:
            outcome.reasons.append(
                f"input MCAP artifact could not be materialized: {str(exc)[:1000]}"
            )
            return outcome
        if copied_hash != source_hash:
            outcome.reasons.append("input MCAP artifact hash differs from replay source")
            return outcome

        outcome.validity = ExecutionValidity.VALID
        outcome.metrics = analysis.metrics
        outcome.artifacts = [
            ArtifactRef(
                kind="input_mcap",
                path=input_copy.relative_to(output_dir).as_posix(),
            ),
            *[
                ArtifactRef(kind="mcap", path=path.relative_to(output_dir).as_posix())
                for path in fresh
            ],
        ]
        if analysis.trace is not None:
            outcome.artifacts.append(
                ArtifactRef(
                    kind="trace",
                    path=trace_path.relative_to(output_dir).as_posix(),
                )
            )
        return outcome

    @staticmethod
    def _aggregate_metrics(outcomes: list[RepeatOutcome]) -> dict[str, MetricValue]:
        key_sets = [set(outcome.metrics) for outcome in outcomes]
        if not key_sets or any(keys != key_sets[0] for keys in key_sets[1:]):
            raise ValueError("valid repeats returned different metric sets")

        aggregated: dict[str, MetricValue] = {}
        for key in sorted(key_sets[0]):
            values = [outcome.metrics[key] for outcome in outcomes]
            first = values[0]
            if isinstance(first, bool):
                if not all(isinstance(value, bool) for value in values):
                    raise ValueError(f"metric {key!r} has mixed types")
                aggregated[key] = all(values)
            elif isinstance(first, (int, float)):
                if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
                    raise ValueError(f"metric {key!r} has mixed types")
                aggregated[key] = sum(float(value) for value in values) / len(values)
            elif isinstance(first, str):
                if any(not isinstance(value, str) for value in values) or len(set(values)) != 1:
                    raise ValueError(f"metric {key!r} has incompatible values")
                aggregated[key] = first
            else:
                raise ValueError(f"metric {key!r} has unsupported type")
        return aggregated

    def _environment(self, case: CaseSpec) -> dict[str, str]:
        assert case.bag is not None
        return {
            "runner": "ros2_replay",
            "adapter": self.suite.runner.adapter or self.adapter.__class__.__qualname__,
            "input_sha256": _bag_sha256(case.bag),
        }
