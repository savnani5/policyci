"""ROS stack commands and exact action scoring for the ALOHA replay demo."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from policyci.checkspec import CaseSpec
from policyci.examples.aloha_replay.action_processor import (
    ALOHA_JOINT_NAMES,
    GRIPPER_INDICES,
)
from policyci.schemas import ReplayAnalysis, ReplayTrace, SignalSeries, SubjectRef

REFERENCE_ACTION_TOPIC = "/episode/reference_action"
OUTPUT_ACTION_TOPIC = "/policy/action"
SOURCE_SAMPLE_RATE_HZ = 50.0


def _validate_tolerance(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _validated_actions(rows: Sequence[Sequence[float]], *, name: str) -> list[list[float]]:
    validated: list[list[float]] = []
    for row in rows:
        if len(row) != len(ALOHA_JOINT_NAMES):
            raise ValueError(f"{name} action must contain exactly 14 values")
        values = [float(value) for value in row]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"{name} action values must be finite")
        validated.append(values)
    return validated


def score_aloha_samples(
    *,
    reference: list[list[float]],
    observed: list[list[float]],
    time_s: list[float],
    gripper_tolerance: float,
    arm_tolerance: float,
) -> ReplayAnalysis:
    """Compare fresh output actions against the recorded ALOHA reference."""
    if not reference or not observed:
        raise ValueError(
            f"input/output message count differs: input={len(reference)}, "
            f"output={len(observed)}"
        )
    if len(reference) != len(observed):
        raise ValueError(
            f"input/output message count differs: input={len(reference)}, "
            f"output={len(observed)}"
        )
    if len(time_s) != len(reference):
        raise ValueError(
            f"timestamp count differs: timestamps={len(time_s)}, "
            f"messages={len(reference)}"
        )

    reference_rows = _validated_actions(reference, name="reference")
    observed_rows = _validated_actions(observed, name="observed")
    times = [float(value) for value in time_s]
    if any(not math.isfinite(value) or value < 0 for value in times):
        raise ValueError("timestamps must be finite and non-negative")
    if any(current <= previous for previous, current in zip(times, times[1:])):
        raise ValueError("timestamps must be strictly increasing")

    gripper_limit = _validate_tolerance(gripper_tolerance, name="gripper_tolerance")
    arm_limit = _validate_tolerance(arm_tolerance, name="arm_tolerance")
    arm_indices = tuple(
        index for index in range(len(ALOHA_JOINT_NAMES)) if index not in GRIPPER_INDICES
    )

    left_expected = [row[GRIPPER_INDICES[0]] for row in reference_rows]
    right_expected = [row[GRIPPER_INDICES[1]] for row in reference_rows]
    left_observed = [row[GRIPPER_INDICES[0]] for row in observed_rows]
    right_observed = [row[GRIPPER_INDICES[1]] for row in observed_rows]
    left_error = [
        abs(actual - expected)
        for expected, actual in zip(left_expected, left_observed, strict=True)
    ]
    right_error = [
        abs(actual - expected)
        for expected, actual in zip(right_expected, right_observed, strict=True)
    ]
    max_gripper_error = [
        max(left, right) for left, right in zip(left_error, right_error, strict=True)
    ]
    max_arm_error = [
        max(abs(actual[index] - expected[index]) for index in arm_indices)
        for expected, actual in zip(reference_rows, observed_rows, strict=True)
    ]
    violating = [
        gripper_error > gripper_limit or arm_error > arm_limit
        for gripper_error, arm_error in zip(
            max_gripper_error,
            max_arm_error,
            strict=True,
        )
    ]
    first_violation = next(
        (index for index, failed in enumerate(violating) if failed),
        -1,
    )

    return ReplayAnalysis(
        metrics={
            "within_tolerance": not any(violating),
            "max_gripper_error": max(max_gripper_error),
            "max_arm_error": max(max_arm_error),
            "message_count": len(reference_rows),
            "first_violation_index": first_violation,
            "first_violation_time_s": (
                times[first_violation] if first_violation >= 0 else -1.0
            ),
        },
        trace=ReplayTrace(
            trace_type="aloha_action",
            series={
                "time_s": SignalSeries(values=times, unit="s"),
                "left_gripper_expected": SignalSeries(values=left_expected),
                "left_gripper_observed": SignalSeries(values=left_observed),
                "right_gripper_expected": SignalSeries(values=right_expected),
                "right_gripper_observed": SignalSeries(values=right_observed),
                "left_gripper_error": SignalSeries(values=left_error),
                "right_gripper_error": SignalSeries(values=right_error),
                "max_gripper_error": SignalSeries(values=max_gripper_error),
                "max_arm_error": SignalSeries(values=max_arm_error),
            },
        ),
    )


def _read_action_samples(path: Path, topic: str) -> tuple[list[list[float]], list[float]]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError("ROS 2 Python libraries are unavailable") from exc

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    topic_types = {
        metadata.name: metadata.type for metadata in reader.get_all_topics_and_types()
    }
    if topic not in topic_types:
        raise ValueError(f"capture is missing topic {topic}")
    message_type = get_message(topic_types[topic])
    samples: list[list[float]] = []
    timestamps_ns: list[int] = []
    while reader.has_next():
        name, data, timestamp_ns = reader.read_next()
        if name != topic:
            continue
        message = deserialize_message(data, message_type)
        samples.append([float(value) for value in message.data])
        timestamps_ns.append(int(timestamp_ns))
    if not timestamps_ns:
        return samples, []
    first = timestamps_ns[0]
    return samples, [(timestamp - first) / 1_000_000_000 for timestamp in timestamps_ns]


class AlohaReplayAdapter:
    """Run and score the deterministic ALOHA action component."""

    def applied_subject_fields(self, subject: SubjectRef, case: CaseSpec) -> set[str]:
        del case
        return (
            {"config.gripper_delay_messages"}
            if "gripper_delay_messages" in subject.config
            else set()
        )

    def subject_command(self, subject: SubjectRef, case: CaseSpec) -> list[str]:
        sample_rate = float(case.judge.get("sample_rate_hz", 0.0))
        if not math.isclose(sample_rate, SOURCE_SAMPLE_RATE_HZ, abs_tol=1e-9):
            raise ValueError("ALOHA replay requires a 50 Hz source fixture")
        delay = subject.config.get("gripper_delay_messages", 0)
        if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
            raise ValueError("config.gripper_delay_messages must be a non-negative integer")
        return [
            "python3",
            "-m",
            "policyci.examples.aloha_replay.ros_node",
            "--gripper-delay-messages",
            str(delay),
        ]

    def readiness_command(self, subject: SubjectRef, case: CaseSpec) -> list[str]:
        del subject
        if case.capture_topics != [OUTPUT_ACTION_TOPIC]:
            raise ValueError(f"ALOHA replay must capture only {OUTPUT_ACTION_TOPIC}")
        return ["ros2", "topic", "info", REFERENCE_ACTION_TOPIC]

    def analyze(
        self,
        subject: SubjectRef,
        case: CaseSpec,
        capture: Path,
    ) -> ReplayAnalysis:
        del subject
        if case.bag is None:
            raise ValueError("ALOHA replay case is missing its input bag")
        capture_files = sorted(capture.rglob("*.mcap"))
        if len(capture_files) != 1:
            raise ValueError(
                f"ALOHA replay requires one capture MCAP, found {len(capture_files)}"
            )
        reference, time_s = _read_action_samples(case.bag, REFERENCE_ACTION_TOPIC)
        observed, _observed_time_s = _read_action_samples(
            capture_files[0],
            OUTPUT_ACTION_TOPIC,
        )
        return score_aloha_samples(
            reference=reference,
            observed=observed,
            time_s=time_s,
            gripper_tolerance=float(case.judge["gripper_tolerance"]),
            arm_tolerance=float(case.judge["arm_tolerance"]),
        )
