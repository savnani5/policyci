"""Stack-specific commands and metric logic for the Twist replay example."""

from __future__ import annotations

from pathlib import Path

from policyci.checkspec import CaseSpec
from policyci.schemas import (
    ReplayAnalysis,
    ReplayTrace,
    SignalSeries,
    SubjectRef,
)


def linear_x(message: object) -> float:
    """Return linear.x from either Twist or TwistStamped."""
    twist = getattr(message, "twist", message)
    linear = getattr(twist, "linear", None)
    if linear is None or not hasattr(linear, "x"):
        raise TypeError("message is neither Twist nor TwistStamped")
    return float(linear.x)


def score_twist_samples(
    *,
    input_x: list[float],
    output_x: list[float],
    tolerance: float,
) -> ReplayAnalysis:
    if not input_x or not output_x:
        raise ValueError("empty input or output samples")
    if len(input_x) != len(output_x):
        raise ValueError(
            f"input/output message count differs: input={len(input_x)}, output={len(output_x)}"
        )
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    errors = [abs(output - expected) for expected, output in zip(input_x, output_x)]
    return ReplayAnalysis(
        metrics={
            "within_tolerance": all(error <= tolerance for error in errors),
            "mean_abs_error": sum(errors) / len(errors),
            "max_abs_error": max(errors),
            "message_count": len(errors),
        },
        trace=ReplayTrace(
            series={
                "expected": SignalSeries(values=list(input_x), unit="m/s"),
                "observed": SignalSeries(values=list(output_x), unit="m/s"),
                "absolute_error": SignalSeries(values=errors, unit="m/s"),
            }
        ),
    )


def _read_twist_x(path: Path, topic: str) -> list[float]:
    """Read Twist.linear.x lazily so the adapter imports without a ROS installation."""
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
    values = []
    while reader.has_next():
        name, data, _timestamp = reader.read_next()
        if name != topic:
            continue
        message = deserialize_message(data, message_type)
        values.append(linear_x(message))
    return values


class TwistReplayAdapter:
    def applied_subject_fields(self, subject: SubjectRef, case: CaseSpec) -> set[str]:
        del case
        fields = {"config.gain", "config.bias"}
        if "bias_after_messages" in subject.config:
            fields.add("config.bias_after_messages")
        return fields

    def subject_command(self, subject: SubjectRef, case: CaseSpec) -> list[str]:
        del case
        gain = float(subject.config.get("gain", 1.0))
        bias = float(subject.config.get("bias", 0.0))
        command = [
            "python3",
            "-m",
            "policyci.examples.ros2_replay.twist_processor",
            "--gain",
            str(gain),
            "--bias",
            str(bias),
        ]
        if "bias_after_messages" in subject.config:
            threshold = subject.config["bias_after_messages"]
            if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
                raise ValueError("config.bias_after_messages must be a non-negative integer")
            command.extend(["--bias-after-messages", str(threshold)])
        return command

    def readiness_command(self, subject: SubjectRef, case: CaseSpec) -> list[str]:
        del subject
        if not case.capture_topics:
            raise ValueError("Twist replay requires one capture topic")
        return ["ros2", "topic", "info", case.capture_topics[0]]

    def analyze(
        self,
        subject: SubjectRef,
        case: CaseSpec,
        capture: Path,
    ) -> ReplayAnalysis:
        if case.bag is None or not case.required_topics or not case.capture_topics:
            raise ValueError("Twist replay case is incomplete")
        capture_files = sorted(capture.rglob("*.mcap"))
        if not capture_files:
            raise ValueError("capture contains no MCAP file")
        input_x = _read_twist_x(case.bag, case.required_topics[0])
        output_x = _read_twist_x(capture_files[0], case.capture_topics[0])
        if "tolerance" not in case.judge:
            raise ValueError("Twist replay requires case.judge.tolerance")
        tolerance = float(case.judge["tolerance"])
        return score_twist_samples(
            input_x=input_x,
            output_x=output_x,
            tolerance=tolerance,
        )
