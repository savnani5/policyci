"""Self-contained Rerun dashboard for sensor-rich ALOHA replay evidence."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from policyci.aloha_visualization import AlohaReplayVisualization
from policyci.examples.aloha_replay.action_processor import (
    ALOHA_JOINT_NAMES,
    GRIPPER_INDICES,
)

CAMERA_TOPICS = {
    "/episode/cam_high/image/compressed": "overhead",
    "/episode/cam_left_wrist/image/compressed": "left_wrist",
    "/episode/cam_right_wrist/image/compressed": "right_wrist",
}
JOINT_TOPIC = "/episode/joint_states"
ACTION_TOPIC = "/episode/reference_action"
TASK_TOPIC = "/episode/task"
EXPECTED_TOPIC_TYPES = {
    **{topic: "sensor_msgs/msg/CompressedImage" for topic in CAMERA_TOPICS},
    JOINT_TOPIC: "sensor_msgs/msg/JointState",
    ACTION_TOPIC: "std_msgs/msg/Float64MultiArray",
    TASK_TOPIC: "std_msgs/msg/String",
}


@dataclass(frozen=True)
class AlohaMcapEvent:
    """One validated event on the shared episode timeline."""

    topic: str
    time_s: float
    jpeg: bytes | None = None
    joint_positions: tuple[float, ...] | None = None
    joint_efforts: tuple[float, ...] | None = None
    action: tuple[float, ...] | None = None
    task: str | None = None


def _rerun_modules():
    try:
        import rerun as rr
        import rerun.blueprint as rrb
        import rerun.blueprint.components as rrbc
    except ImportError as exc:
        raise RuntimeError(
            "Rerun visualization is not installed; run `uv sync --extra visualize`"
        ) from exc
    return rr, rrb, rrbc


def _event_from_message(
    topic: str,
    message: object,
    *,
    time_s: float,
) -> AlohaMcapEvent:
    if topic in CAMERA_TOPICS:
        from PIL import Image

        payload = bytes(message.data)
        if not payload:
            raise ValueError(f"empty JPEG payload on {topic}")
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid JPEG payload on {topic}: {exc}") from exc
        return AlohaMcapEvent(topic=topic, time_s=time_s, jpeg=payload)
    if topic == JOINT_TOPIC:
        names = tuple(str(name) for name in message.name)
        positions = tuple(float(value) for value in message.position)
        efforts = tuple(float(value) for value in message.effort)
        if names != ALOHA_JOINT_NAMES:
            raise ValueError("ALOHA JointState names differ from the declared 14 joints")
        if len(positions) != len(ALOHA_JOINT_NAMES):
            raise ValueError("ALOHA JointState position must contain 14 values")
        if len(efforts) != len(ALOHA_JOINT_NAMES):
            raise ValueError("ALOHA JointState effort must contain 14 values")
        if any(not math.isfinite(value) for value in (*positions, *efforts)):
            raise ValueError("ALOHA JointState values must be finite")
        return AlohaMcapEvent(
            topic=topic,
            time_s=time_s,
            joint_positions=positions,
            joint_efforts=efforts,
        )
    if topic == ACTION_TOPIC:
        action = tuple(float(value) for value in message.data)
        if len(action) != len(ALOHA_JOINT_NAMES):
            raise ValueError("ALOHA reference action must contain 14 values")
        if any(not math.isfinite(value) for value in action):
            raise ValueError("ALOHA reference action values must be finite")
        return AlohaMcapEvent(topic=topic, time_s=time_s, action=action)
    if topic == TASK_TOPIC:
        task = str(message.data).strip()
        if not task:
            raise ValueError("ALOHA task label must not be empty")
        return AlohaMcapEvent(topic=topic, time_s=time_s, task=task)
    raise ValueError(f"unsupported ALOHA MCAP topic: {topic}")


def iter_aloha_mcap(path: Path) -> Iterator[AlohaMcapEvent]:
    """Deserialize and validate the six declared ALOHA topics lazily."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError("ALOHA Rerun rendering requires a ROS 2 environment") from exc

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    available = {
        metadata.name: metadata.type for metadata in reader.get_all_topics_and_types()
    }
    for topic, expected_type in EXPECTED_TOPIC_TYPES.items():
        actual_type = available.get(topic)
        if actual_type != expected_type:
            raise ValueError(
                f"ALOHA input MCAP topic {topic} has type {actual_type!r}; "
                f"expected {expected_type!r}"
            )
    message_types = {
        topic: get_message(message_type)
        for topic, message_type in EXPECTED_TOPIC_TYPES.items()
    }
    last_timestamp_by_topic: dict[str, int] = {}
    seen_counts = {topic: 0 for topic in EXPECTED_TOPIC_TYPES}
    buffered: list[tuple[str, object, int]] = []
    first_reference_ns: int | None = None

    def convert(topic: str, message: object, timestamp_ns: int) -> AlohaMcapEvent:
        assert first_reference_ns is not None
        relative = (timestamp_ns - first_reference_ns) / 1_000_000_000
        if relative < -1e-9:
            raise ValueError("ALOHA MCAP contains events before its reference timeline")
        return _event_from_message(
            topic,
            message,
            time_s=max(0.0, relative),
        )

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        if topic not in EXPECTED_TOPIC_TYPES:
            continue
        timestamp_ns = int(timestamp_ns)
        previous = last_timestamp_by_topic.get(topic)
        if previous is not None and timestamp_ns <= previous:
            raise ValueError(f"non-monotonic timestamps on ALOHA topic {topic}")
        last_timestamp_by_topic[topic] = timestamp_ns
        seen_counts[topic] += 1
        message = deserialize_message(data, message_types[topic])
        if first_reference_ns is None:
            buffered.append((topic, message, timestamp_ns))
            if topic != ACTION_TOPIC:
                continue
            first_reference_ns = timestamp_ns
            for buffered_topic, buffered_message, buffered_timestamp in buffered:
                yield convert(
                    buffered_topic,
                    buffered_message,
                    buffered_timestamp,
                )
            buffered.clear()
            continue
        yield convert(topic, message, timestamp_ns)

    missing = [topic for topic, count in seen_counts.items() if count <= 0]
    if missing:
        raise ValueError("ALOHA input MCAP is missing messages on: " + ", ".join(missing))
    if first_reference_ns is None:
        raise ValueError("ALOHA input MCAP contains no reference action timeline")


def _blueprint(rrb, rrbc, model: AlohaReplayVisualization):
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.TextDocumentView(
                origin="/summary",
                name="Automatic verdict and scope",
            ),
            rrb.Horizontal(
                rrb.Spatial2DView(
                    origin="/sensors/overhead",
                    name="Overhead camera",
                    background=[12, 16, 24],
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(
                        origin="/sensors/left_wrist",
                        name="Left wrist camera",
                        background=[12, 16, 24],
                    ),
                    rrb.Spatial2DView(
                        origin="/sensors/right_wrist",
                        name="Right wrist camera",
                        background=[12, 16, 24],
                    ),
                    row_shares=[1, 1],
                ),
                column_shares=[2, 1],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    origin="/gripper_commands",
                    name="Reference vs baseline vs delayed candidate",
                ),
                rrb.Vertical(
                    rrb.TimeSeriesView(
                        origin="/effort",
                        name="Recorded gripper effort",
                    ),
                    rrb.TimeSeriesView(
                        origin="/error",
                        name="Candidate error vs immutable tolerance",
                    ),
                    row_shares=[1, 1],
                ),
                column_shares=[2, 1],
            ),
            row_shares=[1, 7, 4],
        ),
        rrb.TimePanel(
            state="collapsed",
            timeline=model.timeline,
            playback_speed=1.0,
            fps=50.0,
            play_state=rrbc.PlayState.Playing,
            loop_mode=rrbc.LoopMode.Selection,
            time_selection=rrbc.AbsoluteTimeRange(
                min=int(model.time_s[0] * 1_000_000_000),
                max=int(model.time_s[-1] * 1_000_000_000),
            ),
        ),
        collapse_panels=True,
    )


def _log_styles(recording, rr) -> None:
    styles = {
        "/gripper_commands/left/reference": ([160, 168, 178], "Left reference"),
        "/gripper_commands/left/baseline": ([57, 217, 138], "Left baseline"),
        "/gripper_commands/left/candidate": ([255, 92, 105], "Left candidate"),
        "/gripper_commands/right/reference": ([205, 210, 220], "Right reference"),
        "/gripper_commands/right/baseline": ([135, 240, 190], "Right baseline"),
        "/gripper_commands/right/candidate": ([255, 158, 168], "Right candidate"),
        "/effort/left_gripper": ([100, 170, 255], "Recorded left effort"),
        "/effort/right_gripper": ([244, 189, 80], "Recorded right effort"),
        "/error/baseline": ([57, 217, 138], "Baseline max gripper error"),
        "/error/candidate": ([255, 92, 105], "Candidate max gripper error"),
        "/error/tolerance": ([244, 189, 80], "Gripper tolerance"),
    }
    for entity, (color, name) in styles.items():
        recording.log(
            entity,
            rr.SeriesLines(colors=color, names=name),
            static=True,
        )


def _matches(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def render_aloha_rerun(
    model: AlohaReplayVisualization,
    output_path: Path,
) -> Path:
    """Write one portable dashboard containing cameras, signals, and verdict."""
    rr, rrb, rrbc = _rerun_modules()
    output = output_path.resolve()
    if not output.is_relative_to(model.check_dir.resolve()):
        raise ValueError(f"Rerun output is not confined to the check directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    recording_id = f"{model.check_id}-{model.case_id}-{model.unit_id}"

    with rr.RecordingStream(
        "robolens_aloha_replay_check",
        recording_id=recording_id,
        send_properties=False,
    ) as recording:
        recording.save(output, default_blueprint=_blueprint(rrb, rrbc, model))
        recording.log(
            "/summary",
            rr.TextDocument(model.summary_markdown, media_type="text/markdown"),
            static=True,
        )
        _log_styles(recording, rr)

        camera_counts = {camera: 0 for camera in CAMERA_TOPICS.values()}
        action_index = 0
        joint_count = 0
        max_effort: tuple[float, float] | None = None
        for event in iter_aloha_mcap(model.input_mcap):
            recording.set_time(model.timeline, duration=event.time_s)
            if event.topic in CAMERA_TOPICS:
                camera = CAMERA_TOPICS[event.topic]
                if event.jpeg is None:
                    raise ValueError(f"missing JPEG evidence on {event.topic}")
                if camera_counts[camera] % 5 == 0:
                    recording.log(
                        f"/sensors/{camera}",
                        rr.EncodedImage(
                            contents=event.jpeg,
                            media_type="image/jpeg",
                        ),
                    )
                camera_counts[camera] += 1
            elif event.topic == JOINT_TOPIC:
                if event.joint_efforts is None:
                    raise ValueError("joint event is missing effort evidence")
                left_effort = event.joint_efforts[GRIPPER_INDICES[0]]
                right_effort = event.joint_efforts[GRIPPER_INDICES[1]]
                recording.log("/effort/left_gripper", rr.Scalars(left_effort))
                recording.log("/effort/right_gripper", rr.Scalars(right_effort))
                magnitude = max(abs(left_effort), abs(right_effort))
                if max_effort is None or magnitude > max_effort[0]:
                    max_effort = (magnitude, event.time_s)
                joint_count += 1
            elif event.topic == ACTION_TOPIC:
                if event.action is None or action_index >= len(model.time_s):
                    raise ValueError("reference action count exceeds the replay trace")
                expected_time = model.time_s[action_index]
                if not math.isclose(event.time_s, expected_time, abs_tol=2e-6):
                    raise ValueError(
                        f"reference action time differs at sample {action_index}"
                    )
                if not _matches(
                    event.action[GRIPPER_INDICES[0]],
                    model.left_gripper_expected[action_index],
                ) or not _matches(
                    event.action[GRIPPER_INDICES[1]],
                    model.right_gripper_expected[action_index],
                ):
                    raise ValueError(
                        f"input MCAP reference action differs at sample {action_index}"
                    )
                action_index += 1
            elif event.topic == TASK_TOPIC:
                if event.task is None:
                    raise ValueError("task event is missing its label")
                recording.log(
                    "/events/task",
                    rr.TextLog(f"Task: {event.task}"),
                )

        expected_count = len(model.time_s)
        if action_index != expected_count:
            raise ValueError(
                f"reference action count differs: MCAP={action_index}, trace={expected_count}"
            )
        if joint_count != expected_count:
            raise ValueError(
                f"joint state count differs: MCAP={joint_count}, trace={expected_count}"
            )
        mismatched_cameras = {
            camera: count
            for camera, count in camera_counts.items()
            if count != expected_count
        }
        if mismatched_cameras:
            raise ValueError(
                f"camera frame counts differ from trace: {mismatched_cameras}"
            )

        first_change = next(
            (
                index
                for index in range(1, expected_count)
                if not _matches(
                    model.left_gripper_expected[index],
                    model.left_gripper_expected[index - 1],
                )
                or not _matches(
                    model.right_gripper_expected[index],
                    model.right_gripper_expected[index - 1],
                )
            ),
            None,
        )
        for index, time_s in enumerate(model.time_s):
            recording.set_time(model.timeline, duration=time_s)
            recording.log(
                "/gripper_commands/left/reference",
                rr.Scalars(model.left_gripper_expected[index]),
            )
            recording.log(
                "/gripper_commands/left/baseline",
                rr.Scalars(model.baseline_left_gripper_observed[index]),
            )
            recording.log(
                "/gripper_commands/left/candidate",
                rr.Scalars(model.candidate_left_gripper_observed[index]),
            )
            recording.log(
                "/gripper_commands/right/reference",
                rr.Scalars(model.right_gripper_expected[index]),
            )
            recording.log(
                "/gripper_commands/right/baseline",
                rr.Scalars(model.baseline_right_gripper_observed[index]),
            )
            recording.log(
                "/gripper_commands/right/candidate",
                rr.Scalars(model.candidate_right_gripper_observed[index]),
            )
            recording.log(
                "/error/baseline",
                rr.Scalars(model.baseline_max_gripper_error[index]),
            )
            recording.log(
                "/error/candidate",
                rr.Scalars(model.candidate_max_gripper_error[index]),
            )
            recording.log("/error/tolerance", rr.Scalars(model.gripper_tolerance))
            if index == first_change:
                recording.log(
                    "/events/first_gripper_change",
                    rr.TextLog("First material recorded gripper command change"),
                )
            if index == model.first_candidate_violation_index:
                recording.log(
                    "/events/first_candidate_violation",
                    rr.TextLog(
                        "Automatic gate violation: candidate gripper error exceeds tolerance"
                    ),
                )
        if max_effort is not None:
            recording.set_time(model.timeline, duration=max_effort[1])
            recording.log(
                "/events/max_recorded_gripper_effort",
                rr.TextLog(f"Maximum recorded gripper effort magnitude: {max_effort[0]:.3f}"),
            )

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Rerun did not produce a recording: {output}")
    return output
