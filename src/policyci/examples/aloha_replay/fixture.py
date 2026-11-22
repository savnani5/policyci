"""Prepare one pinned LeRobot ALOHA episode as a ROS 2 MCAP fixture."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from policyci.examples.aloha_replay.action_processor import (
    ALOHA_JOINT_NAMES,
    GRIPPER_INDICES,
)

PINNED_REPO_ID = "lerobot/aloha_static_coffee"
PINNED_CAMERAS = (
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
)


class AlohaSourceManifest(BaseModel):
    """Immutable source selection for the conference fixture."""

    model_config = ConfigDict(extra="forbid")

    repo_id: str
    revision: str
    episode_index: int = Field(ge=0)
    sample_rate_hz: int
    cameras: tuple[str, ...]
    task_label: str = Field(min_length=1)
    license: str = Field(min_length=1)
    source_url: str = Field(min_length=1)

    @field_validator("revision")
    @classmethod
    def revision_is_immutable(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError("revision must be an immutable 40-character commit SHA")
        return value

    @model_validator(mode="after")
    def validate_pinned_source(self) -> "AlohaSourceManifest":
        if self.repo_id != PINNED_REPO_ID:
            raise ValueError(f"repo_id must be {PINNED_REPO_ID}")
        if self.sample_rate_hz != 50:
            raise ValueError("ALOHA source sample rate must be 50 Hz")
        if self.cameras != PINNED_CAMERAS:
            raise ValueError(f"cameras must be exactly {PINNED_CAMERAS}")
        return self


class AlohaFixtureSummary(BaseModel):
    """Integrity summary written beside the generated MCAP."""

    model_config = ConfigDict(extra="forbid")

    source: AlohaSourceManifest
    source_sha256: dict[str, str]
    generated_mcap_sha256: str
    topic_counts: dict[str, int]
    sample_count: int = Field(gt=0)
    duration_s: float = Field(gt=0)
    gripper_tolerance: float = Field(gt=0)


@dataclass(frozen=True)
class AlohaEpisodeData:
    """Robot signals loaded from one LeRobot parquet episode."""

    timestamps_s: list[float]
    states: list[list[float]]
    efforts: list[list[float]]
    actions: list[list[float]]


def selected_source_paths(manifest: AlohaSourceManifest) -> tuple[str, ...]:
    episode = f"episode_{manifest.episode_index:06d}"
    chunk = f"chunk-{manifest.episode_index // 1000:03d}"
    paths = [
        "README.md",
        "meta/info.json",
        f"data/{chunk}/{episode}.parquet",
    ]
    paths.extend(
        f"videos/{chunk}/observation.images.{camera}/{episode}.mp4"
        for camera in manifest.cameras
    )
    return tuple(paths)


def load_source_manifest(path: Path) -> AlohaSourceManifest:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid ALOHA source manifest {path}: {exc}") from exc
    return AlohaSourceManifest.model_validate(payload)


def validate_episode_alignment(
    *,
    sample_count: int,
    camera_frame_counts: dict[str, int],
    timestamps_s: list[float],
    sample_rate_hz: float,
) -> None:
    if sample_count <= 0:
        raise ValueError("sample count must be positive")
    mismatched = {
        camera: count
        for camera, count in camera_frame_counts.items()
        if count != sample_count
    }
    if mismatched:
        raise ValueError(
            f"camera frame count differs from samples: samples={sample_count}, "
            f"cameras={mismatched}"
        )
    if len(timestamps_s) != sample_count:
        raise ValueError(
            f"timestamp count differs from samples: timestamps={len(timestamps_s)}, "
            f"samples={sample_count}"
        )
    timestamps = [float(value) for value in timestamps_s]
    if any(not math.isfinite(value) or value < 0 for value in timestamps):
        raise ValueError("timestamps must be finite and non-negative")
    if any(
        current <= previous for previous, current in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("timestamps must be strictly increasing")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    expected_period = 1.0 / sample_rate_hz
    for previous, current in zip(timestamps, timestamps[1:]):
        if not math.isclose(
            current - previous,
            expected_period,
            rel_tol=0.02,
            abs_tol=1e-6,
        ):
            raise ValueError(f"timestamps do not match the declared {sample_rate_hz:g} Hz")


def recommended_gripper_tolerance(actions: list[list[float]]) -> float:
    if not actions:
        raise ValueError("actions must not be empty")
    ranges = []
    for index in GRIPPER_INDICES:
        values = []
        for action in actions:
            if len(action) != len(ALOHA_JOINT_NAMES):
                raise ValueError("ALOHA action must contain exactly 14 values")
            value = float(action[index])
            if not math.isfinite(value):
                raise ValueError("ALOHA action values must be finite")
            values.append(value)
        channel_range = max(values) - min(values)
        if channel_range <= 0:
            raise ValueError("both gripper channels require a non-zero range")
        ranges.append(channel_range)
    return round(min(ranges) * 0.10, 6)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_robot_rows(rows: list[object], *, name: str) -> list[list[float]]:
    validated: list[list[float]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != len(ALOHA_JOINT_NAMES):
            raise ValueError(f"{name} must contain exactly 14 values per sample")
        values = [float(value) for value in row]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"{name} values must be finite")
        validated.append(values)
    return validated


def read_episode_parquet(path: Path) -> AlohaEpisodeData:
    """Load the exact state, effort, action, and timestamp columns used by the demo."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("ALOHA fixture preparation requires pyarrow") from exc

    try:
        table = parquet.read_table(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read ALOHA episode parquet {path}: {exc}") from exc
    required = (
        "timestamp",
        "observation.state",
        "observation.effort",
        "action",
    )
    missing = [name for name in required if name not in table.column_names]
    if missing:
        raise ValueError(f"episode parquet is missing columns: {', '.join(missing)}")

    timestamps = [float(value) for value in table["timestamp"].to_pylist()]
    states = _validated_robot_rows(
        table["observation.state"].to_pylist(),
        name="observation.state",
    )
    efforts = _validated_robot_rows(
        table["observation.effort"].to_pylist(),
        name="observation.effort",
    )
    actions = _validated_robot_rows(table["action"].to_pylist(), name="action")
    sample_count = len(actions)
    if sample_count <= 0:
        raise ValueError("episode parquet contains no samples")
    lengths = {
        "timestamp": len(timestamps),
        "observation.state": len(states),
        "observation.effort": len(efforts),
        "action": sample_count,
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"episode parquet columns have different lengths: {lengths}")
    if any(not math.isfinite(value) or value < 0 for value in timestamps):
        raise ValueError("timestamps must be finite and non-negative")
    if any(
        current <= previous for previous, current in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("timestamps must be strictly increasing")
    return AlohaEpisodeData(
        timestamps_s=timestamps,
        states=states,
        efforts=efforts,
        actions=actions,
    )


def _decode_video_as_jpeg(path: Path) -> list[bytes]:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("ALOHA fixture preparation requires PyAV") from exc

    frames: list[bytes] = []
    try:
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                buffer = io.BytesIO()
                frame.to_image().convert("RGB").save(
                    buffer,
                    format="JPEG",
                    quality=85,
                    subsampling=2,
                    optimize=False,
                )
                frames.append(buffer.getvalue())
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not decode ALOHA camera video {path}: {exc}") from exc
    if not frames:
        raise ValueError(f"ALOHA camera video contains no frames: {path}")
    return frames


def _source_file(
    source_files: dict[str, Path],
    source_path: str,
) -> Path:
    try:
        path = source_files[source_path]
    except KeyError as exc:
        raise ValueError(f"source file mapping is missing {source_path}") from exc
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"source file is missing or empty: {source_path} ({path})")
    return path


def convert_source_files_to_mcap(
    source_files: dict[str, Path],
    manifest: AlohaSourceManifest,
    output: Path,
) -> AlohaFixtureSummary:
    """Convert one pinned ALOHA episode into a sensor-rich ROS 2 MCAP."""
    try:
        import rosbag2_py
        from rclpy.serialization import serialize_message
        from sensor_msgs.msg import CompressedImage, JointState
        from std_msgs.msg import Float64MultiArray, String
    except ImportError as exc:
        raise RuntimeError("ALOHA MCAP conversion requires a ROS 2 environment") from exc

    selected = selected_source_paths(manifest)
    resolved = {name: _source_file(source_files, name) for name in selected}
    episode_name = f"episode_{manifest.episode_index:06d}"
    chunk = f"chunk-{manifest.episode_index // 1000:03d}"
    episode = read_episode_parquet(
        resolved[f"data/{chunk}/{episode_name}.parquet"]
    )
    camera_frames = {
        camera: _decode_video_as_jpeg(
            resolved[
                f"videos/{chunk}/observation.images.{camera}/{episode_name}.mp4"
            ]
        )
        for camera in manifest.cameras
    }
    validate_episode_alignment(
        sample_count=len(episode.actions),
        camera_frame_counts={
            camera: len(frames) for camera, frames in camera_frames.items()
        },
        timestamps_s=episode.timestamps_s,
        sample_rate_hz=float(manifest.sample_rate_hz),
    )

    destination = output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    topic_types = {
        **{
            f"/episode/{camera}/image/compressed": "sensor_msgs/msg/CompressedImage"
            for camera in manifest.cameras
        },
        "/episode/joint_states": "sensor_msgs/msg/JointState",
        "/episode/reference_action": "std_msgs/msg/Float64MultiArray",
        "/episode/task": "std_msgs/msg/String",
    }
    topic_counts = {
        topic: (1 if topic == "/episode/task" else len(episode.actions))
        for topic in topic_types
    }
    start_ns = 1_000_000_000

    with tempfile.TemporaryDirectory(prefix="robolens-aloha-") as temp_dir:
        bag_uri = Path(temp_dir) / "bag"
        writer = rosbag2_py.SequentialWriter()
        writer.open(
            rosbag2_py.StorageOptions(uri=str(bag_uri), storage_id="mcap"),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr",
                output_serialization_format="cdr",
            ),
        )
        for topic_id, (topic, message_type) in enumerate(topic_types.items()):
            writer.create_topic(
                rosbag2_py.TopicMetadata(
                    id=topic_id,
                    name=topic,
                    type=message_type,
                    serialization_format="cdr",
                )
            )

        first_timestamp = episode.timestamps_s[0]
        task_timestamp_ns = start_ns
        task_message = String()
        task_message.data = manifest.task_label
        writer.write(
            "/episode/task",
            serialize_message(task_message),
            task_timestamp_ns,
        )
        for index, timestamp_s in enumerate(episode.timestamps_s):
            timestamp_ns = start_ns + round((timestamp_s - first_timestamp) * 1e9)
            stamp_sec, stamp_nanosec = divmod(timestamp_ns, 1_000_000_000)

            for camera in manifest.cameras:
                image = CompressedImage()
                image.header.stamp.sec = stamp_sec
                image.header.stamp.nanosec = stamp_nanosec
                image.header.frame_id = camera
                image.format = "jpeg"
                image.data = camera_frames[camera][index]
                writer.write(
                    f"/episode/{camera}/image/compressed",
                    serialize_message(image),
                    timestamp_ns,
                )

            joint_state = JointState()
            joint_state.header.stamp.sec = stamp_sec
            joint_state.header.stamp.nanosec = stamp_nanosec
            joint_state.name = list(ALOHA_JOINT_NAMES)
            joint_state.position = episode.states[index]
            joint_state.effort = episode.efforts[index]
            writer.write(
                "/episode/joint_states",
                serialize_message(joint_state),
                timestamp_ns,
            )

            action = Float64MultiArray()
            action.data = episode.actions[index]
            writer.write(
                "/episode/reference_action",
                serialize_message(action),
                timestamp_ns,
            )
        del writer

        generated = list(bag_uri.rglob("*.mcap"))
        if len(generated) != 1:
            raise RuntimeError(
                f"expected one generated MCAP, found {len(generated)} under {bag_uri}"
            )
        shutil.copyfile(generated[0], destination)

    return AlohaFixtureSummary(
        source=manifest,
        source_sha256={name: sha256_file(path) for name, path in resolved.items()},
        generated_mcap_sha256=sha256_file(destination),
        topic_counts=topic_counts,
        sample_count=len(episode.actions),
        duration_s=episode.timestamps_s[-1] - episode.timestamps_s[0],
        gripper_tolerance=recommended_gripper_tolerance(episode.actions),
    )


def download_source_files(
    manifest: AlohaSourceManifest,
    cache_dir: Path,
) -> dict[str, Path]:
    """Download only the immutable files required for the selected episode."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("ALOHA fixture preparation requires huggingface-hub") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        source_path: Path(
            hf_hub_download(
                repo_id=manifest.repo_id,
                filename=source_path,
                repo_type="dataset",
                revision=manifest.revision,
                cache_dir=cache_dir,
            )
        )
        for source_path in selected_source_paths(manifest)
    }


def prepare_fixture(
    manifest_path: Path,
    cache_dir: Path,
    output: Path,
) -> AlohaFixtureSummary:
    manifest = load_source_manifest(manifest_path)
    source_files = download_source_files(manifest, cache_dir)
    summary = convert_source_files_to_mcap(source_files, manifest, output)
    summary_path = output.with_suffix(output.suffix + ".manifest.json")
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and convert the pinned ALOHA replay fixture",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = prepare_fixture(args.manifest, args.cache_dir, args.output)
    print(args.output.resolve())
    print(f"samples={summary.sample_count} duration_s={summary.duration_s:.2f}")
    print(f"gripper_tolerance={summary.gripper_tolerance:g}")


if __name__ == "__main__":
    main()
