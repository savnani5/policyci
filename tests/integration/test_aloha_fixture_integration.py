from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from policyci.examples.aloha_replay.fixture import (
    AlohaSourceManifest,
    convert_source_files_to_mcap,
    selected_source_paths,
)

pytestmark = pytest.mark.integration


def _write_video(path: Path, colors: list[tuple[int, int, int]]) -> None:
    import av
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=50)
        stream.width = 32
        stream.height = 24
        stream.pix_fmt = "yuv420p"
        for color in colors:
            frame = av.VideoFrame.from_image(Image.new("RGB", (32, 24), color))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.skipif(
    importlib.util.find_spec("rosbag2_py") is None,
    reason="requires ROS 2",
)
def test_convert_two_frame_episode_to_six_topic_ros2_mcap(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import rosbag2_py

    manifest = AlohaSourceManifest(
        repo_id="lerobot/aloha_static_coffee",
        revision="37fdb4dc78dc8a3b3f1fc44f7c7c60a207d18937",
        episode_index=0,
        sample_rate_hz=50,
        cameras=("cam_high", "cam_left_wrist", "cam_right_wrist"),
        task_label="Make coffee",
        license="MIT",
        source_url="https://huggingface.co/datasets/lerobot/aloha_static_coffee",
    )
    source_files = {path: tmp_path / "source" / path for path in selected_source_paths(manifest)}
    for path in source_files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    source_files["README.md"].write_text("MIT")
    source_files["meta/info.json"].write_text("{}")
    pq.write_table(
        pa.table(
            {
                "timestamp": [0.0, 0.02],
                "observation.state": [[0.0] * 14, [1.0] * 14],
                "observation.effort": [[2.0] * 14, [3.0] * 14],
                "action": [[0.0] * 14, [0.5] * 14],
            }
        ),
        source_files["data/chunk-000/episode_000000.parquet"],
    )
    for camera in manifest.cameras:
        _write_video(
            source_files[
                f"videos/chunk-000/observation.images.{camera}/episode_000000.mp4"
            ],
            [(255, 0, 0), (0, 255, 0)],
        )

    output = tmp_path / "episode.mcap"
    summary = convert_source_files_to_mcap(source_files, manifest, output)

    assert output.is_file() and output.stat().st_size > 0
    assert summary.sample_count == 2
    assert summary.topic_counts == {
        "/episode/cam_high/image/compressed": 2,
        "/episode/cam_left_wrist/image/compressed": 2,
        "/episode/cam_right_wrist/image/compressed": 2,
        "/episode/joint_states": 2,
        "/episode/reference_action": 2,
        "/episode/task": 1,
    }

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(output), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    topic_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    assert topic_types["/episode/joint_states"] == "sensor_msgs/msg/JointState"
    assert topic_types["/episode/reference_action"] == "std_msgs/msg/Float64MultiArray"
    assert topic_types["/episode/task"] == "std_msgs/msg/String"
