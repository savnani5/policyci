"""Generate the deterministic stop-drive-stop MCAP used by the ROS 2 demo."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from policyci.examples.ros2_replay.twist_processor import demo_command_profile


def generate_fixture(output: Path) -> Path:
    try:
        import rosbag2_py
        from geometry_msgs.msg import TwistStamped
        from rclpy.serialization import serialize_message
    except ImportError as exc:
        raise RuntimeError("fixture generation requires a ROS 2 environment") from exc

    destination = output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="robolens-stop-drive-stop-") as temp_dir:
        bag_uri = Path(temp_dir) / "bag"
        writer = rosbag2_py.SequentialWriter()
        writer.open(
            rosbag2_py.StorageOptions(uri=str(bag_uri), storage_id="mcap"),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr",
                output_serialization_format="cdr",
            ),
        )
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                id=0,
                name="/vehicle/cmd_vel",
                type="geometry_msgs/msg/TwistStamped",
                serialization_format="cdr",
            )
        )
        start_ns = 1_000_000_000
        step_ns = 250_000_000
        for index, command in enumerate(demo_command_profile()):
            timestamp_ns = start_ns + index * step_ns
            message = TwistStamped()
            message.header.stamp.sec = timestamp_ns // 1_000_000_000
            message.header.stamp.nanosec = timestamp_ns % 1_000_000_000
            message.twist.linear.x = command
            writer.write(
                "/vehicle/cmd_vel",
                serialize_message(message),
                timestamp_ns,
            )
        del writer

        generated = list(bag_uri.rglob("*.mcap"))
        if len(generated) != 1:
            raise RuntimeError(
                f"expected one generated MCAP, found {len(generated)} under {bag_uri}"
            )
        shutil.copyfile(generated[0], destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generated = generate_fixture(args.output)
    print(generated)


if __name__ == "__main__":
    main()
