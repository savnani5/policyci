"""ROS 2 action component used by the ALOHA replay demo."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from policyci.examples.aloha_replay.action_processor import GripperDelay


def transform_action_message(
    values: Sequence[float],
    delay: GripperDelay,
) -> tuple[float, ...]:
    return delay.transform(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gripper-delay-messages", type=int, required=True)
    args, ros_args = parser.parse_known_args()

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray

    class AlohaActionProcessor(Node):
        def __init__(self, delay_messages: int) -> None:
            super().__init__("robolens_aloha_action_processor")
            self.delay = GripperDelay(delay_messages)
            self.publisher = self.create_publisher(
                Float64MultiArray,
                "/policy/action",
                10,
            )
            self.subscription = self.create_subscription(
                Float64MultiArray,
                "/episode/reference_action",
                self.on_action,
                10,
            )

        def on_action(self, message: Float64MultiArray) -> None:
            output = Float64MultiArray()
            output.data = list(transform_action_message(message.data, self.delay))
            self.publisher.publish(output)

    rclpy.init(args=ros_args)
    node = AlohaActionProcessor(args.gripper_delay_messages)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
