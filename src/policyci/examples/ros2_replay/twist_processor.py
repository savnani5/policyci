"""ROS 2 node under test: transform TwistStamped velocity with gain and bias."""

from __future__ import annotations

import argparse


def demo_command_profile() -> tuple[float, ...]:
    """Deterministic stop, drive, stop command sequence for the live demo."""
    return (0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0)


def transform_linear_x(
    value: float,
    *,
    gain: float,
    bias: float,
    message_index: int = 0,
    bias_after_messages: int | None = None,
) -> float:
    active_bias = (
        bias
        if bias_after_messages is None or message_index >= bias_after_messages
        else 0.0
    )
    return value * gain + active_bias


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--bias", type=float, required=True)
    parser.add_argument("--bias-after-messages", type=int)
    args, ros_args = parser.parse_known_args()

    import rclpy
    from geometry_msgs.msg import TwistStamped
    from rclpy.node import Node

    class TwistProcessor(Node):
        def __init__(
            self,
            gain: float,
            bias: float,
            bias_after_messages: int | None,
        ) -> None:
            super().__init__("robolens_twist_processor")
            self.gain = gain
            self.bias = bias
            self.bias_after_messages = bias_after_messages
            self.message_index = 0
            self.publisher = self.create_publisher(TwistStamped, "/user/cmd_vel", 10)
            self.subscription = self.create_subscription(
                TwistStamped,
                "/vehicle/cmd_vel",
                self.on_input,
                10,
            )

        def on_input(self, message: TwistStamped) -> None:
            output = TwistStamped()
            output.header = message.header
            output.twist.linear.x = transform_linear_x(
                message.twist.linear.x,
                gain=self.gain,
                bias=self.bias,
                message_index=self.message_index,
                bias_after_messages=self.bias_after_messages,
            )
            self.message_index += 1
            output.twist.linear.y = message.twist.linear.y
            output.twist.linear.z = message.twist.linear.z
            output.twist.angular.x = message.twist.angular.x
            output.twist.angular.y = message.twist.angular.y
            output.twist.angular.z = message.twist.angular.z
            self.publisher.publish(output)

    rclpy.init(args=ros_args)
    node = TwistProcessor(args.gain, args.bias, args.bias_after_messages)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
