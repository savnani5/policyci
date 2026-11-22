from __future__ import annotations

from policyci.examples.aloha_replay.action_processor import GripperDelay
from policyci.examples.aloha_replay.ros_node import transform_action_message


def test_transform_action_message_uses_persistent_delay_state() -> None:
    delay = GripperDelay(delay_messages=1)
    first = [float(index) for index in range(14)]
    second = [float(index) + 10.0 for index in range(14)]

    assert transform_action_message(first, delay) == tuple(first)
    output = transform_action_message(second, delay)
    assert output[6] == first[6]
    assert output[13] == first[13]
    assert output[:6] == tuple(second[:6])
    assert output[7:13] == tuple(second[7:13])
