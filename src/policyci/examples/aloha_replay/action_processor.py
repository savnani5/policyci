"""Pure action transformation used by the ALOHA replay subject."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

ALOHA_JOINT_NAMES = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
)
GRIPPER_INDICES = (6, 13)


class GripperDelay:
    """Delay only the bimanual gripper channels by a fixed message count."""

    def __init__(self, delay_messages: int) -> None:
        if (
            isinstance(delay_messages, bool)
            or not isinstance(delay_messages, int)
            or delay_messages < 0
        ):
            raise ValueError("delay_messages must be a non-negative integer")
        self.delay_messages = delay_messages
        self._history: deque[tuple[float, float]] = deque()

    def transform(self, action: Sequence[float]) -> tuple[float, ...]:
        if len(action) != len(ALOHA_JOINT_NAMES):
            raise ValueError("ALOHA action must contain exactly 14 values")
        output = [float(value) for value in action]
        if any(not math.isfinite(value) for value in output):
            raise ValueError("ALOHA action values must be finite")

        current_grippers = tuple(output[index] for index in GRIPPER_INDICES)
        self._history.append(current_grippers)
        delayed_grippers = (
            self._history[0]
            if len(self._history) <= self.delay_messages
            else self._history.popleft()
        )
        for index, value in zip(GRIPPER_INDICES, delayed_grippers, strict=True):
            output[index] = value
        return tuple(output)
