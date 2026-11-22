from __future__ import annotations

import pytest

from policyci.examples.aloha_replay.action_processor import (
    ALOHA_JOINT_NAMES,
    GripperDelay,
)


def _action(offset: float) -> tuple[float, ...]:
    return tuple(float(index) + offset for index in range(len(ALOHA_JOINT_NAMES)))


def test_gripper_delay_preserves_arms_and_delays_only_grippers() -> None:
    delay = GripperDelay(delay_messages=2)
    first = _action(0.0)
    second = _action(10.0)
    third = _action(20.0)

    assert delay.transform(first) == first

    second_output = delay.transform(second)
    assert second_output[6] == first[6]
    assert second_output[13] == first[13]
    assert second_output[:6] == second[:6]
    assert second_output[7:13] == second[7:13]

    third_output = delay.transform(third)
    assert third_output[6] == first[6]
    assert third_output[13] == first[13]
    assert third_output[:6] == third[:6]
    assert third_output[7:13] == third[7:13]


def test_zero_delay_returns_each_action_unchanged() -> None:
    delay = GripperDelay(delay_messages=0)

    assert delay.transform(_action(0.0)) == _action(0.0)
    assert delay.transform(_action(10.0)) == _action(10.0)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_gripper_delay_rejects_invalid_delay(value: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        GripperDelay(delay_messages=value)  # type: ignore[arg-type]


def test_gripper_delay_rejects_non_aloha_action() -> None:
    delay = GripperDelay(delay_messages=1)

    with pytest.raises(ValueError, match="exactly 14"):
        delay.transform([0.0] * 13)
