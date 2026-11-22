"""Pure scoring and command construction for the bounded Twist replay example."""

from __future__ import annotations

from pathlib import Path

import pytest

from policyci.examples.ros2_replay.adapter import (
    TwistReplayAdapter,
    linear_x,
    score_twist_samples,
)
from policyci.examples.ros2_replay import twist_processor
from policyci.examples.ros2_replay.twist_processor import transform_linear_x
from policyci.checkspec import CaseSpec
from policyci.schemas import SubjectRef


def test_known_bad_gain_fails_tolerance():
    analysis = score_twist_samples(
        input_x=[1.0, 0.5],
        output_x=[1.4, 0.7],
        tolerance=0.05,
    )
    assert analysis.metrics["within_tolerance"] is False
    assert analysis.metrics["mean_abs_error"] == pytest.approx(0.3)
    assert analysis.metrics["message_count"] == 2
    assert analysis.trace is not None
    assert analysis.trace.series["expected"].values == [1.0, 0.5]
    assert analysis.trace.series["observed"].values == [1.4, 0.7]
    assert analysis.trace.series["absolute_error"].values == pytest.approx([0.4, 0.2])
    assert {series.unit for series in analysis.trace.series.values()} == {"m/s"}


def test_fixed_gain_satisfies_tolerance():
    analysis = score_twist_samples(
        input_x=[1.0, 0.5],
        output_x=[1.0, 0.5],
        tolerance=0.05,
    )
    assert analysis.metrics["within_tolerance"] is True
    assert analysis.metrics["max_abs_error"] == 0.0
    assert analysis.trace is not None


def test_zero_command_bias_creates_detectable_failure():
    assert transform_linear_x(0.0, gain=1.0, bias=0.4) == 0.4
    assert transform_linear_x(0.0, gain=1.0, bias=0.0) == 0.0


def test_candidate_regression_activates_only_at_final_stop_segment():
    commands = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]

    observed = [
        transform_linear_x(
            command,
            gain=1.0,
            bias=0.4,
            message_index=index,
            bias_after_messages=7,
        )
        for index, command in enumerate(commands)
    ]

    assert observed == [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4]


def test_demo_profile_is_stop_drive_stop():
    demo_command_profile = getattr(twist_processor, "demo_command_profile", None)

    assert callable(demo_command_profile)
    assert demo_command_profile() == (
        0.0,
        0.0,
        0.0,
        0.5,
        0.5,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
    )


def test_linear_x_supports_twist_and_twist_stamped_messages():
    class Vector:
        x = 1.25

    class Twist:
        linear = Vector()

    class TwistStamped:
        twist = Twist()

    assert linear_x(Twist()) == 1.25
    assert linear_x(TwistStamped()) == 1.25


@pytest.mark.parametrize(
    ("input_x", "output_x", "message"),
    [
        ([], [], "empty"),
        ([1.0], [], "empty"),
        ([1.0, 2.0], [1.0], "count"),
    ],
)
def test_scoring_rejects_unsound_samples(input_x, output_x, message):
    with pytest.raises(ValueError, match=message):
        score_twist_samples(input_x=input_x, output_x=output_x, tolerance=0.05)


def test_adapter_builds_subject_and_readiness_commands(tmp_path: Path):
    adapter = TwistReplayAdapter()
    subject = SubjectRef(
        id="stop-regression",
        config={"gain": 1.0, "bias": 0.4, "bias_after_messages": 7},
    )
    case = CaseSpec(
        id="twist",
        bag=tmp_path / "input.mcap",
        required_topics=["/vehicle/cmd_vel"],
        capture_topics=["/user/cmd_vel"],
        judge={"tolerance": 0.05},
    )
    subject_command = adapter.subject_command(subject, case)
    assert subject_command[0] == "python3"
    assert "--gain" in subject_command
    assert subject_command[subject_command.index("--gain") + 1] == "1.0"
    assert subject_command[subject_command.index("--bias") + 1] == "0.4"
    assert subject_command[subject_command.index("--bias-after-messages") + 1] == "7"
    assert adapter.readiness_command(subject, case) == [
        "ros2",
        "topic",
        "info",
        "/user/cmd_vel",
    ]
    assert adapter.applied_subject_fields(subject, case) == {
        "config.gain",
        "config.bias",
        "config.bias_after_messages",
    }
