from __future__ import annotations

from pathlib import Path

import pytest

from policyci.checkspec import CaseSpec, load_check_suite, load_subject
from policyci.examples.aloha_replay.adapter import (
    AlohaReplayAdapter,
    REFERENCE_ACTION_TOPIC,
    score_aloha_samples,
)
from policyci.schemas import SubjectRef


def _case(*, sample_rate_hz: float = 50.0) -> CaseSpec:
    return CaseSpec(
        id="aloha_coffee_replay",
        bag=Path("episode.mcap"),
        required_topics=["/episode/reference_action"],
        capture_topics=["/policy/action"],
        judge={
            "sample_rate_hz": sample_rate_hz,
            "gripper_tolerance": 0.05,
            "arm_tolerance": 1e-9,
        },
    )


def _action(*, left_gripper: float = 0.0, right_gripper: float = 0.0) -> list[float]:
    values = [0.0] * 14
    values[6] = left_gripper
    values[13] = right_gripper
    return values


def test_score_aloha_samples_emits_multichannel_trace_and_first_failure() -> None:
    reference = [_action(), _action(), _action()]
    observed = [_action(), _action(left_gripper=0.25), _action()]

    analysis = score_aloha_samples(
        reference=reference,
        observed=observed,
        time_s=[0.0, 0.02, 0.04],
        gripper_tolerance=0.05,
        arm_tolerance=1e-9,
    )

    assert analysis.metrics == {
        "within_tolerance": False,
        "max_gripper_error": 0.25,
        "max_arm_error": 0.0,
        "message_count": 3,
        "first_violation_index": 1,
        "first_violation_time_s": 0.02,
    }
    assert analysis.trace is not None
    assert analysis.trace.trace_type == "aloha_action"
    assert analysis.trace.series["left_gripper_error"].values == [0.0, 0.25, 0.0]
    assert analysis.trace.series["right_gripper_error"].values == [0.0, 0.0, 0.0]
    assert analysis.trace.series["max_gripper_error"].values == [0.0, 0.25, 0.0]


def test_score_aloha_samples_uses_negative_sentinel_when_all_samples_pass() -> None:
    reference = [_action(), _action(left_gripper=0.1)]

    analysis = score_aloha_samples(
        reference=reference,
        observed=[row.copy() for row in reference],
        time_s=[0.0, 0.02],
        gripper_tolerance=0.05,
        arm_tolerance=1e-9,
    )

    assert analysis.metrics["within_tolerance"] is True
    assert analysis.metrics["first_violation_index"] == -1
    assert analysis.metrics["first_violation_time_s"] == -1.0


def test_score_aloha_samples_detects_arm_regression() -> None:
    reference = [_action()]
    observed = [_action()]
    observed[0][2] = 0.01

    analysis = score_aloha_samples(
        reference=reference,
        observed=observed,
        time_s=[0.0],
        gripper_tolerance=0.05,
        arm_tolerance=1e-9,
    )

    assert analysis.metrics["within_tolerance"] is False
    assert analysis.metrics["max_arm_error"] == 0.01


@pytest.mark.parametrize(
    ("reference", "observed", "time_s", "message"),
    [
        ([_action()], [], [0.0], "message count differs"),
        ([_action()], [_action()], [], "timestamp count differs"),
        ([[0.0] * 13], [[0.0] * 13], [0.0], "exactly 14"),
        ([_action(), _action()], [_action(), _action()], [0.0, 0.0], "strictly increasing"),
    ],
)
def test_score_aloha_samples_rejects_unsound_inputs(
    reference: list[list[float]],
    observed: list[list[float]],
    time_s: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        score_aloha_samples(
            reference=reference,
            observed=observed,
            time_s=time_s,
            gripper_tolerance=0.05,
            arm_tolerance=1e-9,
        )


def test_adapter_applies_only_declared_delay_config() -> None:
    subject = SubjectRef(
        id="aloha-gripper-delay-300ms",
        config={"gripper_delay_messages": 15},
    )
    adapter = AlohaReplayAdapter()

    assert adapter.applied_subject_fields(subject, _case()) == {
        "config.gripper_delay_messages"
    }
    assert adapter.subject_command(subject, _case())[-2:] == [
        "--gripper-delay-messages",
        "15",
    ]


def test_adapter_readiness_waits_for_the_subject_input_subscription() -> None:
    subject = SubjectRef(id="baseline", config={"gripper_delay_messages": 0})

    assert AlohaReplayAdapter().readiness_command(subject, _case()) == [
        "ros2",
        "topic",
        "info",
        REFERENCE_ACTION_TOPIC,
    ]


def test_adapter_requires_50hz_fixture() -> None:
    subject = SubjectRef(id="candidate", config={"gripper_delay_messages": 15})

    with pytest.raises(ValueError, match="50 Hz"):
        AlohaReplayAdapter().subject_command(subject, _case(sample_rate_hz=30.0))


@pytest.mark.parametrize("delay", [-1, True, 1.5])
def test_adapter_rejects_invalid_delay(delay: object) -> None:
    subject = SubjectRef(id="candidate", config={"gripper_delay_messages": delay})

    with pytest.raises(ValueError, match="non-negative integer"):
        AlohaReplayAdapter().subject_command(subject, _case())


def test_checked_in_aloha_subjects_define_zero_and_15_message_delays() -> None:
    baseline = load_subject("subjects/aloha-action-baseline.yaml")
    candidate = load_subject("subjects/aloha-gripper-delay.yaml")

    assert baseline.config == {"gripper_delay_messages": 0}
    assert candidate.config == {"gripper_delay_messages": 15}


def test_checked_in_aloha_suite_uses_resolved_episode_tolerance() -> None:
    suite = load_check_suite("suites/aloha-coffee-replay.yaml")

    assert suite.cases[0].judge["gripper_tolerance"] == 0.051004
    assert suite.cases[0].playback_topics == ["/episode/reference_action"]
    assert suite.execution.playback_discovery_delay_s == 5.0
