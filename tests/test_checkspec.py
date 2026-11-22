"""Strict file contracts for RoboLens checks."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from policyci.checkspec import load_check_suite, load_subject
from policyci.schemas import (
    ArtifactRef,
    CaseExecution,
    ExecutionValidity,
    SubjectRef,
)


def _write(path: Path, value: str) -> Path:
    path.write_text(value)
    return path


def test_loads_libero_suite_and_subject(tmp_path: Path):
    suite_path = _write(
        tmp_path / "suite.yaml",
        """
version: 1
name: manipulation-release
runner:
  type: libero
  suite: libero_object
execution:
  episodes: 2
  seeds: [1000, 1001]
gate:
  metric: success_rate
  mode: max_drop
  max_drop_pp: 10
  min_valid_episodes: 2
cases:
  - id: soup_to_basket
    task: 0
""",
    )
    subject_path = _write(
        tmp_path / "subject.yaml",
        """
id: smolvla-pr-184
checkpoint: HuggingFaceVLA/smolvla_libero
code_sha: 6a8c921
config:
  n_action_steps: 4
""",
    )

    suite = load_check_suite(suite_path)
    subject = load_subject(subject_path)

    assert suite.runner.type == "libero"
    assert suite.runner.suite == "libero_object"
    assert suite.cases[0].resolved_seeds(suite.execution) == [1000, 1001]
    assert subject.config == {"n_action_steps": 4}


def test_generates_consecutive_seeds_from_episode_count(tmp_path: Path):
    path = _write(
        tmp_path / "suite.yaml",
        """
version: 1
name: smoke
runner: {type: libero, suite: libero_object}
execution: {episodes: 3, seed: 41}
gate: {metric: success_rate, mode: max_drop, min_valid_episodes: 2}
cases:
  - {id: soup, task: 0}
""",
    )
    suite = load_check_suite(path)
    assert suite.cases[0].resolved_seeds(suite.execution) == [41, 42, 43]


def test_rejects_episode_and_seed_count_mismatch(tmp_path: Path):
    path = _write(
        tmp_path / "suite.yaml",
        """
version: 1
name: smoke
runner: {type: libero, suite: libero_object}
execution: {episodes: 3, seeds: [1, 2]}
gate: {metric: success_rate, mode: max_drop, min_valid_episodes: 2}
cases:
  - {id: soup, task: 0}
""",
    )
    with pytest.raises(ValueError, match="episodes.*seeds|seeds.*episodes"):
        load_check_suite(path)


def test_rejects_duplicate_explicit_seeds(tmp_path: Path):
    path = _write(
        tmp_path / "suite.yaml",
        """
version: 1
name: smoke
runner: {type: libero, suite: libero_object}
execution: {episodes: 2, seeds: [1000, 1000]}
gate: {metric: success_rate, mode: max_drop, min_valid_episodes: 2}
cases:
  - {id: soup, task: 0}
""",
    )

    with pytest.raises(ValueError, match="duplicate seeds"):
        load_check_suite(path)


def test_rejects_unsupported_libero_perturbation(tmp_path: Path):
    path = _write(
        tmp_path / "suite.yaml",
        """
version: 1
name: smoke
runner: {type: libero, suite: libero_object}
execution: {episodes: 1, seeds: [1]}
gate: {metric: success_rate, mode: max_drop, min_valid_episodes: 1}
cases:
  - id: soup
    task: 0
    perturbation: {pose_cm: 4}
""",
    )
    with pytest.raises(ValueError, match="perturbation|extra"):
        load_check_suite(path)


def test_rejects_duplicate_case_ids(tmp_path: Path):
    path = _write(
        tmp_path / "suite.yaml",
        """
version: 1
name: smoke
runner: {type: libero, suite: libero_object}
execution: {episodes: 1, seeds: [1]}
gate: {metric: success_rate, mode: max_drop, min_valid_episodes: 1}
cases:
  - {id: soup, task: 0}
  - {id: soup, task: 1}
""",
    )
    with pytest.raises(ValueError, match="duplicate case id"):
        load_check_suite(path)


@pytest.mark.parametrize("case_id", ["../escape", "nested/case", ".", ""])
def test_rejects_case_ids_that_are_not_safe_path_components(
    tmp_path: Path,
    case_id: str,
):
    path = _write(
        tmp_path / "suite.yaml",
        f"""
version: 1
name: smoke
runner: {{type: libero, suite: libero_object}}
gate: {{metric: success_rate, mode: max_drop}}
cases:
  - id: {case_id!r}
    task: 0
""",
    )
    with pytest.raises(ValueError, match="id|string_pattern|at least 1 character"):
        load_check_suite(path)


def test_replay_suite_resolves_relative_bag_path(tmp_path: Path):
    case_dir = tmp_path / "cases" / "crossing"
    case_dir.mkdir(parents=True)
    bag = case_dir / "input.mcap"
    bag.write_bytes(b"fixture")
    path = _write(
        tmp_path / "suite.yaml",
        """
version: 1
name: field-regressions
runner:
  type: ros2_replay
  adapter: examples.adapter:ReplayAdapter
gate:
  metric: collision_free
  mode: require_value
  expect: true
  min_valid_repeats: 2
cases:
  - id: crossing
    bag: cases/crossing/input.mcap
    repeats: 2
    required_topics: [/scan]
    capture_topics: [/cmd_vel]
""",
    )
    suite = load_check_suite(path)
    assert suite.cases[0].bag == bag.resolve()


def test_replay_case_keeps_immutable_judge_configuration(tmp_path: Path):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    path = _write(
        tmp_path / "suite.yaml",
        """
version: 1
name: replay
runner: {type: ros2_replay, adapter: example:Adapter}
gate: {metric: within_tolerance, mode: require_value, expect: true}
cases:
  - id: stop
    bag: input.mcap
    required_topics: [/input]
    capture_topics: [/output]
    judge: {tolerance: 0.05}
""",
    )

    suite = load_check_suite(path)

    assert suite.cases[0].judge == {"tolerance": 0.05}


def test_replay_playback_topics_must_be_a_required_topic_subset(tmp_path: Path) -> None:
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    path = _write(
        tmp_path / "suite.yaml",
        """
version: 1
name: replay
runner: {type: ros2_replay, adapter: example:Adapter}
gate: {metric: within_tolerance, mode: require_value, expect: true}
cases:
  - id: component
    bag: input.mcap
    required_topics: [/camera, /action]
    playback_topics: [/other]
    capture_topics: [/output]
""",
    )

    with pytest.raises(ValueError, match="playback topics.*required"):
        load_check_suite(path)


def test_loads_sensor_rich_aloha_replay_suite() -> None:
    suite = load_check_suite("suites/aloha-coffee-replay.yaml")

    case = suite.cases[0]
    assert suite.runner.adapter == (
        "policyci.examples.aloha_replay.adapter:AlohaReplayAdapter"
    )
    assert suite.gate.metric == "within_tolerance"
    assert suite.gate.expect is True
    assert case.repeats == 1
    assert case.judge == {
        "sample_rate_hz": 50.0,
        "gripper_tolerance": 0.051004,
        "arm_tolerance": 1e-9,
    }
    assert case.required_topics == [
        "/episode/cam_high/image/compressed",
        "/episode/cam_left_wrist/image/compressed",
        "/episode/cam_right_wrist/image/compressed",
        "/episode/joint_states",
        "/episode/reference_action",
        "/episode/task",
    ]
    assert case.capture_topics == ["/policy/action"]


def test_loads_mujoco_grasp_suite(tmp_path: Path):
    suite = _write(
        tmp_path / "suite.yaml",
        """
version: 1
name: mujoco-grasp-regression
runner: {type: mujoco}
execution: {episodes: 1, seeds: [1000]}
gate: {metric: lift_success, mode: require_value, expect: true, min_valid_repeats: 1}
cases:
  - id: grasp_offset
    judge: {lift_threshold_m: 0.08, hold_duration_s: 0.5}
""",
    )

    loaded = load_check_suite(suite)

    assert loaded.runner.type == "mujoco"
    assert loaded.cases[0].resolved_seeds(loaded.execution) == [1000]


def test_mujoco_case_accepts_and_resolves_object_position_offset(tmp_path: Path):
    suite = _write(
        tmp_path / "suite.yaml",
        """
version: 1
name: mujoco-object-pose
runner: {type: mujoco}
gate: {metric: placement_success, mode: require_value, expect: true}
cases:
  - id: shifted
    object_position_offset_m: [0.05, 0.0, 0.0]
    judge: {placement_hold_duration_s: 0.5}
  - id: nominal
    judge: {placement_hold_duration_s: 0.5}
""",
    )

    loaded = load_check_suite(suite)

    assert loaded.cases[0].resolved_object_position_offset_m() == (0.05, 0.0, 0.0)
    assert loaded.cases[1].resolved_object_position_offset_m() == (0.0, 0.0, 0.0)


def test_mujoco_case_accepts_and_resolves_object_observation_bias(tmp_path: Path):
    suite = _write(
        tmp_path / "suite.yaml",
        """
version: 1
name: mujoco-perception-bias
runner: {type: mujoco}
gate: {metric: placement_success, mode: require_value, expect: true}
cases:
  - id: biased
    object_observation_bias_m: [0.05, 0.0, 0.0]
    judge: {placement_hold_duration_s: 0.5}
  - id: nominal
    judge: {placement_hold_duration_s: 0.5}
""",
    )

    loaded = load_check_suite(suite)

    assert loaded.cases[0].resolved_object_observation_bias_m() == (0.05, 0.0, 0.0)
    assert loaded.cases[1].resolved_object_observation_bias_m() == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "offset",
    ["[0.0, 0.0]", "[0.11, 0.0, 0.0]", "[.nan, 0.0, 0.0]", "true"],
)
def test_mujoco_case_rejects_invalid_object_position_offset(
    tmp_path: Path,
    offset: str,
):
    suite = _write(
        tmp_path / "suite.yaml",
        f"""
version: 1
name: bad-mujoco-object-pose
runner: {{type: mujoco}}
gate: {{metric: placement_success, mode: require_value, expect: true}}
cases:
  - id: shifted
    object_position_offset_m: {offset}
    judge: {{placement_hold_duration_s: 0.5}}
""",
    )

    with pytest.raises(ValueError, match="object_position_offset_m"):
        load_check_suite(suite)


@pytest.mark.parametrize(
    "bias",
    ["[0.0, 0.0]", "[0.11, 0.0, 0.0]", "[.nan, 0.0, 0.0]", "true"],
)
def test_mujoco_case_rejects_invalid_object_observation_bias(
    tmp_path: Path,
    bias: str,
):
    suite = _write(
        tmp_path / "suite.yaml",
        f"""
version: 1
name: bad-mujoco-perception-bias
runner: {{type: mujoco}}
gate: {{metric: placement_success, mode: require_value, expect: true}}
cases:
  - id: biased
    object_observation_bias_m: {bias}
    judge: {{placement_hold_duration_s: 0.5}}
""",
    )

    with pytest.raises(ValueError, match="object_observation_bias_m"):
        load_check_suite(suite)


def test_replay_case_rejects_mujoco_object_position_offset(tmp_path: Path):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    suite = _write(
        tmp_path / "suite.yaml",
        """
version: 1
name: replay
runner: {type: ros2_replay, adapter: example:Adapter}
gate: {metric: within_tolerance, mode: require_value, expect: true}
cases:
  - id: stop
    bag: input.mcap
    required_topics: [/input]
    capture_topics: [/output]
    object_position_offset_m: [0.05, 0.0, 0.0]
    judge: {tolerance: 0.05}
""",
    )

    with pytest.raises(ValueError, match="object_position_offset_m"):
        load_check_suite(suite)


def test_replay_case_rejects_mujoco_object_observation_bias(tmp_path: Path):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    suite = _write(
        tmp_path / "suite.yaml",
        """
version: 1
name: replay
runner: {type: ros2_replay, adapter: example:Adapter}
gate: {metric: within_tolerance, mode: require_value, expect: true}
cases:
  - id: stop
    bag: input.mcap
    required_topics: [/input]
    capture_topics: [/output]
    object_observation_bias_m: [0.05, 0.0, 0.0]
    judge: {tolerance: 0.05}
""",
    )

    with pytest.raises(ValueError, match="object_observation_bias_m"):
        load_check_suite(suite)


def test_checked_in_mujoco_pick_place_demo_inputs_are_exact():
    repo = Path(__file__).resolve().parents[1]

    suite = load_check_suite(repo / "suites/mujoco-grasp-regression.yaml")
    baseline = load_subject(repo / "subjects/grasp-offset-bad.yaml")
    candidate = load_subject(repo / "subjects/grasp-offset-fixed.yaml")

    assert suite.runner.type == "mujoco"
    assert suite.execution.seeds == [1000]
    assert suite.gate.metric == "placement_success"
    assert suite.cases[0].judge == {"placement_hold_duration_s": 0.5}
    assert baseline.config["action_frame_offset_m"] == [0.06, 0.0, 0.0]
    assert candidate.config["action_frame_offset_m"] == [0.0, 0.0, 0.0]
    assert baseline.code_sha == candidate.code_sha == "mujoco-pick-place-demo-v2"


@pytest.mark.parametrize(
    "judge",
    [
        "{}",
        "{lift_threshold_m: 0.08}",
        "{lift_threshold_m: -0.1, hold_duration_s: 0.5}",
        "{lift_threshold_m: .nan, hold_duration_s: 0.5}",
        "{lift_threshold_m: 0.08, hold_duration_s: .inf}",
        "{lift_threshold_m: 0.08, hold_duration_s: 0.5, tolerance: 1}",
    ],
)
def test_mujoco_case_requires_exact_positive_judge_fields(
    tmp_path: Path,
    judge: str,
):
    suite = _write(
        tmp_path / "suite.yaml",
        f"""
version: 1
name: bad-mujoco
runner: {{type: mujoco}}
gate: {{metric: lift_success, mode: require_value, expect: true}}
cases:
  - id: grasp_offset
    judge: {judge}
""",
    )

    with pytest.raises(ValueError, match="lift_threshold_m|hold_duration_s|judge"):
        load_check_suite(suite)


@pytest.mark.parametrize(
    ("metric", "judge"),
    [
        ("lift_success", "{placement_hold_duration_s: 0.5}"),
        (
            "placement_success",
            "{lift_threshold_m: 0.08, hold_duration_s: 0.5}",
        ),
    ],
)
def test_mujoco_gate_metric_must_match_judge_kind(
    tmp_path: Path,
    metric: str,
    judge: str,
):
    suite = _write(
        tmp_path / "suite.yaml",
        f"""
version: 1
name: mismatched-mujoco
runner: {{type: mujoco}}
gate: {{metric: {metric}, mode: require_value, expect: true}}
cases:
  - id: mismatch
    judge: {judge}
""",
    )

    with pytest.raises(ValueError, match="metric.*judge|judge.*metric"):
        load_check_suite(suite)


def test_case_execution_round_trip_has_validity_and_relative_artifacts():
    execution = CaseExecution(
        subject=SubjectRef(id="baseline", checkpoint="org/model"),
        case_id="soup",
        runner="libero",
        validity=ExecutionValidity.VALID,
        metrics={"success_rate": 100.0},
        valid_units=1,
        unit_ids=["1000"],
        artifacts=[ArtifactRef(kind="video", path="artifacts/episode-1000.mp4")],
    )
    loaded = CaseExecution.model_validate_json(execution.model_dump_json())
    assert loaded.validity is ExecutionValidity.VALID
    assert loaded.artifacts[0].path == "artifacts/episode-1000.mp4"


@pytest.mark.parametrize("path", ["/tmp/video.mp4", "../video.mp4", "safe/../../video.mp4"])
def test_artifact_paths_must_be_portable(path: str):
    with pytest.raises(ValidationError, match="relative|traversal"):
        ArtifactRef(kind="video", path=path)
