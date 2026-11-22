"""Tests for schemas + store round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from policyci.schemas import (
    EpisodeRecord,
    GateVerdict,
    ReplayAnalysis,
    ReplayTrace,
    RunManifest,
    SignalSeries,
    SimulationSample,
    SimulationTrace,
    TaskResult,
    TaskStatus,
)
from policyci.runner.client import _attach_disk_videos
from policyci.store import list_runs, load_run, save_run
from policyci.runner.modal_app import _parse_eval_info


FIXTURES = Path(__file__).parent / "fixtures"


def _sim_sample(time_s: float, *, z: float = 0.02) -> SimulationSample:
    return SimulationSample(
        time_s=time_s,
        object_position_m=(0.0, 0.0, z),
        object_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        eef_position_m=(0.0, 0.0, 0.10),
        eef_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        gripper_width_m=0.04,
        left_finger_contact=False,
        right_finger_contact=False,
    )


def test_simulation_trace_requires_monotonic_finite_state():
    trace = SimulationTrace(
        sample_rate_hz=20.0,
        seed=1000,
        initial_state_sha256="a" * 64,
        model_sha256="b" * 64,
        samples=[_sim_sample(0.0), _sim_sample(0.05)],
    )

    assert trace.sample_count == 2
    assert trace.duration_s == pytest.approx(0.05)


@pytest.mark.parametrize(
    "samples",
    [
        [],
        [_sim_sample(0.0), _sim_sample(0.0)],
        [_sim_sample(0.1), _sim_sample(0.05)],
    ],
)
def test_simulation_trace_rejects_empty_or_nonmonotonic_samples(samples):
    with pytest.raises(ValueError):
        SimulationTrace(
            sample_rate_hz=20.0,
            seed=1000,
            initial_state_sha256="a" * 64,
            model_sha256="b" * 64,
            samples=samples,
        )


def test_simulation_sample_rejects_nonfinite_vectors():
    with pytest.raises(ValueError):
        SimulationSample.model_validate(
            {
                **_sim_sample(0.0).model_dump(),
                "object_position_m": [0.0, 0.0, float("nan")],
            }
        )


def test_check_states_are_part_of_the_serialized_contract():
    assert TaskStatus.NOT_EVALUATED.value == "not_evaluated"
    assert GateVerdict.BLOCK.value == "block"
    assert GateVerdict.NO_VERDICT.value == "no_verdict"


def test_replay_trace_requires_aligned_finite_series():
    trace = ReplayTrace(
        series={
            "expected": SignalSeries(values=[0.0, 0.0], unit="m/s"),
            "observed": SignalSeries(values=[0.4, 0.4], unit="m/s"),
        }
    )

    assert trace.sample_count == 2
    analysis = ReplayAnalysis(metrics={"within_tolerance": False}, trace=trace)
    assert analysis.trace == trace


def test_replay_trace_defaults_old_artifacts_to_scalar_command():
    trace = ReplayTrace.model_validate(
        {"series": {"expected": {"values": [0.0]}}}
    )

    assert trace.trace_type == "scalar_command"


def test_replay_trace_accepts_aloha_action_type():
    trace = ReplayTrace(
        trace_type="aloha_action",
        series={"time_s": SignalSeries(values=[0.0], unit="s")},
    )

    assert trace.trace_type == "aloha_action"


@pytest.mark.parametrize(
    "payload",
    [
        {"series": {}},
        {"series": {"expected": {"values": []}}},
        {
            "series": {
                "expected": {"values": [0.0]},
                "observed": {"values": [0.0, 0.1]},
            }
        },
        {"series": {"../escape": {"values": [0.0]}}},
        {"series": {"expected": {"values": [float("nan")]}}},
    ],
)
def test_replay_trace_rejects_unsound_series(payload):
    with pytest.raises(ValueError):
        ReplayTrace.model_validate(payload)


def test_parse_flat_eval_info():
    eval_info = json.loads((FIXTURES / "eval_info_flat.json").read_text())
    parsed = _parse_eval_info(
        eval_info, suite="libero_object", policy_id="test/policy", seed=1000
    )
    assert parsed["overall_pc_success"] == 80.0
    assert len(parsed["tasks"]) == 1
    assert parsed["tasks"][0]["pc_success"] == 80.0
    assert parsed["tasks"][0]["n_episodes"] == 5
    assert parsed["tasks"][0]["n_successes"] == 4
    assert len(parsed["episodes"]) == 5
    assert parsed["episodes"][2]["success"] is False


def test_manifest_roundtrip(tmp_path: Path):
    eval_info = json.loads((FIXTURES / "eval_info_flat.json").read_text())
    parsed = _parse_eval_info(
        eval_info, suite="libero_object", policy_id="test/policy", seed=1000
    )
    manifest = RunManifest(
        run_id="run_test_001",
        policy_id="test/policy",
        suite="libero_object",
        seed=1000,
        n_episodes=5,
        task_ids=[0],
        status="succeeded",
        gpu_seconds=42.0,
        gpu_type="L4",
        tasks=[TaskResult.model_validate(t) for t in parsed["tasks"]],
        overall_pc_success=parsed["overall_pc_success"],
        eval_info=eval_info,
        episodes=[EpisodeRecord.model_validate(e) for e in parsed["episodes"]],
    )
    save_run(manifest, tmp_path)
    loaded = load_run("run_test_001", tmp_path)
    assert loaded.policy_id == "test/policy"
    assert loaded.overall_pc_success == 80.0
    assert len(loaded.episodes) == 5
    assert (tmp_path / "run_test_001" / "eval_info.json").exists()
    assert "run_test_001" in list_runs(tmp_path)


def test_parse_nested_eval_info():
    eval_info = json.loads((FIXTURES / "eval_info_nested.json").read_text())
    parsed = _parse_eval_info(
        eval_info, suite="libero_object", policy_id="test/policy", seed=1000
    )
    assert parsed["overall_pc_success"] == 80.0
    assert len(parsed["tasks"]) == 1
    assert parsed["tasks"][0]["task"] == "libero_object/0"
    assert parsed["tasks"][0]["n_successes"] == 4
    assert len(parsed["episodes"]) == 5


def test_parse_day1_real_schema():
    """Real schema captured from Modal derisk (lerobot 0.4.4 list per_task)."""
    eval_info = json.loads((FIXTURES / "eval_info_day1.json").read_text())
    parsed = _parse_eval_info(
        eval_info,
        suite="libero_object",
        policy_id="HuggingFaceVLA/smolvla_libero",
        seed=4242,
    )
    assert parsed["overall_pc_success"] == 0.0
    assert len(parsed["tasks"]) == 1
    assert parsed["tasks"][0]["task"] == "libero_object/0"
    assert parsed["tasks"][0]["n_episodes"] == 1
    assert len(parsed["episodes"]) == 1
    assert parsed["episodes"][0]["success"] is False
    assert parsed["episodes"][0]["seed"] == 4242
    assert parsed["tasks"][0]["seeds"] == [4242]


def test_aggregate_video_copy_rewrites_existing_episode_paths(tmp_path: Path):
    destination = tmp_path / "aggregate"
    videos = destination / "videos"
    videos.mkdir(parents=True)
    (videos / "shard_0000_episode.mp4").write_bytes(b"video")
    manifest = RunManifest(
        run_id="aggregate",
        policy_id="org/model",
        suite="libero_object",
        seed=1000,
        n_episodes=1,
        status="succeeded",
        episodes=[
            EpisodeRecord(
                episode_id="episode",
                task="libero_object/0",
                suite="libero_object",
                seed=1000,
                success=True,
                policy_id="org/model",
                video_path="videos/episode.mp4",
            )
        ],
    )

    localized = _attach_disk_videos(manifest, destination, force_rewrite=True)

    assert localized.episodes[0].video_path == "videos/shard_0000_episode.mp4"
