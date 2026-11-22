"""Soundness tests for the physics-backed manipulation runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from policyci.checkspec import CaseSpec, CheckSuite, ExecutionSpec, GateSpec, RunnerSpec
from policyci.runner.mujoco import MujocoRunner, SimulationEpisode, _default_adapter
from policyci.schemas import (
    ExecutionValidity,
    SimulationSample,
    SimulationTrace,
    SubjectRef,
)


def mujoco_suite(*, seeds: list[int] | None = None) -> CheckSuite:
    resolved_seeds = seeds or [1000]
    return CheckSuite(
        version=1,
        name="mujoco-grasp-regression",
        runner=RunnerSpec(type="mujoco"),
        execution=ExecutionSpec(episodes=len(resolved_seeds), seeds=resolved_seeds),
        gate=GateSpec(
            metric="lift_success",
            mode="require_value",
            expect=True,
            min_valid_repeats=len(resolved_seeds),
        ),
        cases=[mujoco_case()],
    )


def mujoco_case() -> CaseSpec:
    return CaseSpec(
        id="grasp_offset",
        judge={"lift_threshold_m": 0.08, "hold_duration_s": 0.5},
    )


def pick_place_case() -> CaseSpec:
    return CaseSpec(
        id="pick_place_offset",
        judge={"placement_hold_duration_s": 0.5},
    )


def pick_place_suite() -> CheckSuite:
    return CheckSuite(
        version=1,
        name="mujoco-pick-place-regression",
        runner=RunnerSpec(type="mujoco"),
        execution=ExecutionSpec(episodes=1, seeds=[1000]),
        gate=GateSpec(
            metric="placement_success",
            mode="require_value",
            expect=True,
            min_valid_repeats=1,
        ),
        cases=[pick_place_case()],
    )


def candidate_subject(
    *,
    offset: object = (0.0, 0.0, 0.0),
) -> SubjectRef:
    return SubjectRef(
        id="candidate",
        code_sha="demo-code",
        config={"action_frame_offset_m": offset},
    )


def _trace(seed: int, *, lifted: bool = True) -> SimulationTrace:
    samples = []
    for index in range(15):
        lift = 0.09 if lifted and index >= 4 else 0.0
        contact = lifted and index >= 4
        samples.append(
            SimulationSample(
                time_s=index * 0.05,
                object_position_m=(0.0, 0.0, 0.02 + lift),
                object_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                eef_position_m=(0.0, 0.0, 0.10 + lift),
                eef_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                gripper_width_m=0.02,
                left_finger_contact=contact,
                right_finger_contact=contact,
            )
        )
    return SimulationTrace(
        sample_rate_hz=20.0,
        seed=seed,
        initial_state_sha256="a" * 64,
        model_sha256="b" * 64,
        samples=samples,
    )


@dataclass
class FakeMujocoAdapter:
    applied_fields: set[str] | None = None
    trace_seed: int | None = None
    write_video: bool = True
    runtime_by_seed: bool = False
    placement: bool = False

    def applied_subject_fields(
        self,
        subject: SubjectRef,
        case: CaseSpec,
    ) -> set[str]:
        del subject, case
        if self.applied_fields is not None:
            return self.applied_fields
        return {"code_sha", "config.action_frame_offset_m"}

    def run_episode(
        self,
        subject: SubjectRef,
        case: CaseSpec,
        *,
        seed: int,
        output_dir: Path,
    ) -> SimulationEpisode:
        video = output_dir / "episode.mp4"
        if self.write_video:
            video.write_bytes(b"fake-video")
        offset = subject.config["action_frame_offset_m"]
        lifted = list(offset) == [0.0, 0.0, 0.0]
        runtime_suffix = seed if self.runtime_by_seed else 0
        trace_seed = self.trace_seed if self.trace_seed is not None else seed
        trace = _trace(trace_seed, lifted=lifted)
        if self.placement:
            trace = trace.model_copy(
                update={
                    "samples": [
                        sample.model_copy(
                            update={
                                "target_position_m": (0.20, 0.20, 0.80),
                                "task_success": lifted and index >= 4,
                            }
                        )
                        for index, sample in enumerate(trace.samples)
                    ]
                }
            )
        return SimulationEpisode(
            trace=trace,
            video_path=video,
            runtime={
                "robosuite_version": "1.5.2",
                "mujoco_version": "3.7.0",
                "runtime_suffix": runtime_suffix,
            },
        )


def test_runner_materializes_video_trace_and_exact_metrics(tmp_path: Path):
    runner = MujocoRunner(mujoco_suite(), adapter=FakeMujocoAdapter())

    result = runner.run_case(candidate_subject(), mujoco_case(), tmp_path)

    assert result.validity is ExecutionValidity.VALID
    assert result.metrics["lift_success"] is True
    assert result.metrics["max_object_lift_m"] == pytest.approx(0.09)
    assert result.unit_ids == ["1000"]
    assert {artifact.kind for artifact in result.artifacts} == {
        "video",
        "trajectory",
    }
    assert (tmp_path / "repeats/1000/episode.mp4").stat().st_size > 0
    assert (tmp_path / "repeats/1000/trajectory.json").is_file()
    assert result.environment["initial_state_sha256_by_seed"] == {
        "1000": "a" * 64
    }


def test_runner_derives_pick_place_metric_selected_by_case_judge(tmp_path: Path):
    runner = MujocoRunner(
        pick_place_suite(),
        adapter=FakeMujocoAdapter(placement=True),
    )

    result = runner.run_case(
        candidate_subject(),
        pick_place_case(),
        tmp_path,
    )

    assert result.validity is ExecutionValidity.VALID
    assert result.metrics["placement_success"] is True
    assert result.metrics["placement_success_duration_s"] == pytest.approx(0.5)


def test_default_adapter_dispatches_without_changing_legacy_lift_behavior():
    dispatcher = _default_adapter()

    assert dispatcher.adapter_for(mujoco_case()).__class__.__name__ == (
        "RobosuiteLiftAdapter"
    )
    assert dispatcher.adapter_for(pick_place_case()).__class__.__name__ == (
        "RobosuitePickPlaceAdapter"
    )


def test_runner_rejects_unapplied_subject_configuration():
    adapter = FakeMujocoAdapter(applied_fields=set())
    runner = MujocoRunner(mujoco_suite(), adapter=adapter)

    preflight = runner.preflight(candidate_subject(), mujoco_case())

    assert preflight.validity is ExecutionValidity.NO_VERDICT
    assert "action_frame_offset_m" in " ".join(preflight.reasons)


@pytest.mark.parametrize(
    "offset",
    [None, [0.0, 0.0], [0.0, 0.0, 0.11], [0.0, float("nan"), 0.0], True],
)
def test_runner_rejects_invalid_action_frame_offset(offset: object):
    runner = MujocoRunner(mujoco_suite(), adapter=FakeMujocoAdapter())

    preflight = runner.preflight(candidate_subject(offset=offset), mujoco_case())

    assert preflight.validity is ExecutionValidity.NO_VERDICT
    assert "action_frame_offset_m" in " ".join(preflight.reasons)


def test_runner_rejects_unknown_subject_config():
    subject = candidate_subject().model_copy(
        update={
            "config": {
                "action_frame_offset_m": [0.0, 0.0, 0.0],
                "gain": 20,
            }
        }
    )

    preflight = MujocoRunner(
        mujoco_suite(),
        adapter=FakeMujocoAdapter(),
    ).preflight(subject, mujoco_case())

    assert preflight.validity is ExecutionValidity.NO_VERDICT
    assert "gain" in " ".join(preflight.reasons)


def test_runner_rejects_trace_seed_mismatch(tmp_path: Path):
    adapter = FakeMujocoAdapter(trace_seed=999)

    result = MujocoRunner(mujoco_suite(), adapter=adapter).run_case(
        candidate_subject(),
        mujoco_case(),
        tmp_path,
    )

    assert result.validity is ExecutionValidity.NO_VERDICT
    assert "seed" in " ".join(result.reasons)


def test_runner_rejects_missing_video(tmp_path: Path):
    adapter = FakeMujocoAdapter(write_video=False)

    result = MujocoRunner(mujoco_suite(), adapter=adapter).run_case(
        candidate_subject(),
        mujoco_case(),
        tmp_path,
    )

    assert result.validity is ExecutionValidity.NO_VERDICT
    assert "video" in " ".join(result.reasons)


def test_runner_rejects_inconsistent_repeat_runtime_identity(tmp_path: Path):
    suite = mujoco_suite(seeds=[1000, 1001])
    adapter = FakeMujocoAdapter(runtime_by_seed=True)

    result = MujocoRunner(suite, adapter=adapter).run_case(
        candidate_subject(),
        mujoco_case(),
        tmp_path,
    )

    assert result.validity is ExecutionValidity.NO_VERDICT
    assert "runtime" in " ".join(result.reasons)
