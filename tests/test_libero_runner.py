"""LIBERO adapter for the runner-independent RoboLens check contract."""

from __future__ import annotations

from pathlib import Path

from policyci.checkspec import CaseSpec, CheckSuite, ExecutionSpec, GateSpec, RunnerSpec
from policyci.runner.libero import LiberoRunner
from policyci.schemas import (
    EpisodeRecord,
    ExecutionValidity,
    RunManifest,
    SubjectRef,
    TaskResult,
)


def _suite() -> CheckSuite:
    return CheckSuite(
        version=1,
        name="release",
        runner=RunnerSpec(type="libero", suite="libero_object"),
        execution=ExecutionSpec(episodes=2, seeds=[1000, 1001]),
        gate=GateSpec(
            metric="success_rate",
            mode="max_drop",
            max_drop_pp=10,
            min_valid_episodes=2,
        ),
        cases=[CaseSpec(id="soup", task=0)],
    )


def _manifest(
    policy_path: str,
    *,
    status: str = "succeeded",
    seeds: list[int] | None = None,
) -> RunManifest:
    resolved_seeds = seeds or [1000, 1001]
    return RunManifest(
        run_id="aggregate",
        policy_id=policy_path,
        suite="libero_object",
        seed=resolved_seeds[0],
        n_episodes=len(resolved_seeds),
        status=status,
        error="worker failed" if status != "succeeded" else None,
        gpu_type="L4",
        tasks=[
            TaskResult(
                task="libero_object/0",
                task_id=0,
                suite="libero_object",
                n_episodes=len(resolved_seeds),
                n_successes=len(resolved_seeds) - 1,
                pc_success=50.0,
                seeds=resolved_seeds,
            )
        ],
        overall_pc_success=50.0,
        episodes=[
            EpisodeRecord(
                episode_id=f"soup-{seed}",
                task="libero_object/0",
                task_id=0,
                suite="libero_object",
                seed=seed,
                success=index == 0,
                policy_id=policy_path,
                video_path=f"videos/{seed}.mp4",
            )
            for index, seed in enumerate(resolved_seeds)
        ],
    )


def test_preflight_rejects_missing_checkpoint():
    runner = LiberoRunner(_suite(), eval_fanout=lambda *args, **kwargs: None)
    result = runner.preflight(SubjectRef(id="candidate"), _suite().cases[0])
    assert result.validity is ExecutionValidity.NO_VERDICT
    assert "checkpoint" in result.reasons[0]


def test_preflight_rejects_unknown_subject_config():
    runner = LiberoRunner(_suite(), eval_fanout=lambda *args, **kwargs: None)
    subject = SubjectRef(
        id="candidate",
        checkpoint="org/model",
        config={"temperature": 0.5},
    )
    result = runner.preflight(subject, _suite().cases[0])
    assert result.validity is ExecutionValidity.NO_VERDICT
    assert "temperature" in result.reasons[0]


def test_preflight_rejects_declared_code_identity_it_does_not_apply():
    runner = LiberoRunner(_suite(), eval_fanout=lambda *args, **kwargs: None)
    subject = SubjectRef(
        id="candidate",
        checkpoint="org/model",
        image="robot:latest",
        code_sha="deadbeef",
    )

    result = runner.preflight(subject, _suite().cases[0])

    assert result.validity is ExecutionValidity.NO_VERDICT
    assert "image" in result.reasons[0]
    assert "code_sha" in result.reasons[0]


def test_run_case_uses_exact_seeds_and_subject_inference_config(tmp_path: Path):
    observed: dict = {}

    def eval_fanout(policy_path: str, shards: list[dict], **kwargs) -> RunManifest:
        observed.update(policy_path=policy_path, shards=shards, kwargs=kwargs)
        manifest = _manifest(policy_path)
        source = kwargs["runs_root"] / manifest.run_id / "videos"
        source.mkdir(parents=True)
        for seed in [1000, 1001]:
            (source / f"{seed}.mp4").write_bytes(f"video-{seed}".encode())
        return manifest

    subject = SubjectRef(
        id="candidate",
        checkpoint="org/candidate",
        config={"n_action_steps": 4, "batch_size": 1},
    )
    runner = LiberoRunner(_suite(), eval_fanout=eval_fanout)

    execution = runner.run_case(subject, _suite().cases[0], tmp_path / "case")

    assert observed["policy_path"] == "org/candidate"
    assert [shard["seed"] for shard in observed["shards"]] == [1000, 1001]
    assert all(shard["n_episodes"] == 1 for shard in observed["shards"])
    assert all(shard["task_ids"] == [0] for shard in observed["shards"])
    assert all(shard["n_action_steps"] == 4 for shard in observed["shards"])
    assert execution.validity is ExecutionValidity.VALID
    assert execution.unit_ids == ["1000", "1001"]
    assert execution.metrics["success_rate"] == 50.0
    assert execution.artifacts[0].path == "artifacts/1000.mp4"
    assert (tmp_path / "case/artifacts/1000.mp4").exists()


def test_failed_manifest_is_no_verdict(tmp_path: Path):
    def failed_fanout(policy_path: str, shards: list[dict], **kwargs) -> RunManifest:
        del shards
        manifest = _manifest(policy_path, status="failed")
        source = kwargs["runs_root"] / manifest.run_id / "videos"
        source.mkdir(parents=True)
        (source / "1000.mp4").write_bytes(b"partial-evidence")
        return manifest

    runner = LiberoRunner(_suite(), eval_fanout=failed_fanout)
    execution = runner.run_case(
        SubjectRef(id="candidate", checkpoint="org/candidate"),
        _suite().cases[0],
        tmp_path / "case",
    )
    assert execution.validity is ExecutionValidity.NO_VERDICT
    assert "worker failed" in execution.reasons[0]
    assert execution.unit_ids == ["1000", "1001"]
    assert [artifact.model_dump() for artifact in execution.artifacts] == [
        {"kind": "video", "path": "artifacts/1000.mp4"}
    ]
    assert (tmp_path / "case/artifacts/1000.mp4").read_bytes() == b"partial-evidence"


def test_seed_mismatch_is_no_verdict(tmp_path: Path):
    runner = LiberoRunner(
        _suite(),
        eval_fanout=lambda policy_path, shards, **kwargs: _manifest(
            policy_path, seeds=[1000, 1002]
        ),
    )
    execution = runner.run_case(
        SubjectRef(id="candidate", checkpoint="org/candidate"),
        _suite().cases[0],
        tmp_path / "case",
    )
    assert execution.validity is ExecutionValidity.NO_VERDICT
    assert "seeds" in execution.reasons[0]
