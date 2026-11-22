"""End-to-end file contract from YAML inputs to evidence output."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import policyci.cli as cli
from policyci.runner.base import PreflightResult
from policyci.schemas import (
    ArtifactRef,
    CaseExecution,
    EpisodeRecord,
    ExecutionValidity,
)


class ArtifactRunner:
    def __init__(self, *, omit_candidate_artifact: bool = False):
        self.omit_candidate_artifact = omit_candidate_artifact

    def preflight(self, subject, case):
        return PreflightResult.valid()

    def run_case(self, subject, case, output_dir):
        relative = "artifacts/1000.mp4"
        video = output_dir / relative
        if not (self.omit_candidate_artifact and subject.id == "candidate"):
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(subject.id.encode())
        return CaseExecution(
            subject=subject,
            case_id=case.id,
            runner="fake",
            validity=ExecutionValidity.VALID,
            metrics={"success_rate": 90.0 if subject.id == "baseline" else 40.0},
            valid_units=1,
            unit_ids=["1000"],
            episodes=[
                EpisodeRecord(
                    episode_id=f"{subject.id}-1000",
                    task=case.id,
                    suite="fake",
                    seed=1000,
                    success=subject.id == "baseline",
                    policy_id=subject.id,
                    video_path=relative,
                )
            ],
            artifacts=[ArtifactRef(kind="video", path=relative)],
            synthetic=True,
        )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """
version: 1
name: release
runner: {type: libero, suite: libero_object}
execution: {episodes: 1, seeds: [1000]}
gate: {metric: success_rate, mode: max_drop, max_drop_pp: 10, min_valid_episodes: 1}
cases:
  - {id: soup, task: 0}
"""
    )
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text("id: baseline\ncheckpoint: fixture/baseline\n")
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text("id: candidate\ncheckpoint: fixture/candidate\n")
    return suite, baseline, candidate


def _invoke(tmp_path: Path, suite: Path, baseline: Path, candidate: Path):
    return CliRunner().invoke(
        cli.app,
        [
            "check",
            "--suite",
            str(suite),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )


def test_yaml_to_paired_block_evidence_and_missing_artifact_no_verdict(
    tmp_path: Path,
    monkeypatch,
):
    suite, baseline, candidate = _write_inputs(tmp_path)
    monkeypatch.setattr(cli, "_build_check_runner", lambda loaded: ArtifactRunner())

    blocked = _invoke(tmp_path, suite, baseline, candidate)

    assert blocked.exit_code == 1
    assert "BLOCK" in blocked.stdout
    first_root = next((tmp_path / "runs").glob("check_*"))
    assert (first_root / "evidence.html").exists()
    assert (first_root / "pairs/soup/1000/pair.json").exists()
    assert str(tmp_path) not in (first_root / "result.json").read_text()

    second_tmp = tmp_path / "missing"
    second_tmp.mkdir()
    suite, baseline, candidate = _write_inputs(second_tmp)
    monkeypatch.setattr(
        cli,
        "_build_check_runner",
        lambda loaded: ArtifactRunner(omit_candidate_artifact=True),
    )

    incomplete = _invoke(second_tmp, suite, baseline, candidate)

    assert incomplete.exit_code == 2
    assert "NO_VERDICT" in incomplete.stdout
