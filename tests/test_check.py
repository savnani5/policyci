"""Runner-independent RoboLens check orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from policyci.check import _artifact_refs_for_unit, execute_check
from policyci.checkspec import CaseSpec, CheckSuite, ExecutionSpec, GateSpec, RunnerSpec
from policyci.runner.base import PreflightResult
from policyci.schemas import (
    ArtifactRef,
    CaseExecution,
    EpisodeRecord,
    ExecutionValidity,
    GateVerdict,
    SubjectRef,
    TaskStatus,
)


class FakeRunner:
    def __init__(
        self,
        values: dict[tuple[str, str], float | bool],
        *,
        invalid: set[tuple[str, str]] | None = None,
        unit_ids: dict[tuple[str, str], list[str]] | None = None,
        artifact_units: set[str] | None = None,
        zero_byte_artifacts: bool = False,
        runner_name: str = "fake",
    ):
        self.values = values
        self.invalid = invalid or set()
        self.unit_ids = unit_ids or {}
        self.artifact_units = artifact_units
        self.zero_byte_artifacts = zero_byte_artifacts
        self.runner_name = runner_name

    def preflight(self, subject: SubjectRef, case: CaseSpec) -> PreflightResult:
        if (subject.id, case.id) in self.invalid:
            return PreflightResult.no_verdict("fixture is unavailable")
        return PreflightResult.valid()

    def run_case(
        self,
        subject: SubjectRef,
        case: CaseSpec,
        output_dir: Path,
    ) -> CaseExecution:
        units = self.unit_ids.get((subject.id, case.id), ["1000", "1001"])
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        episodes = []
        artifacts = []
        for unit in units:
            has_artifact = self.artifact_units is None or unit in self.artifact_units
            video_path = None
            if has_artifact:
                video = artifacts_dir / f"{unit}.mp4"
                video.write_bytes(
                    b""
                    if self.zero_byte_artifacts
                    else f"{subject.id}:{case.id}:{unit}".encode()
                )
                video_path = f"artifacts/{unit}.mp4"
                artifacts.append(ArtifactRef(kind="video", path=video_path))
            episodes.append(
                EpisodeRecord(
                    episode_id=f"{case.id}-{unit}",
                    task=case.id,
                    suite="fake",
                    seed=int(unit),
                    success=True,
                    policy_id=subject.id,
                    video_path=video_path,
                )
            )
        return CaseExecution(
            subject=subject,
            case_id=case.id,
            runner=self.runner_name,
            validity=ExecutionValidity.VALID,
            metrics={
                "success_rate": self.values[(subject.id, case.id)],
                "collision_free": self.values[(subject.id, case.id)],
            },
            valid_units=len(units),
            unit_ids=units,
            episodes=episodes,
            artifacts=artifacts,
            synthetic=True,
        )


def _subject(subject_id: str) -> SubjectRef:
    return SubjectRef(id=subject_id, checkpoint=f"fixtures/{subject_id}")


def _suite(*cases: CaseSpec, gate: GateSpec | None = None) -> CheckSuite:
    return CheckSuite(
        version=1,
        name="release",
        runner=RunnerSpec(type="libero", suite="libero_object"),
        execution=ExecutionSpec(episodes=2, seeds=[1000, 1001]),
        gate=gate
        or GateSpec(
            metric="success_rate",
            mode="max_drop",
            max_drop_pp=10,
            min_valid_episodes=2,
        ),
        cases=list(cases) or [CaseSpec(id="soup", task=0)],
    )


def test_regression_blocks_and_writes_portable_paired_evidence(tmp_path: Path):
    baseline = _subject("baseline")
    candidate = _subject("candidate")
    suite = _suite(CaseSpec(id="soup", task=0))
    runner = FakeRunner({("baseline", "soup"): 90.0, ("candidate", "soup"): 60.0})

    result, root = execute_check(
        suite,
        baseline,
        candidate,
        runner,
        runs_root=tmp_path,
        check_id="check_block",
    )

    assert result.gate.verdict is GateVerdict.BLOCK
    assert result.gate.exit_code == 1
    assert result.comparisons[0].status is TaskStatus.REGRESSED
    assert (root / "check.json").exists()
    assert (root / "result.json").exists()
    assert (root / "baseline/cases/soup/execution.json").exists()
    assert (root / "candidate/cases/soup/execution.json").exists()
    assert (root / "pairs/soup/1000/pair.json").exists()
    pair = json.loads((root / "pairs/soup/1000/pair.json").read_text())
    assert pair["baseline_artifacts"][0]["path"] == (
        "baseline/cases/soup/artifacts/1000.mp4"
    )
    assert str(tmp_path) not in (root / "result.json").read_text()


def test_unchanged_candidate_passes(tmp_path: Path):
    runner = FakeRunner({("baseline", "soup"): 90.0, ("candidate", "soup"): 85.0})
    result, _ = execute_check(
        _suite(),
        _subject("baseline"),
        _subject("candidate"),
        runner,
        runs_root=tmp_path,
    )
    assert result.gate.verdict is GateVerdict.PASS
    assert result.gate.exit_code == 0


def test_candidate_preflight_failure_is_no_verdict(tmp_path: Path):
    runner = FakeRunner(
        {("baseline", "soup"): 90.0, ("candidate", "soup"): 90.0},
        invalid={("candidate", "soup")},
    )
    result, root = execute_check(
        _suite(),
        _subject("baseline"),
        _subject("candidate"),
        runner,
        runs_root=tmp_path,
    )
    assert result.gate.verdict is GateVerdict.NO_VERDICT
    assert result.gate.exit_code == 2
    assert result.comparisons[0].status is TaskStatus.NOT_EVALUATED
    candidate_execution = json.loads(
        (root / "candidate/cases/soup/execution.json").read_text()
    )
    assert candidate_execution["validity"] == "no_verdict"
    assert candidate_execution["reasons"] == ["fixture is unavailable"]


def test_required_regression_precedes_other_no_verdict(tmp_path: Path):
    suite = _suite(CaseSpec(id="soup", task=0), CaseSpec(id="mug", task=1))
    runner = FakeRunner(
        {
            ("baseline", "soup"): 90.0,
            ("candidate", "soup"): 20.0,
            ("baseline", "mug"): 90.0,
            ("candidate", "mug"): 90.0,
        },
        invalid={("candidate", "mug")},
    )
    result, _ = execute_check(
        suite,
        _subject("baseline"),
        _subject("candidate"),
        runner,
        runs_root=tmp_path,
    )
    assert result.gate.verdict is GateVerdict.BLOCK


def test_optional_no_verdict_does_not_prevent_pass(tmp_path: Path):
    suite = _suite(
        CaseSpec(id="soup", task=0),
        CaseSpec(id="mug", task=1, required=False),
    )
    runner = FakeRunner(
        {
            ("baseline", "soup"): 90.0,
            ("candidate", "soup"): 90.0,
            ("baseline", "mug"): 90.0,
            ("candidate", "mug"): 90.0,
        },
        invalid={("candidate", "mug")},
    )
    result, _ = execute_check(
        suite,
        _subject("baseline"),
        _subject("candidate"),
        runner,
        runs_root=tmp_path,
    )
    assert result.gate.verdict is GateVerdict.PASS


def test_mismatched_unit_identity_is_no_verdict(tmp_path: Path):
    runner = FakeRunner(
        {("baseline", "soup"): 90.0, ("candidate", "soup"): 90.0},
        unit_ids={
            ("baseline", "soup"): ["1000", "1001"],
            ("candidate", "soup"): ["1000", "1002"],
        },
    )
    result, _ = execute_check(
        _suite(),
        _subject("baseline"),
        _subject("candidate"),
        runner,
        runs_root=tmp_path,
    )
    assert result.gate.verdict is GateVerdict.NO_VERDICT
    assert "unit identities differ" in result.comparisons[0].reason


def test_not_evaluated_comparison_keeps_pairs_for_common_units(tmp_path: Path):
    runner = FakeRunner(
        {("baseline", "soup"): 90.0, ("candidate", "soup"): 90.0},
        unit_ids={
            ("baseline", "soup"): ["1000", "1001"],
            ("candidate", "soup"): ["1000", "1002"],
        },
    )

    result, _ = execute_check(
        _suite(),
        _subject("baseline"),
        _subject("candidate"),
        runner,
        runs_root=tmp_path,
    )

    assert result.comparisons[0].status is TaskStatus.NOT_EVALUATED
    assert [pair.unit_id for pair in result.comparisons[0].evidence_pairs] == ["1000"]


def test_valid_execution_without_evidence_is_no_verdict(tmp_path: Path):
    runner = FakeRunner(
        {("baseline", "soup"): 90.0, ("candidate", "soup"): 90.0},
        artifact_units=set(),
    )

    result, root = execute_check(
        _suite(),
        _subject("baseline"),
        _subject("candidate"),
        runner,
        runs_root=tmp_path,
    )

    assert result.gate.verdict is GateVerdict.NO_VERDICT
    execution = json.loads((root / "baseline/cases/soup/execution.json").read_text())
    assert "evidence" in " ".join(execution["reasons"])


def test_zero_byte_evidence_is_no_verdict(tmp_path: Path):
    runner = FakeRunner(
        {("baseline", "soup"): 90.0, ("candidate", "soup"): 90.0},
        zero_byte_artifacts=True,
    )

    result, _ = execute_check(
        _suite(),
        _subject("baseline"),
        _subject("candidate"),
        runner,
        runs_root=tmp_path,
    )

    assert result.gate.verdict is GateVerdict.NO_VERDICT


def test_corrupt_libero_video_is_no_verdict(tmp_path: Path, monkeypatch):
    runner = FakeRunner(
        {("baseline", "soup"): 90.0, ("candidate", "soup"): 90.0},
        runner_name="libero",
    )
    monkeypatch.setattr(
        "policyci.check._probe_video",
        lambda path: (False, f"{path.name}: invalid media"),
    )

    result, root = execute_check(
        _suite(),
        _subject("baseline"),
        _subject("candidate"),
        runner,
        runs_root=tmp_path,
    )

    assert result.gate.verdict is GateVerdict.NO_VERDICT
    execution = json.loads((root / "baseline/cases/soup/execution.json").read_text())
    assert "corrupt" in " ".join(execution["reasons"])


def test_require_value_blocks_bad_candidate_and_marks_fix(tmp_path: Path):
    gate = GateSpec(
        metric="collision_free",
        mode="require_value",
        expect=True,
        min_valid_episodes=2,
    )
    suite = _suite(gate=gate)
    bad = FakeRunner(
        {("baseline", "soup"): True, ("candidate", "soup"): False}
    )
    bad_result, _ = execute_check(
        suite,
        _subject("baseline"),
        _subject("candidate"),
        bad,
        runs_root=tmp_path,
        check_id="bad",
    )
    assert bad_result.gate.verdict is GateVerdict.BLOCK

    fixed = FakeRunner(
        {("baseline", "soup"): False, ("candidate", "soup"): True}
    )
    fixed_result, _ = execute_check(
        suite,
        _subject("baseline"),
        _subject("candidate"),
        fixed,
        runs_root=tmp_path,
        check_id="fixed",
    )
    assert fixed_result.gate.verdict is GateVerdict.PASS
    assert fixed_result.comparisons[0].status is TaskStatus.FIXED


def test_non_finite_numeric_metric_is_not_evaluated(tmp_path: Path):
    runner = FakeRunner(
        {("baseline", "soup"): 90.0, ("candidate", "soup"): float("nan")}
    )

    result, _ = execute_check(
        _suite(),
        _subject("baseline"),
        _subject("candidate"),
        runner,
        runs_root=tmp_path,
    )

    assert result.gate.verdict is GateVerdict.NO_VERDICT
    assert result.comparisons[0].status is TaskStatus.NOT_EVALUATED
    assert "finite" in result.comparisons[0].reason


def test_require_value_boolean_expectation_is_type_strict(tmp_path: Path):
    gate = GateSpec(
        metric="collision_free",
        mode="require_value",
        expect=True,
        min_valid_episodes=2,
    )
    runner = FakeRunner(
        {("baseline", "soup"): True, ("candidate", "soup"): 1}
    )

    result, _ = execute_check(
        _suite(gate=gate),
        _subject("baseline"),
        _subject("candidate"),
        runner,
        runs_root=tmp_path,
    )

    assert result.gate.verdict is GateVerdict.NO_VERDICT
    assert "type" in result.comparisons[0].reason


def test_replay_check_metadata_does_not_serialize_absolute_bag_path(tmp_path: Path):
    bag = tmp_path / "customer-data" / "input.mcap"
    bag.parent.mkdir()
    bag.write_bytes(b"fixture")
    suite = CheckSuite(
        version=1,
        name="replay",
        runner=RunnerSpec(type="ros2_replay", adapter="example.module:Adapter"),
        gate=GateSpec(
            metric="success_rate",
            mode="max_drop",
            max_drop_pp=10,
            min_valid_repeats=1,
        ),
        cases=[
            CaseSpec(
                id="soup",
                bag=bag,
                required_topics=["/input"],
                capture_topics=["/output"],
                judge={"tolerance": 0.05},
            )
        ],
    )
    runner = FakeRunner({("baseline", "soup"): 90.0, ("candidate", "soup"): 90.0})
    result, root = execute_check(
        suite,
        _subject("baseline"),
        _subject("candidate"),
        runner,
        runs_root=tmp_path / "runs",
    )
    check_json = (root / "check.json").read_text()
    assert str(tmp_path) not in check_json
    assert '"bag": "input.mcap"' in check_json
    assert result.suite_version == 1
    assert result.case_judges == {"soup": {"tolerance": 0.05}}
    result_json = json.loads((root / "result.json").read_text())
    assert result_json["case_judges"] == {"soup": {"tolerance": 0.05}}


def test_pairs_nested_replay_artifact_by_unit_directory():
    execution = CaseExecution(
        subject=_subject("baseline"),
        case_id="stop_drift",
        runner="ros2_replay",
        validity=ExecutionValidity.VALID,
        metrics={"within_tolerance": False},
        valid_units=1,
        unit_ids=["repeat-000"],
        artifacts=[
            ArtifactRef(
                kind="mcap",
                path="repeats/repeat-000/capture/capture_0.mcap",
            )
        ],
    )

    refs = _artifact_refs_for_unit(
        execution,
        "repeat-000",
        "baseline",
        "stop_drift",
    )

    assert refs == [
        ArtifactRef(
            kind="mcap",
            path=(
                "baseline/cases/stop_drift/"
                "repeats/repeat-000/capture/capture_0.mcap"
            ),
        )
    ]


def test_pairs_episode_video_and_additional_unit_artifact_once():
    execution = CaseExecution(
        subject=_subject("baseline"),
        case_id="grasp_offset",
        runner="mujoco",
        validity=ExecutionValidity.VALID,
        metrics={"lift_success": False},
        valid_units=1,
        unit_ids=["1000"],
        episodes=[
            EpisodeRecord(
                episode_id="grasp-1000",
                task="grasp_offset",
                suite="mujoco",
                seed=1000,
                success=False,
                policy_id="baseline",
                video_path="repeats/1000/episode.mp4",
            )
        ],
        artifacts=[
            ArtifactRef(kind="video", path="repeats/1000/episode.mp4"),
            ArtifactRef(kind="trajectory", path="repeats/1000/trajectory.json"),
        ],
    )

    refs = _artifact_refs_for_unit(
        execution,
        "1000",
        "baseline",
        "grasp_offset",
    )

    assert refs == [
        ArtifactRef(
            kind="video",
            path="baseline/cases/grasp_offset/repeats/1000/episode.mp4",
        ),
        ArtifactRef(
            kind="trajectory",
            path="baseline/cases/grasp_offset/repeats/1000/trajectory.json",
        ),
    ]
