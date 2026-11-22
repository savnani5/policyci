"""Static, paired evidence report for release decisions."""

from __future__ import annotations

from pathlib import Path

from policyci.report.html import render_check_report
from policyci.schemas import (
    ArtifactRef,
    CheckResult,
    Comparison,
    EvidencePair,
    ExecutionValidity,
    GateResult,
    GateVerdict,
    SubjectRef,
    TaskStatus,
)


def _result(verdict: GateVerdict = GateVerdict.BLOCK) -> CheckResult:
    return CheckResult(
        check_id="check_demo",
        suite_name="manipulation-release",
        runner="libero",
        baseline=SubjectRef(
            id="baseline",
            checkpoint="org/model",
            code_sha="aaaa111",
            config={"n_action_steps": 10},
        ),
        candidate=SubjectRef(
            id="candidate",
            checkpoint="org/model",
            code_sha="bbbb222",
            config={"n_action_steps": 4},
        ),
        comparisons=[
            Comparison(
                case_id="soup",
                status=TaskStatus.REGRESSED
                if verdict is GateVerdict.BLOCK
                else TaskStatus.NOT_EVALUATED,
                baseline_validity=ExecutionValidity.VALID,
                candidate_validity=ExecutionValidity.VALID
                if verdict is GateVerdict.BLOCK
                else ExecutionValidity.NO_VERDICT,
                metric="success_rate",
                baseline_value=90.0,
                candidate_value=40.0 if verdict is GateVerdict.BLOCK else None,
                delta=-50.0 if verdict is GateVerdict.BLOCK else None,
                reason="" if verdict is GateVerdict.BLOCK else "candidate output missing",
                evidence_pairs=[
                    EvidencePair(
                        case_id="soup",
                        unit_id="1000",
                        baseline_artifacts=[
                            ArtifactRef(
                                kind="video",
                                path="baseline/cases/soup/artifacts/1000.mp4",
                            )
                        ],
                        candidate_artifacts=[
                            ArtifactRef(
                                kind="video",
                                path="candidate/cases/soup/artifacts/1000.mp4",
                            )
                        ],
                    )
                ],
            )
        ],
        gate=GateResult(
            verdict=verdict,
            exit_code=1 if verdict is GateVerdict.BLOCK else 2,
            summary="BLOCK: soup regressed"
            if verdict is GateVerdict.BLOCK
            else "NO_VERDICT: candidate output missing",
        ),
        gate_config={
            "metric": "success_rate",
            "mode": "max_drop",
            "max_drop_pp": 10,
            "min_valid_episodes": 5,
        },
        suite_version=1,
        case_judges={"soup": {"tolerance": 0.05, "judge_version": "twist-v1"}},
    )


def test_render_check_report_pairs_same_case_and_seed(tmp_path: Path):
    root = tmp_path / "check_demo"
    for side in ("baseline", "candidate"):
        video = root / side / "cases/soup/artifacts/1000.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(side.encode())

    output = root / "evidence.html"
    html = render_check_report(_result(), check_dir=root, output_path=output)

    assert output.exists()
    assert "BLOCK" in html
    assert "baseline" in html and "candidate" in html
    assert "config.n_action_steps" in html
    assert "10" in html and "4" in html
    assert "soup" in html and "1000" in html
    assert "success_rate" in html and "-50.0" in html
    assert "twist-v1" in html and "0.05" in html
    assert "Suite version" in html and ">1<" in html
    assert "baseline/cases/soup/artifacts/1000.mp4" in html
    assert "candidate/cases/soup/artifacts/1000.mp4" in html
    assert "Robustness profiles" not in html
    assert "Episode videos" not in html


def test_render_check_report_makes_paths_relative_to_external_report(tmp_path: Path):
    root = tmp_path / "runs/check_demo"
    root.mkdir(parents=True)
    output = tmp_path / "reports/evidence.html"
    html = render_check_report(_result(), check_dir=root, output_path=output)
    assert "../runs/check_demo/baseline/cases/soup/artifacts/1000.mp4" in html


def test_render_check_report_explains_no_verdict(tmp_path: Path):
    root = tmp_path / "check_demo"
    root.mkdir()
    html = render_check_report(
        _result(GateVerdict.NO_VERDICT),
        check_dir=root,
        output_path=root / "evidence.html",
    )
    assert "NO_VERDICT" in html
    assert "candidate output missing" in html
    assert "not_evaluated" in html
