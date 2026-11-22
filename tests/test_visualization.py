"""Pure loading and validation for paired RoboLens visualizations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from policyci import visualization
from policyci.schemas import (
    ArtifactRef,
    CheckResult,
    Comparison,
    EvidencePair,
    ExecutionValidity,
    GateResult,
    GateVerdict,
    ReplayTrace,
    SignalSeries,
    SubjectRef,
    TaskStatus,
)
from policyci.visualization import (
    CheckVisualization,
    load_check_visualization,
    load_visualization,
)


def _write_trace(path: Path, *, expected: list[float], observed: list[float]) -> None:
    errors = [abs(actual - target) for target, actual in zip(expected, observed)]
    trace = ReplayTrace(
        series={
            "expected": SignalSeries(values=expected, unit="m/s"),
            "observed": SignalSeries(values=observed, unit="m/s"),
            "absolute_error": SignalSeries(values=errors, unit="m/s"),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(trace.model_dump_json(indent=2))


def _write_check(tmp_path: Path) -> Path:
    root = tmp_path / "check_demo"
    case_id = "zero_command_drift"
    unit_id = "repeat-000"
    side_artifacts: dict[str, list[ArtifactRef]] = {}
    for side, observed in (("baseline", [0.4, 0.4]), ("candidate", [0.0, 0.0])):
        prefix = f"{side}/cases/{case_id}/repeats/{unit_id}"
        trace_relative = f"{prefix}/trace.json"
        mcap_relative = f"{prefix}/capture/capture_0.mcap"
        _write_trace(root / trace_relative, expected=[0.0, 0.0], observed=observed)
        mcap = root / mcap_relative
        mcap.parent.mkdir(parents=True, exist_ok=True)
        mcap.write_bytes(f"{side}-mcap".encode())
        side_artifacts[side] = [
            ArtifactRef(kind="mcap", path=mcap_relative),
            ArtifactRef(kind="trace", path=trace_relative),
        ]

    result = CheckResult(
        check_id="check_demo",
        suite_name="twist-replay",
        runner="ros2_replay",
        baseline=SubjectRef(id="known-bad", config={"bias": 0.4}),
        candidate=SubjectRef(id="fixed", config={"bias": 0.0}),
        comparisons=[
            Comparison(
                case_id="unchanged_case",
                status=TaskStatus.UNCHANGED,
                baseline_validity=ExecutionValidity.VALID,
                candidate_validity=ExecutionValidity.VALID,
                metric="within_tolerance",
                baseline_value=True,
                candidate_value=True,
            ),
            Comparison(
                case_id=case_id,
                status=TaskStatus.FIXED,
                baseline_validity=ExecutionValidity.VALID,
                candidate_validity=ExecutionValidity.VALID,
                metric="within_tolerance",
                baseline_value=False,
                candidate_value=True,
                evidence_pairs=[
                    EvidencePair(
                        case_id=case_id,
                        unit_id=unit_id,
                        baseline_artifacts=side_artifacts["baseline"],
                        candidate_artifacts=side_artifacts["candidate"],
                    )
                ],
            ),
        ],
        gate=GateResult(
            verdict=GateVerdict.PASS,
            exit_code=0,
            summary="PASS: all required cases satisfy the configured gate",
        ),
        gate_config={"metric": "within_tolerance", "mode": "require_value"},
        case_judges={case_id: {"tolerance": 0.05}},
    )
    root.mkdir(exist_ok=True)
    (root / "result.json").write_text(result.model_dump_json(indent=2))
    return root


def _result_payload(root: Path) -> dict:
    return json.loads((root / "result.json").read_text())


def _write_result_payload(root: Path, payload: dict) -> None:
    (root / "result.json").write_text(json.dumps(payload, indent=2))


def test_loader_prefers_changed_case_and_first_repeat(tmp_path: Path):
    root = _write_check(tmp_path)

    model = load_check_visualization(root)

    assert model.case_id == "zero_command_drift"
    assert model.unit_id == "repeat-000"
    assert model.comparison_status is TaskStatus.FIXED
    assert model.verdict is GateVerdict.PASS
    assert model.baseline_observed == [0.4, 0.4]
    assert model.candidate_observed == [0.0, 0.0]
    assert model.expected == [0.0, 0.0]
    assert model.tolerance == 0.05
    assert model.baseline_unit_within_tolerance is False
    assert model.candidate_unit_within_tolerance is True
    assert len(model.baseline_mcaps) == 1
    assert len(model.candidate_mcaps) == 1
    assert "FIXED" in model.summary_markdown
    assert "Aggregate automatic metric" in model.summary_markdown
    assert "Selected repeat from exact samples" in model.summary_markdown
    assert "automatically" in model.summary_markdown


def test_first_violation_finds_candidate_regression_boundary():
    first_violation_index = getattr(visualization, "first_violation_index", None)

    assert callable(first_violation_index)
    assert first_violation_index(
        [0.0, 0.0, 0.0, 0.4, 0.4],
        tolerance=0.05,
    ) == 3
    assert first_violation_index([0.0, 0.05], tolerance=0.05) is None


def test_generic_loader_dispatches_ros2_replay(tmp_path: Path):
    root = _write_check(tmp_path)

    model = load_visualization(root)

    assert isinstance(model, CheckVisualization)


def test_generic_loader_rejects_unsupported_runner(tmp_path: Path):
    root = _write_check(tmp_path)
    payload = _result_payload(root)
    payload["runner"] = "libero"
    _write_result_payload(root, payload)

    with pytest.raises(ValueError, match="unsupported visualization runner"):
        load_visualization(root)


def test_loader_honors_explicit_case_and_repeat(tmp_path: Path):
    root = _write_check(tmp_path)

    model = load_check_visualization(
        root,
        case_id="zero_command_drift",
        unit_id="repeat-000",
    )

    assert model.case_id == "zero_command_drift"
    assert model.unit_id == "repeat-000"


@pytest.mark.parametrize(
    ("case_id", "unit_id", "message"),
    [
        ("missing", None, "case"),
        ("zero_command_drift", "repeat-999", "repeat"),
    ],
)
def test_loader_rejects_unknown_selection(
    tmp_path: Path,
    case_id: str,
    unit_id: str | None,
    message: str,
):
    root = _write_check(tmp_path)

    with pytest.raises(ValueError, match=message):
        load_check_visualization(root, case_id=case_id, unit_id=unit_id)


def test_loader_rejects_missing_trace(tmp_path: Path):
    root = _write_check(tmp_path)
    payload = _result_payload(root)
    artifacts = payload["comparisons"][1]["evidence_pairs"][0]["candidate_artifacts"]
    payload["comparisons"][1]["evidence_pairs"][0]["candidate_artifacts"] = [
        artifact for artifact in artifacts if artifact["kind"] != "trace"
    ]
    _write_result_payload(root, payload)

    with pytest.raises(ValueError, match="trace"):
        load_check_visualization(root)


def test_loader_rejects_duplicate_trace(tmp_path: Path):
    root = _write_check(tmp_path)
    payload = _result_payload(root)
    artifacts = payload["comparisons"][1]["evidence_pairs"][0]["candidate_artifacts"]
    trace = next(artifact for artifact in artifacts if artifact["kind"] == "trace")
    artifacts.append(dict(trace))
    _write_result_payload(root, payload)

    with pytest.raises(ValueError, match="exactly one"):
        load_check_visualization(root)


def test_loader_rejects_missing_mcap(tmp_path: Path):
    root = _write_check(tmp_path)
    next(root.rglob("baseline/**/capture_0.mcap")).unlink()

    with pytest.raises(ValueError, match="MCAP|mcap"):
        load_check_visualization(root)


def test_loader_rejects_mismatched_expected_signal(tmp_path: Path):
    root = _write_check(tmp_path)
    candidate_trace = next(root.rglob("candidate/**/trace.json"))
    _write_trace(candidate_trace, expected=[0.1, 0.1], observed=[0.0, 0.0])

    with pytest.raises(ValueError, match="expected"):
        load_check_visualization(root)


def test_loader_rejects_absolute_error_that_does_not_match_samples(tmp_path: Path):
    root = _write_check(tmp_path)
    candidate_trace = next(root.rglob("candidate/**/trace.json"))
    payload = json.loads(candidate_trace.read_text())
    payload["series"]["absolute_error"]["values"] = [0.2, 0.2]
    candidate_trace.write_text(json.dumps(payload, indent=2))

    with pytest.raises(ValueError, match="absolute_error"):
        load_check_visualization(root)


def test_loader_rejects_aggregate_metric_that_contradicts_traces(tmp_path: Path):
    root = _write_check(tmp_path)
    payload = _result_payload(root)
    payload["comparisons"][1]["candidate_value"] = False
    _write_result_payload(root, payload)

    with pytest.raises(ValueError, match="automatic metric"):
        load_check_visualization(root)


def test_loader_rejects_escaping_artifact_path(tmp_path: Path):
    root = _write_check(tmp_path)
    payload = _result_payload(root)
    artifacts = payload["comparisons"][1]["evidence_pairs"][0]["candidate_artifacts"]
    next(artifact for artifact in artifacts if artifact["kind"] == "trace")["path"] = (
        "../outside.json"
    )
    _write_result_payload(root, payload)

    with pytest.raises(ValueError, match="traversal|confined|relative"):
        load_check_visualization(root)
