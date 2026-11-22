"""Real ROS 2 acceptance test for the fixed/regressed Twist replay."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("ros2") is None, reason="ROS 2 runtime is unavailable"),
]


def _run_check(suite: Path, runs_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "policyci.cli",
            "check",
            "--suite",
            str(suite),
            "--baseline",
            str(ROOT / "subjects/twist-fixed.yaml"),
            "--candidate",
            str(ROOT / "subjects/twist-stop-regression.yaml"),
            "--runs-dir",
            str(runs_dir),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=180,
    )


def test_fixed_to_regressed_replay_and_induced_soundness_fault(tmp_path: Path):
    completed = _run_check(ROOT / "suites/ros2-twist-replay.yaml", tmp_path / "runs")
    assert completed.returncode == 1, completed.stdout + completed.stderr

    check_root = next((tmp_path / "runs").glob("check_*"))
    result = json.loads((check_root / "result.json").read_text())
    assert result["gate"]["verdict"] == "block"
    assert result["comparisons"][0]["status"] == "regressed"
    assert result["comparisons"][0]["baseline_value"] is True
    assert result["comparisons"][0]["candidate_value"] is False
    evidence_pairs = result["comparisons"][0]["evidence_pairs"]
    assert len(evidence_pairs) == 3
    for pair in evidence_pairs:
        assert {artifact["kind"] for artifact in pair["baseline_artifacts"]} >= {
            "input_mcap",
            "mcap",
            "trace",
        }
        assert {artifact["kind"] for artifact in pair["candidate_artifacts"]} >= {
            "input_mcap",
            "mcap",
            "trace",
        }
        for side in ("baseline_artifacts", "candidate_artifacts"):
            for artifact in pair[side]:
                artifact_path = (check_root / artifact["path"]).resolve()
                assert artifact_path.is_relative_to(check_root.resolve())
                assert artifact_path.is_file()
                assert artifact_path.stat().st_size > 0
    baseline_captures = list(
        (check_root / "baseline/cases/stop_command_regression/repeats").rglob(
            "capture/*.mcap"
        )
    )
    candidate_captures = list(
        (check_root / "candidate/cases/stop_command_regression/repeats").rglob(
            "capture/*.mcap"
        )
    )
    assert baseline_captures and candidate_captures
    input_bag = ROOT / "examples/ros2_replay/fixtures/stop_drive_stop.mcap"
    assert baseline_captures[0].read_bytes() != input_bag.read_bytes()
    assert candidate_captures[0].read_bytes() != input_bag.read_bytes()
    portable_inputs = list(
        (check_root / "baseline/cases/stop_command_regression/repeats").glob(
            "repeat-*/input.mcap"
        )
    )
    assert len(portable_inputs) == 3
    assert all(path.read_bytes() == input_bag.read_bytes() for path in portable_inputs)

    bad_suite_data = yaml.safe_load((ROOT / "suites/ros2-twist-replay.yaml").read_text())
    bad_suite_data["cases"][0]["bag"] = str(input_bag)
    bad_suite_data["cases"][0]["capture_topics"] = ["/missing/output"]
    bad_suite = tmp_path / "missing-output.yaml"
    bad_suite.write_text(yaml.safe_dump(bad_suite_data, sort_keys=False))
    incomplete = _run_check(bad_suite, tmp_path / "fault-runs")
    assert incomplete.returncode == 2, incomplete.stdout + incomplete.stderr
    fault_root = next((tmp_path / "fault-runs").glob("check_*"))
    fault_result = json.loads((fault_root / "result.json").read_text())
    assert fault_result["gate"]["verdict"] == "no_verdict"
