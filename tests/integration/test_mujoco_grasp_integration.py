"""Real OSMesa Robosuite acceptance for the manipulation hero demo."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from policyci.check import execute_check
from policyci.checkspec import load_check_suite, load_subject
from policyci.mujoco_metrics import derive_placement_metrics
from policyci.runner.mujoco import MujocoRunner
from policyci.schemas import GateVerdict, SimulationTrace, TaskStatus


def _simulation_runtime_available() -> bool:
    return all(
        importlib.util.find_spec(module) is not None
        for module in ("imageio", "mujoco", "numpy", "robosuite")
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not _simulation_runtime_available(),
    reason="requires Robosuite, MuJoCo, ImageIO, and NumPy",
)
def test_real_panda_offset_regression_is_fixed(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    suite = load_check_suite(repo / "suites/mujoco-grasp-regression.yaml")
    baseline = load_subject(repo / "subjects/grasp-offset-bad.yaml")
    candidate = load_subject(repo / "subjects/grasp-offset-fixed.yaml")

    result, check_root = execute_check(
        suite,
        baseline,
        candidate,
        MujocoRunner(suite),
        runs_root=tmp_path / "runs",
    )

    assert result.gate.verdict is GateVerdict.PASS
    assert len(result.comparisons) == 1
    comparison = result.comparisons[0]
    assert comparison.status is TaskStatus.FIXED
    assert comparison.baseline_value is False
    assert comparison.candidate_value is True
    assert len(comparison.evidence_pairs) == 1
    pair = comparison.evidence_pairs[0]
    assert pair.unit_id == "1000"

    traces = {}
    metrics = {}
    ffprobe = shutil.which("ffprobe")
    assert ffprobe is not None
    for side, artifacts in (
        ("baseline", pair.baseline_artifacts),
        ("candidate", pair.candidate_artifacts),
    ):
        assert sorted(artifact.kind for artifact in artifacts) == [
            "trajectory",
            "video",
        ]
        paths = {}
        for artifact in artifacts:
            path = (check_root / artifact.path).resolve()
            assert path.is_relative_to(check_root.resolve())
            assert path.stat().st_size > 0
            paths[artifact.kind] = path
        trace = SimulationTrace.model_validate_json(paths["trajectory"].read_text())
        assert all(sample.has_sensor_evidence for sample in trace.samples)
        traces[side] = trace
        metrics[side] = derive_placement_metrics(
            trace,
            hold_duration_s=0.5,
        )
        decoded = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames,width,height",
                "-of",
                "json",
                str(paths["video"]),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert decoded.returncode == 0, decoded.stderr
        stream = json.loads(decoded.stdout)["streams"][0]
        assert int(stream["nb_read_frames"]) == 240
        assert (stream["width"], stream["height"]) == (1280, 720)
        assert trace.duration_s <= 12.0

    assert traces["baseline"].seed == traces["candidate"].seed == 1000
    assert traces["baseline"].initial_state_sha256 == traces["candidate"].initial_state_sha256
    assert traces["baseline"].model_sha256 == traces["candidate"].model_sha256
    assert [sample.time_s for sample in traces["baseline"].samples] == [
        sample.time_s for sample in traces["candidate"].samples
    ]

    assert metrics["baseline"]["placement_success"] is False
    assert metrics["baseline"]["final_object_target_xy_distance_m"] >= 0.15
    assert metrics["candidate"]["placement_success"] is True
    assert metrics["candidate"]["min_object_target_xy_distance_m"] <= 0.08
    assert metrics["candidate"]["max_object_lift_m"] >= 0.12
    assert (
        max(sample.left_finger_normal_force_n or 0.0 for sample in traces["candidate"].samples)
        > 0.0
    )
    assert (
        max(sample.right_finger_normal_force_n or 0.0 for sample in traces["candidate"].samples)
        > 0.0
    )
    assert (
        max(
            sum(component * component for component in sample.wrist_force_xyz_n or ()) ** 0.5
            for sample in traces["candidate"].samples
        )
        > 0.0
    )
