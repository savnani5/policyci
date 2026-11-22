"""Real Robosuite acceptance for the Actuate pose and perception gate."""

from __future__ import annotations

import importlib.util
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from policyci.checkspec import load_check_suite, load_subject
from policyci.robustness_check import execute_robustness_check
from policyci.robustness_rerun import render_robustness_rerun
from policyci.robustness_visualization import load_robustness_visualization
from policyci.runner.mujoco import MujocoRunner
from policyci.schemas import GateVerdict, SimulationTrace


def _simulation_runtime_available() -> bool:
    return all(
        importlib.util.find_spec(module) is not None
        for module in ("imageio", "mujoco", "numpy", "robosuite", "rerun")
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not _simulation_runtime_available(),
    reason="requires Robosuite, MuJoCo, ImageIO, NumPy, and Rerun",
)
def test_real_object_aware_controller_is_blocked_by_perception_bias(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    suite = load_check_suite(repo / "suites/mujoco-pick-place-robustness.yaml")
    subject = load_subject(repo / "subjects/grasp-offset-fixed.yaml")

    result, check_root = execute_robustness_check(
        suite,
        subject,
        MujocoRunner(suite),
        runs_root=tmp_path / "runs",
    )

    assert result.gate.verdict is GateVerdict.BLOCK
    assert result.gate.exit_code == 1
    assert [case.case_id for case in result.cases] == [
        "nominal_center",
        "object_right_5cm",
        "perception_bias_5cm",
    ]
    assert result.cases[0].passed is True
    assert result.cases[1].passed is True
    assert result.cases[2].passed is False
    assert result.cases[2].object_observation_bias_m == (0.05, 0.0, 0.0)
    assert result.cases[2].metrics["max_object_lift_m"] < 0.001
    assert result.cases[2].metrics["bilateral_contact_duration_s"] == 0.0
    assert result.cases[2].metrics["final_object_target_xy_distance_m"] > 0.5
    assert len({case.environment["controller_sha256"] for case in result.cases}) == 1
    assert len({case.environment["model_sha256"] for case in result.cases}) == 1

    traces = {}
    for case in result.cases:
        assert case.validity.value == "valid"
        assert sorted(artifact.kind for artifact in case.artifacts) == [
            "trajectory",
            "video",
        ]
        paths = {
            artifact.kind: (check_root / artifact.path).resolve()
            for artifact in case.artifacts
        }
        assert all(path.is_relative_to(check_root.resolve()) for path in paths.values())
        assert all(path.stat().st_size > 0 for path in paths.values())
        trace = SimulationTrace.model_validate_json(paths["trajectory"].read_text())
        assert trace.sample_count == 240
        assert trace.duration_s <= 12.0
        traces[case.case_id] = trace

    nominal = traces["nominal_center"].samples[0].object_position_m
    for case in result.cases:
        observed = traces[case.case_id].samples[0].object_position_m
        declared = case.object_position_offset_m
        assert all(
            math.isclose(
                observed[axis] - nominal[axis],
                declared[axis],
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            for axis in range(3)
        )

    model = load_robustness_visualization(check_root)
    assert model.perturbed_case_id == "perception_bias_5cm"
    output = render_robustness_rerun(
        model,
        check_root / "visualizations/robustness/actuate.rrd",
    )
    rerun = shutil.which("rerun")
    assert rerun is not None
    verified = subprocess.run(
        [rerun, "rrd", "verify", str(output)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert verified.returncode == 0, verified.stderr or verified.stdout
