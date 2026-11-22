"""Flat suite YAML (tests:) + axis normalization."""

from __future__ import annotations

from pathlib import Path

import pytest

from policyci.suite import (
    normalize_axis,
    load_suite_spec,
    Scenario,
)


ROOT = Path(__file__).resolve().parents[1]


def test_normalize_axis_aliases():
    assert normalize_axis("instruction_rephrase") == "instruction"
    assert normalize_axis("object_pose") == "pose"
    assert normalize_axis("instruction") == "instruction"
    assert normalize_axis("pose") == "pose"
    assert normalize_axis("nominal") == "nominal"
    assert normalize_axis(None) == "nominal"
    assert normalize_axis("") == "nominal"


def test_flat_suite_expands_one_scenario_per_test(tmp_path: Path):
    path = tmp_path / "flat.yaml"
    path.write_text(
        """
description: smoke
env: libero_object
episodes: 2
seed: 1000
gate:
  threshold_pp: 12
  min_episodes: 3
tests:
  - task: 0
  - task: 0
    instruction: "grab the soup can and put it into the basket"
  - task: 0
    pose_cm: 4.0
    level: 2
"""
    )
    spec = load_suite_spec(path)
    assert spec.gate.threshold_pp == 12.0
    assert spec.gate.min_episodes == 3
    scenarios = spec.expand()
    assert len(scenarios) == 3

    nom, instr, pose = scenarios
    assert nom.axis == "nominal" and nom.level == 0
    assert nom.n_episodes == 2 and nom.seed == 1000

    assert instr.axis == "instruction"
    assert instr.level == 1
    assert "soup can" in (instr.instruction or "")

    assert pose.axis == "pose"
    assert pose.level == 2
    assert pose.params["magnitude_cm"] == 4.0

    shard = instr.to_shard("pol/x")
    assert shard["perturbation"]["axis"] == "instruction"
    assert shard["perturbation"]["instruction"]


def test_flat_rejects_instruction_and_pose_together(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
env: libero_object
tests:
  - task: 0
    instruction: "hi"
    pose_cm: 2.0
"""
    )
    with pytest.raises(ValueError, match="instruction.*pose|pose.*instruction"):
        load_suite_spec(path).expand()


def test_flat_rejects_unknown_keys(tmp_path: Path):
    path = tmp_path / "unk.yaml"
    path.write_text(
        """
env: libero_object
tests:
  - task: 0
    lighting: dim
"""
    )
    with pytest.raises(ValueError, match="Unknown|lighting"):
        load_suite_spec(path).expand()


def test_legacy_axes_suite_still_loads():
    spec = load_suite_spec(ROOT / "suites" / "libero_object_robustness.yaml")
    scenarios = spec.expand()
    assert len(scenarios) == 2 + 6 + 6
    # Legacy expand now emits canonical axis names
    axes = {s.axis for s in scenarios}
    assert "instruction" in axes or "instruction_rephrase" in axes
    # Prefer canonical after normalize in expand
    assert all(normalize_axis(s.axis) in {"nominal", "instruction", "pose"} for s in scenarios)
