"""Suite expansion + perturbation unit tests."""

from __future__ import annotations

from pathlib import Path

from policyci.perturb import (
    InstructionPerturbationWrapper,
    ObjectPosePerturbationWrapper,
    build_perturbation,
    pose_jitter_params,
    rephrase_instruction,
    wrap_env,
)
from policyci.suite import load_suite_spec, nominal_suite


ROOT = Path(__file__).resolve().parents[1]


def test_nominal_suite_is_zero_perturbation():
    spec = nominal_suite(suite="libero_object", task_ids=[0, 1], seeds=[1000], n_episodes=5)
    scenarios = spec.expand()
    assert len(scenarios) == 2
    assert all(s.axis == "nominal" and s.level == 0 for s in scenarios)
    assert all(s.n_episodes == 5 for s in scenarios)


def test_yaml_suite_expansion():
    spec = load_suite_spec(ROOT / "suites" / "libero_object_robustness.yaml")
    scenarios = spec.expand()
    # nominal: 1 task × 2 seeds = 2
    # instruction: 3 levels × 1 task × 2 seeds = 6
    # object_pose: 3 levels × 1 task × 2 seeds = 6
    assert len(scenarios) == 2 + 6 + 6
    axes = {s.axis for s in scenarios}
    assert axes == {"nominal", "instruction", "pose"}
    instr = [s for s in scenarios if s.axis == "instruction" and s.level == 3][0]
    assert "cylindrical soup" in (instr.instruction or "")
    pose = [s for s in scenarios if s.axis == "pose" and s.level == 2][0]
    assert pose.params["magnitude_cm"] == 4.0  # 2.0 * level 2


def test_rephrase_levels():
    # Known (suite, task) hits the DEFAULT_REPHRASES table
    assert "alphabet soup" in rephrase_instruction("libero_object", 0, 1, "ignored")
    assert rephrase_instruction("libero_object", 0, 0, "pick up the cup") == "pick up the cup"
    # Unknown task falls back to light mutation of original
    assert "grab" in rephrase_instruction("other_suite", 99, 1, "pick up the cup")
    custom = rephrase_instruction("x", 0, 2, "orig", overrides={2: "custom text"})
    assert custom == "custom text"


def test_pose_jitter_scales():
    assert pose_jitter_params(0)["magnitude_cm"] == 0.0
    assert pose_jitter_params(3, magnitude_cm=2.0)["magnitude_cm"] == 6.0


def test_build_perturbation_nominal_is_none():
    assert build_perturbation("nominal", 0) is None
    assert build_perturbation(None, 1) is None


class _FakeEnv:
    def __init__(self):
        self.language_instruction = "original"
        self.closed = False

    def reset(self, **kwargs):
        return {"obs": 1}, {"ok": True}

    def step(self, action):
        return {"obs": 2}, 0.0, False, False, {}

    def close(self):
        self.closed = True


def test_instruction_wrapper_overrides_info():
    env = InstructionPerturbationWrapper(_FakeEnv(), "new instruction")
    obs, info = env.reset()
    assert info["language_instruction"] == "new instruction"
    assert env.language_instruction == "new instruction"


def test_pose_wrapper_records_jitter():
    env = ObjectPosePerturbationWrapper(_FakeEnv(), magnitude_cm=3.0, seed_offset=1)
    obs, info = env.reset()
    assert "pose_jitter" in info
    assert abs(info["pose_jitter"]["dx_m"]) <= 0.03 + 1e-9


def test_wrap_env_dispatch():
    env = _FakeEnv()
    wrapped = wrap_env(env, {"axis": "instruction", "instruction": "hi", "level": 1})
    assert isinstance(wrapped, InstructionPerturbationWrapper)
    wrapped2 = wrap_env(env, {"axis": "pose", "level": 1, "params": {"magnitude_cm": 1.0}})
    assert isinstance(wrapped2, ObjectPosePerturbationWrapper)
    # Legacy aliases still work
    wrapped3 = wrap_env(
        env, {"axis": "instruction_rephrase", "instruction": "hi", "level": 1}
    )
    assert isinstance(wrapped3, InstructionPerturbationWrapper)
