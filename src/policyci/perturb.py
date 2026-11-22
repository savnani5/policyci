"""Perturbation engine — instruction rephrase + object-pose axes.

Sits above the Gymnasium env interface. Wrappers are applied at env creation
time; the CI layer never touches sim internals.
"""

from __future__ import annotations

from typing import Any

# Leveled instruction rewrites. Keys are (suite, task_id) → {level: text}.
# Level 1 = synonym swap, Level 2 = paraphrase, Level 3 = full rewrite.
DEFAULT_REPHRASES: dict[tuple[str, int], dict[int, str]] = {
    ("libero_object", 0): {
        1: "pick up the alphabet soup and place it in the basket",
        2: "grab the soup can and put it into the basket",
        3: "move the cylindrical soup container from the table into the woven basket",
    },
}


def rephrase_instruction(
    suite: str,
    task_id: int,
    level: int,
    original: str = "",
    overrides: dict[int, str] | None = None,
) -> str:
    """Return a leveled rephrase of the task instruction."""
    if level <= 0:
        return original
    if overrides and level in overrides:
        return overrides[level]
    table = DEFAULT_REPHRASES.get((suite, task_id), {})
    if level in table:
        return table[level]
    # Fallback: lightly mutate the original
    if not original:
        return f"[rephrase-L{level}]"
    if level == 1:
        return original.replace("pick up", "grab").replace("place", "put")
    if level == 2:
        return f"Please {original.rstrip('.')}."
    return f"Your goal: {original.rstrip('.')}, carefully."


def pose_jitter_params(level: int, magnitude_cm: float = 2.0) -> dict[str, Any]:
    """Object-pose axis params: cm-scale XY jitter at reset, scaled by level."""
    if level <= 0:
        return {"magnitude_cm": 0.0, "axes": ["x", "y"]}
    return {
        "magnitude_cm": float(magnitude_cm) * level,
        "axes": ["x", "y"],
        "seed_offset": level * 17,
    }


def build_perturbation(
    axis: str | None,
    level: int,
    *,
    suite: str = "libero_object",
    task_id: int = 0,
    original_instruction: str = "",
    magnitude_cm: float = 2.0,
    instruction_overrides: dict[int, str] | None = None,
) -> dict[str, Any] | None:
    """Build a perturbation dict for a scenario (None if nominal)."""
    from policyci.suite import normalize_axis

    if not axis or level <= 0:
        return None
    canon = normalize_axis(axis)
    if canon == "nominal":
        return None
    if canon == "instruction":
        instr = rephrase_instruction(
            suite, task_id, level, original_instruction, instruction_overrides
        )
        return {
            "axis": "instruction",
            "level": level,
            "instruction": instr,
            "params": {"original": original_instruction},
        }
    if canon == "pose":
        return {
            "axis": "pose",
            "level": level,
            "params": pose_jitter_params(level, magnitude_cm),
        }
    raise ValueError(f"Unknown perturbation axis: {axis}")


class InstructionPerturbationWrapper:
    """Gymnasium wrapper that overrides the language instruction after reset.

    Applied above LeRobot's env — does not require LIBERO internals.
    Compatible with any env that exposes `task` / `language_instruction` in info
    or as an attribute.
    """

    def __init__(self, env: Any, instruction: str):
        self.env = env
        self.instruction = instruction

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info or {})
        info["language_instruction"] = self.instruction
        info["task"] = self.instruction
        # Common attribute names used by LIBERO / LeRobot processors
        for attr in ("language_instruction", "task", "task_description"):
            if hasattr(self.env, attr):
                try:
                    setattr(self.env, attr, self.instruction)
                except Exception:
                    pass
        return obs, info

    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()


class ObjectPosePerturbationWrapper:
    """Gymnasium wrapper that jitters object XY poses after reset.

    Best-effort: looks for common robosuite/LIBERO init-state hooks. If the env
    does not expose a mutable object pose API, the wrapper records the intended
    jitter in `info` so the run is still self-describing (and CI can flag that
    the axis was requested).
    """

    def __init__(self, env: Any, magnitude_cm: float = 2.0, seed_offset: int = 0):
        self.env = env
        self.magnitude_m = magnitude_cm / 100.0
        self.seed_offset = seed_offset

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, **kwargs):
        import random

        obs, info = self.env.reset(**kwargs)
        info = dict(info or {})
        rng = random.Random(self.seed_offset)
        dx = rng.uniform(-self.magnitude_m, self.magnitude_m)
        dy = rng.uniform(-self.magnitude_m, self.magnitude_m)
        applied = False

        # Try LIBERO / robosuite-style object pose mutation (numpy optional)
        sim = getattr(self.env, "sim", None) or getattr(getattr(self.env, "env", None), "sim", None)
        if sim is not None and hasattr(sim, "data"):
            try:
                qpos = sim.data.qpos
                if len(qpos) >= 2:
                    qpos[0] = float(qpos[0]) + dx
                    qpos[1] = float(qpos[1]) + dy
                    if hasattr(sim, "forward"):
                        sim.forward()
                    applied = True
            except Exception:
                applied = False

        info["pose_jitter"] = {
            "dx_m": float(dx),
            "dy_m": float(dy),
            "magnitude_m": self.magnitude_m,
            "applied": applied,
        }
        return obs, info

    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()


def wrap_env(env: Any, perturbation: dict[str, Any] | None) -> Any:
    """Apply the appropriate wrapper(s) for a perturbation dict."""
    if not perturbation:
        return env
    from policyci.suite import normalize_axis

    axis = normalize_axis(perturbation.get("axis"))
    if axis == "instruction" and perturbation.get("instruction"):
        return InstructionPerturbationWrapper(env, perturbation["instruction"])
    if axis == "pose":
        params = perturbation.get("params") or {}
        return ObjectPosePerturbationWrapper(
            env,
            magnitude_cm=float(params.get("magnitude_cm", 2.0)),
            seed_offset=int(params.get("seed_offset", 0)),
        )
    return env
