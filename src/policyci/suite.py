"""YAML suite specs → expanded scenario list.

Supports:
- Flat `tests:` format (preferred) — one scenario per test entry
- Legacy `axes:` × levels × seeds matrix (still loads)

Canonical perturbation axes: `instruction`, `pose`, `nominal`.
Legacy aliases `instruction_rephrase` / `object_pose` accepted on read.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Canonical names → legacy names still seen in old YAML / manifests
_AXIS_ALIASES: dict[str, str] = {
    "instruction_rephrase": "instruction",
    "object_pose": "pose",
    "instruction": "instruction",
    "pose": "pose",
    "nominal": "nominal",
}

ALLOWED_TEST_KEYS = frozenset(
    {"task", "episodes", "seed", "instruction", "pose_cm", "axis", "level", "env"}
)


def normalize_axis(axis: str | None) -> str:
    """Map legacy axis names to canonical `instruction` | `pose` | `nominal`."""
    if axis is None or axis == "":
        return "nominal"
    key = str(axis).strip()
    if key not in _AXIS_ALIASES:
        raise ValueError(
            f"Unknown perturbation axis {axis!r}. "
            f"Known: {sorted(set(_AXIS_ALIASES.values()))} "
            f"(aliases: instruction_rephrase, object_pose)"
        )
    return _AXIS_ALIASES[key]


@dataclass
class GateConfig:
    threshold_pp: float = 10.0
    min_episodes: int = 5


@dataclass
class Scenario:
    suite: str
    task_id: int
    seed: int
    n_episodes: int = 1
    axis: str | None = None  # None / "nominal" = no perturbation
    level: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    instruction: str | None = None

    def to_shard(self, policy_path: str, run_prefix: str | None = None) -> dict[str, Any]:
        del run_prefix  # reserved
        axis = normalize_axis(self.axis)
        perturbation = None
        if axis != "nominal" and self.level > 0:
            perturbation = {
                "axis": axis,
                "level": self.level,
                "params": self.params,
                "instruction": self.instruction,
            }
        return {
            "policy_path": policy_path,
            "suite": self.suite,
            "task_ids": [self.task_id],
            "n_episodes": self.n_episodes,
            "seed": self.seed,
            "perturbation": perturbation,
        }


@dataclass
class SuiteSpec:
    name: str
    suite: str  # libero_object, libero_spatial, ...
    task_ids: list[int] = field(default_factory=lambda: [0])
    seeds: list[int] = field(default_factory=lambda: [1000])
    n_episodes: int = 1
    axes: list[dict[str, Any]] = field(default_factory=list)
    # Flat format
    tests: list[dict[str, Any]] | None = None
    gate: GateConfig = field(default_factory=GateConfig)
    description: str = ""
    seed: int = 1000  # default seed for flat tests
    format: str = "legacy"  # "flat" | "legacy"

    def expand(self) -> list[Scenario]:
        if self.format == "flat" or self.tests is not None:
            return self._expand_flat()
        return self._expand_legacy()

    def _expand_flat(self) -> list[Scenario]:
        if not self.tests:
            raise ValueError("Flat suite requires a non-empty tests: list")
        scenarios: list[Scenario] = []
        for i, raw in enumerate(self.tests):
            if not isinstance(raw, dict):
                raise ValueError(f"tests[{i}] must be a mapping, got {type(raw).__name__}")
            unknown = set(raw.keys()) - ALLOWED_TEST_KEYS
            if unknown:
                raise ValueError(
                    f"tests[{i}] unknown keys {sorted(unknown)}. "
                    f"Allowed: {sorted(ALLOWED_TEST_KEYS)}"
                )
            if "task" not in raw:
                raise ValueError(f"tests[{i}] missing required field 'task'")

            has_instr = "instruction" in raw and raw["instruction"] is not None
            has_pose = "pose_cm" in raw and raw["pose_cm"] is not None
            if has_instr and has_pose:
                raise ValueError(
                    f"tests[{i}] cannot set both instruction and pose_cm "
                    "(write two separate tests)"
                )

            task_id = int(raw["task"])
            n_episodes = int(raw.get("episodes", self.n_episodes))
            seed = int(raw.get("seed", self.seed))
            env = str(raw.get("env", self.suite))

            if has_instr:
                level = int(raw.get("level", 1))
                axis_raw = raw.get("axis", "instruction")
                axis = normalize_axis(axis_raw)
                if axis not in ("instruction", "nominal"):
                    raise ValueError(
                        f"tests[{i}] instruction field requires axis=instruction, got {axis}"
                    )
                scenarios.append(
                    Scenario(
                        suite=env,
                        task_id=task_id,
                        seed=seed,
                        n_episodes=n_episodes,
                        axis="instruction",
                        level=level,
                        instruction=str(raw["instruction"]),
                        params={},
                    )
                )
            elif has_pose:
                level = int(raw.get("level", 1))
                axis_raw = raw.get("axis", "pose")
                axis = normalize_axis(axis_raw)
                if axis not in ("pose", "nominal"):
                    raise ValueError(
                        f"tests[{i}] pose_cm field requires axis=pose, got {axis}"
                    )
                magnitude = float(raw["pose_cm"])
                scenarios.append(
                    Scenario(
                        suite=env,
                        task_id=task_id,
                        seed=seed,
                        n_episodes=n_episodes,
                        axis="pose",
                        level=level,
                        params={"magnitude_cm": magnitude, "axes": ["x", "y"]},
                    )
                )
            else:
                # Nominal — optional explicit axis/level ignored unless axis set oddly
                axis_raw = raw.get("axis")
                if axis_raw is not None and normalize_axis(axis_raw) != "nominal":
                    raise ValueError(
                        f"tests[{i}] has axis={axis_raw!r} but no instruction/pose_cm"
                    )
                scenarios.append(
                    Scenario(
                        suite=env,
                        task_id=task_id,
                        seed=seed,
                        n_episodes=n_episodes,
                        axis="nominal",
                        level=0,
                    )
                )
        return scenarios

    def _expand_legacy(self) -> list[Scenario]:
        scenarios: list[Scenario] = []
        # Always include nominal (level 0)
        for task_id in self.task_ids:
            for seed in self.seeds:
                scenarios.append(
                    Scenario(
                        suite=self.suite,
                        task_id=task_id,
                        seed=seed,
                        n_episodes=self.n_episodes,
                        axis="nominal",
                        level=0,
                    )
                )

        for axis_cfg in self.axes:
            axis = normalize_axis(axis_cfg["name"])
            levels = axis_cfg.get("levels", [1])
            params_by_level = axis_cfg.get("params_by_level", {})
            instructions_by_task = axis_cfg.get("instructions_by_task", {})
            for task_id in self.task_ids:
                for level in levels:
                    params = dict(
                        params_by_level.get(level) or params_by_level.get(str(level)) or {}
                    )
                    # Scale pose jitter by level if magnitude given at top level
                    if "magnitude_cm" in axis_cfg and axis == "pose":
                        params.setdefault("magnitude_cm", float(axis_cfg["magnitude_cm"]) * level)
                    instr = None
                    task_instrs = instructions_by_task.get(task_id) or instructions_by_task.get(
                        str(task_id)
                    )
                    if isinstance(task_instrs, dict):
                        instr = task_instrs.get(level) or task_instrs.get(str(level))
                    elif isinstance(task_instrs, list) and level - 1 < len(task_instrs):
                        instr = task_instrs[level - 1]
                    for seed in self.seeds:
                        scenarios.append(
                            Scenario(
                                suite=self.suite,
                                task_id=task_id,
                                seed=seed,
                                n_episodes=self.n_episodes,
                                axis=axis,
                                level=int(level),
                                params=params,
                                instruction=instr,
                            )
                        )
        return scenarios


def load_suite_spec(path: Path | str) -> SuiteSpec:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Suite YAML must be a mapping: {path}")

    gate_raw = data.get("gate") or {}
    gate = GateConfig(
        threshold_pp=float(gate_raw.get("threshold_pp", 10.0)),
        min_episodes=int(gate_raw.get("min_episodes", 5)),
    )

    name = str(data.get("name") or data.get("description") or Path(path).stem)
    description = str(data.get("description") or "")
    suite = str(data.get("env") or data.get("suite") or data.get("env_task") or "libero_object")
    n_episodes = int(data.get("episodes") or data.get("n_episodes") or 1)
    seed = int(data.get("seed") or (data.get("seeds") or [1000])[0])

    if "tests" in data:
        tests = data["tests"]
        if not isinstance(tests, list):
            raise ValueError("tests: must be a list")
        return SuiteSpec(
            name=name,
            suite=suite,
            n_episodes=n_episodes,
            seed=seed,
            tests=tests,
            gate=gate,
            description=description,
            format="flat",
            task_ids=[],
            seeds=[],
            axes=[],
        )

    # Legacy matrix format
    warnings.warn(
        f"{path}: legacy axes:/task_ids: suite format; prefer flat tests: "
        "(see the flat suite example in README.md and suites/soup_basket.yaml)",
        DeprecationWarning,
        stacklevel=2,
    )
    return SuiteSpec(
        name=name,
        suite=suite,
        task_ids=list(data.get("task_ids", [0])),
        seeds=list(data.get("seeds", [seed])),
        n_episodes=n_episodes,
        seed=seed,
        axes=list(data.get("axes", [])),
        gate=gate,
        description=description,
        format="legacy",
        tests=None,
    )


def nominal_suite(
    suite: str = "libero_object",
    task_ids: list[int] | None = None,
    seeds: list[int] | None = None,
    n_episodes: int = 10,
    name: str = "nominal",
) -> SuiteSpec:
    """Zero-perturbation suite — what v0 runs use."""
    return SuiteSpec(
        name=name,
        suite=suite,
        task_ids=task_ids or [0],
        seeds=seeds or [1000],
        n_episodes=n_episodes,
        axes=[],
        format="legacy",
    )


DEFAULT_ROBUSTNESS_SUITE_YAML = """
name: libero_object_robustness
suite: libero_object
task_ids: [0]
seeds: [1000, 1001]
n_episodes: 2
axes:
  - name: instruction_rephrase
    levels: [1, 2, 3]
    instructions_by_task:
      0:
        1: "pick up the alphabet soup and place it in the basket"
        2: "grab the soup can and put it into the basket"
        3: "move the cylindrical soup container from the table into the woven basket"
  - name: object_pose
    levels: [1, 2, 3]
    magnitude_cm: 2.0
"""
