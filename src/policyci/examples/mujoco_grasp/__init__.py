"""Deterministic Robosuite Panda Lift example."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from policyci.examples.mujoco_grasp.adapter import RobosuiteLiftAdapter

__all__ = ["RobosuiteLiftAdapter"]


def __getattr__(name: str):
    if name == "RobosuiteLiftAdapter":
        from policyci.examples.mujoco_grasp.adapter import RobosuiteLiftAdapter

        return RobosuiteLiftAdapter
    raise AttributeError(name)
