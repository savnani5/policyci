"""Small, dependency-lazy helpers for exact MuJoCo sensor evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def sensor_vector(
    sim,
    *,
    name: str,
    expected_dimension: int,
) -> tuple[float, ...]:
    """Read one exact compiled MuJoCo sensor by name."""

    if name not in sim.model.sensor_names:
        raise ValueError(f"MuJoCo sensor {name!r} is not present")
    sensor_id = int(sim.model.sensor_name2id(name))
    dimension = int(sim.model.sensor_dim[sensor_id])
    if dimension != expected_dimension:
        raise ValueError(
            f"MuJoCo sensor {name!r} has dimension {dimension}, expected {expected_dimension}"
        )
    start = int(sim.model.sensor_adr[sensor_id])
    return tuple(float(value) for value in sim.data.sensordata[start : start + dimension])


def validate_sensor_evidence_sample(
    sample: Mapping[str, object],
) -> tuple[float, float, float, float, tuple[float, float, float]]:
    """Validate and normalize the raw sensor fields used by demo acceptance."""

    def scalar(name: str, *, minimum: float, maximum: float | None = None) -> float:
        value = sample.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < minimum:
            raise ValueError(f"{name} must be finite and >= {minimum}")
        if maximum is not None and normalized > maximum:
            raise ValueError(f"{name} must be <= {maximum}")
        return normalized

    command = scalar("gripper_command", minimum=-1.0, maximum=1.0)
    width = scalar("gripper_width_m", minimum=0.0)
    left_normal_force = scalar("left_finger_normal_force_n", minimum=0.0)
    right_normal_force = scalar("right_finger_normal_force_n", minimum=0.0)
    raw_wrist_force = sample.get("wrist_force_xyz_n")
    if (
        not isinstance(raw_wrist_force, Sequence)
        or isinstance(raw_wrist_force, (str, bytes))
        or len(raw_wrist_force) != 3
    ):
        raise ValueError("wrist_force_xyz_n must contain exactly three finite numbers")
    wrist_force = tuple(
        float(value)
        for value in raw_wrist_force
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    if len(wrist_force) != 3 or any(not math.isfinite(value) for value in wrist_force):
        raise ValueError("wrist_force_xyz_n must contain exactly three finite numbers")
    return command, width, left_normal_force, right_normal_force, wrist_force


def gripper_closing_delta_m(
    *,
    commands: Sequence[float],
    widths_m: Sequence[float],
) -> float:
    """Return open-command width minus close-command width."""

    if len(commands) != len(widths_m) or not commands:
        raise ValueError("gripper commands and widths must be non-empty and aligned")
    open_widths = [width for command, width in zip(commands, widths_m) if command < 0.0]
    closed_widths = [width for command, width in zip(commands, widths_m) if command > 0.0]
    if not open_widths or not closed_widths:
        raise ValueError("gripper evidence requires both open and close commands")
    return float(max(open_widths) - min(closed_widths))


def finger_normal_force_n(
    mujoco_module,
    sim,
    *,
    finger_geom_id: int,
    object_geom_ids: set[int],
) -> float:
    """Sum contact-frame normal force between one finger pad and the object."""

    import numpy as np

    total_force_n = 0.0
    for contact_index in range(int(sim.data.ncon)):
        contact = sim.data.contact[contact_index]
        geom_pair = {int(contact.geom1), int(contact.geom2)}
        if finger_geom_id not in geom_pair or not geom_pair.intersection(object_geom_ids):
            continue
        force_torque = np.zeros(6, dtype=np.float64)
        mujoco_module.mj_contactForce(
            sim.model._model,
            sim.data._data,
            contact_index,
            force_torque,
        )
        total_force_n += max(0.0, float(force_torque[0]))
    return total_force_n
