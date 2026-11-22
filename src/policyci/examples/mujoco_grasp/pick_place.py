"""Real Robosuite Panda pick-and-place rollout used by the conference demo."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from policyci.runner.mujoco import SimulationEpisode
from policyci.schemas import SimulationSample, SimulationTrace, SubjectRef

if TYPE_CHECKING:
    from policyci.checkspec import CaseSpec

CONTROL_HZ = 20
HORIZON = 240
CAMERA_NAME = "agentview"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FOVY_DEG = 34.0


def position_delta(
    *,
    actual_eef: Sequence[float],
    target: Sequence[float],
    action_frame_offset: Sequence[float],
    gain: float,
) -> tuple[float, float, float]:
    """Translate a target into clipped controller input with an injected frame fault."""

    return tuple(
        max(-1.0, min(1.0, gain * (goal - (actual + offset))))
        for actual, goal, offset in zip(actual_eef, target, action_frame_offset)
    )


def phase_action(
    translation: Sequence[float],
    *,
    gripper_command: float,
) -> tuple[float, float, float, float, float, float, float]:
    return (
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
        0.0,
        0.0,
        0.0,
        gripper_command,
    )


def offset_object_qpos(
    nominal_qpos: Sequence[float],
    offset_m: Sequence[float],
) -> tuple[float, ...]:
    """Apply an XYZ scenario offset without changing object orientation."""

    if len(nominal_qpos) < 3:
        raise ValueError("object joint position must contain XYZ")
    if len(offset_m) != 3:
        raise ValueError("object position offset must contain XYZ")
    shifted = [float(value) for value in nominal_qpos]
    for axis, offset in enumerate(offset_m):
        shifted[axis] += float(offset)
    return tuple(shifted)


def perceived_object_position(
    actual_position_m: Sequence[float],
    observation_bias_m: Sequence[float],
) -> tuple[float, ...]:
    """Apply a deterministic object-localization error to the controller input."""

    return tuple(
        float(actual) + float(bias)
        for actual, bias in zip(actual_position_m, observation_bias_m)
    )


def pick_place_phase_specs(
    *,
    object_position: Sequence[float],
    target_position: Sequence[float],
) -> tuple[tuple[str, int, tuple[float, float, float], float], ...]:
    """Build the fixed 12-second grasp, transport, place, and retreat story."""

    object_x, object_y, object_z = (float(value) for value in object_position)
    target_x, target_y, target_z = (float(value) for value in target_position)
    grasp = (object_x, object_y, object_z + 0.020)
    object_hover = (object_x, object_y, target_z + 0.28)
    target_hover = (target_x, target_y, target_z + 0.28)
    place = (target_x, target_y, target_z + 0.10)
    return (
        ("approach", 35, (object_x, object_y, object_z + 0.16), -1.0),
        ("align", 30, grasp, -1.0),
        ("grasp", 25, grasp, 1.0),
        ("lift", 30, object_hover, 1.0),
        ("transport", 50, target_hover, 1.0),
        ("place", 30, place, 1.0),
        ("release", 15, place, -1.0),
        ("retreat", 25, target_hover, -1.0),
    )


def _xyzw(wxyz: Sequence[float]) -> tuple[float, float, float, float]:
    return (float(wxyz[1]), float(wxyz[2]), float(wxyz[3]), float(wxyz[0]))


def _first_geom_name(value: str | Sequence[str]) -> str:
    if isinstance(value, str):
        return value
    if not value:
        raise ValueError("gripper finger-pad geometry is empty")
    return value[0]


def single_component(components: Mapping[str, object], *, name: str) -> object:
    if len(components) != 1:
        raise ValueError(f"Panda demo requires exactly one {name}")
    return next(iter(components.values()))


def _eef_site_id(robot) -> int:
    site_id = robot.eef_site_id
    if isinstance(site_id, dict):
        if len(site_id) != 1:
            raise ValueError("Panda demo requires exactly one active arm")
        return int(next(iter(site_id.values())))
    return int(site_id)


def model_identity_sha256(controller_config: Mapping[str, object]) -> str:
    """Hash the deterministic model inputs, excluding unstable XML serialization."""

    model_spec = {
        "task": "PickPlaceCan",
        "robots": ["Panda"],
        "controller_config": controller_config,
        "initialization_noise": None,
        "control_hz": CONTROL_HZ,
        "horizon": HORIZON,
        "camera": {
            "name": CAMERA_NAME,
            "width": CAMERA_WIDTH,
            "height": CAMERA_HEIGHT,
            "fovy_deg": CAMERA_FOVY_DEG,
        },
        "z_rotation": 0.0,
    }
    canonical = json.dumps(
        model_spec,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class RobosuitePickPlaceAdapter:
    """Record synchronized RGB and exact state for one Panda PickPlaceCan rollout."""

    def applied_subject_fields(
        self,
        subject: SubjectRef,
        case: CaseSpec,
    ) -> set[str]:
        del case
        fields = {"config.action_frame_offset_m"}
        if subject.code_sha is not None:
            fields.add("code_sha")
        return fields

    def run_episode(
        self,
        subject: SubjectRef,
        case: CaseSpec,
        *,
        seed: int,
        output_dir: Path,
    ) -> SimulationEpisode:
        import importlib.metadata

        import imageio.v2 as imageio
        import mujoco
        import numpy as np
        import robosuite
        from robosuite.controllers import load_composite_controller_config
        from robosuite.utils.sim_utils import check_contact

        from policyci.examples.mujoco_grasp.sensors import (
            finger_normal_force_n,
            sensor_vector,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        offset = tuple(float(value) for value in subject.config["action_frame_offset_m"])
        object_offset = case.resolved_object_position_offset_m()
        object_observation_bias = case.resolved_object_observation_bias_m()
        controller_config = load_composite_controller_config(controller="BASIC")
        model_sha256 = model_identity_sha256(controller_config)
        env = robosuite.make(
            env_name="PickPlaceCan",
            robots="Panda",
            controller_configs=controller_config,
            initialization_noise=None,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=CAMERA_NAME,
            camera_heights=CAMERA_HEIGHT,
            camera_widths=CAMERA_WIDTH,
            control_freq=CONTROL_HZ,
            horizon=HORIZON,
            ignore_done=True,
            hard_reset=False,
            reward_shaping=True,
            z_rotation=0.0,
            seed=seed,
        )
        writer = None
        try:
            observation = env.reset()
            camera_id = int(env.sim.model.camera_name2id(CAMERA_NAME))
            env.sim.model.cam_fovy[camera_id] = CAMERA_FOVY_DEG
            env.sim.forward()

            active_object = env.objects[env.object_id]
            object_body_id = int(env.obj_body_id[active_object.name])
            object_joint = active_object.joints[0]
            nominal_qpos = np.asarray(env.sim.data.get_joint_qpos(object_joint)).copy()
            nominal_object = np.asarray(env.sim.data.body_xpos[object_body_id]).copy()
            shifted_qpos = offset_object_qpos(nominal_qpos, object_offset)
            env.sim.data.set_joint_qpos(object_joint, np.asarray(shifted_qpos))
            env.sim.forward()
            initial_object = np.asarray(env.sim.data.body_xpos[object_body_id]).copy()
            expected_object = nominal_object + np.asarray(object_offset)
            if not np.allclose(initial_object, expected_object, rtol=0.0, atol=1e-6):
                raise ValueError(
                    "object position perturbation was not applied: "
                    f"expected={expected_object.tolist()}, actual={initial_object.tolist()}"
                )

            object_geoms = active_object.contact_geoms
            object_geom_ids = {
                int(env.sim.model.geom_name2id(name)) for name in object_geoms
            }
            robot = env.robots[0]
            gripper = single_component(robot.gripper, name="gripper")
            wrist_force_sensor = gripper.important_sensors["force_ee"]
            left_name = _first_geom_name(gripper.important_geoms["left_fingerpad"])
            right_name = _first_geom_name(gripper.important_geoms["right_fingerpad"])
            left_id = int(env.sim.model.geom_name2id(left_name))
            right_id = int(env.sim.model.geom_name2id(right_name))
            initial_bytes = b"".join(
                (
                    np.asarray(env.sim.data.qpos, dtype=np.float64).tobytes(),
                    np.asarray(env.sim.data.qvel, dtype=np.float64).tobytes(),
                )
            )
            initial_state_sha256 = hashlib.sha256(initial_bytes).hexdigest()

            target_position = np.asarray(env.target_bin_placements[env.object_id])
            perceived_object = perceived_object_position(
                initial_object,
                object_observation_bias,
            )
            phase_specs = pick_place_phase_specs(
                object_position=perceived_object,
                target_position=target_position,
            )
            eef_site_id = _eef_site_id(robot)

            video_path = output_dir / "episode.mp4"
            writer = imageio.get_writer(
                video_path,
                fps=CONTROL_HZ,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )
            samples: list[SimulationSample] = []
            step_index = 0
            for subtask, phase_steps, target, gripper_command in phase_specs:
                for _ in range(phase_steps):
                    eef_position = np.asarray(env.sim.data.site_xpos[eef_site_id]).copy()
                    translation = position_delta(
                        actual_eef=eef_position,
                        target=target,
                        action_frame_offset=offset,
                        gain=10.0,
                    )
                    action = np.asarray(
                        phase_action(translation, gripper_command=gripper_command),
                        dtype=np.float64,
                    )
                    observation, _, _, _ = env.step(action)
                    frame = np.asarray(observation[f"{CAMERA_NAME}_image"])[::-1]
                    writer.append_data(frame)

                    object_position = np.asarray(env.sim.data.body_xpos[object_body_id]).copy()
                    object_quaternion = np.asarray(env.sim.data.body_xquat[object_body_id]).copy()
                    eef_position = np.asarray(env.sim.data.site_xpos[eef_site_id]).copy()
                    eef_matrix = np.asarray(env.sim.data.site_xmat[eef_site_id]).reshape(3, 3)
                    from robosuite.utils.transform_utils import mat2quat

                    eef_quaternion_xyzw = tuple(float(value) for value in mat2quat(eef_matrix))
                    left_position = np.asarray(env.sim.data.geom_xpos[left_id])
                    right_position = np.asarray(env.sim.data.geom_xpos[right_id])
                    gripper_width = float(np.linalg.norm(left_position - right_position))
                    left_normal_force_n = finger_normal_force_n(
                        mujoco,
                        env.sim,
                        finger_geom_id=left_id,
                        object_geom_ids=object_geom_ids,
                    )
                    right_normal_force_n = finger_normal_force_n(
                        mujoco,
                        env.sim,
                        finger_geom_id=right_id,
                        object_geom_ids=object_geom_ids,
                    )
                    wrist_force_xyz_n = sensor_vector(
                        env.sim,
                        name=wrist_force_sensor,
                        expected_dimension=3,
                    )
                    samples.append(
                        SimulationSample(
                            time_s=step_index / CONTROL_HZ,
                            object_position_m=tuple(float(value) for value in object_position),
                            object_quaternion_xyzw=_xyzw(object_quaternion),
                            eef_position_m=tuple(float(value) for value in eef_position),
                            eef_quaternion_xyzw=eef_quaternion_xyzw,
                            gripper_width_m=gripper_width,
                            left_finger_contact=bool(
                                check_contact(env.sim, left_name, object_geoms)
                            ),
                            right_finger_contact=bool(
                                check_contact(env.sim, right_name, object_geoms)
                            ),
                            gripper_command=float(gripper_command),
                            left_finger_normal_force_n=left_normal_force_n,
                            right_finger_normal_force_n=right_normal_force_n,
                            wrist_force_xyz_n=wrist_force_xyz_n,
                            target_position_m=tuple(float(value) for value in target_position),
                            task_success=bool(env._check_success()),
                            subtask=subtask,
                        )
                    )
                    step_index += 1
        finally:
            if writer is not None:
                writer.close()
            env.close()

        source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        controller_hash = hashlib.sha256(
            json.dumps(controller_config, sort_keys=True, default=str).encode()
        ).hexdigest()
        runtime: dict[str, str | int | float] = {
            "robosuite_version": importlib.metadata.version("robosuite"),
            "mujoco_version": importlib.metadata.version("mujoco"),
            "imageio_version": importlib.metadata.version("imageio"),
            "camera": f"{CAMERA_NAME}:{CAMERA_WIDTH}x{CAMERA_HEIGHT}",
            "camera_fovy_deg": CAMERA_FOVY_DEG,
            "control_hz": CONTROL_HZ,
            "horizon": HORIZON,
            "task": "PickPlaceCan",
            "adapter_sha256": source_hash,
            "controller_sha256": controller_hash,
            "subject_code_sha": subject.code_sha or "",
            "object_offset_x_m": object_offset[0],
            "object_offset_y_m": object_offset[1],
            "object_offset_z_m": object_offset[2],
            "object_observation_bias_x_m": object_observation_bias[0],
            "object_observation_bias_y_m": object_observation_bias[1],
            "object_observation_bias_z_m": object_observation_bias[2],
            "nominal_object_x_m": float(nominal_object[0]),
            "nominal_object_y_m": float(nominal_object[1]),
            "nominal_object_z_m": float(nominal_object[2]),
        }
        return SimulationEpisode(
            trace=SimulationTrace(
                sample_rate_hz=CONTROL_HZ,
                seed=seed,
                initial_state_sha256=initial_state_sha256,
                model_sha256=model_sha256,
                samples=samples,
            ),
            video_path=video_path,
            runtime=runtime,
        )
