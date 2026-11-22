"""Real Robosuite Panda Lift rollout used by the conference demo."""

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
HORIZON = 160
CAMERA_NAME = "agentview"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


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


def _model_xml(sim) -> str:
    get_xml = getattr(sim.model, "get_xml", None)
    if callable(get_xml):
        return str(get_xml())
    raise ValueError("MuJoCo model does not expose XML for identity hashing")


class RobosuiteLiftAdapter:
    """Record synchronized RGB and exact state for one Panda Lift rollout."""

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
        del case
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
        controller_config = load_composite_controller_config(controller="BASIC")
        env = robosuite.make(
            env_name="Lift",
            robots="Panda",
            controller_configs=controller_config,
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
            seed=seed,
        )
        writer = None
        try:
            observation = env.reset()
            cube_joint = env.cube.joints[0]
            cube_qpos = np.asarray(env.sim.data.get_joint_qpos(cube_joint)).copy()
            cube_qpos[3:] = (1.0, 0.0, 0.0, 0.0)
            env.sim.data.set_joint_qpos(cube_joint, cube_qpos)
            env.sim.forward()
            initial_bytes = b"".join(
                (
                    np.asarray(env.sim.data.qpos, dtype=np.float64).tobytes(),
                    np.asarray(env.sim.data.qvel, dtype=np.float64).tobytes(),
                )
            )
            initial_state_sha256 = hashlib.sha256(initial_bytes).hexdigest()
            model_sha256 = hashlib.sha256(_model_xml(env.sim).encode()).hexdigest()

            cube_body_id = int(env.cube_body_id)
            initial_cube = np.asarray(env.sim.data.body_xpos[cube_body_id]).copy()
            phase_specs = (
                (40, initial_cube + np.array((0.0, 0.0, 0.12)), -1.0),
                (35, initial_cube + np.array((0.0, 0.0, 0.015)), -1.0),
                (15, initial_cube + np.array((0.0, 0.0, 0.015)), 1.0),
                (70, initial_cube + np.array((0.0, 0.0, 0.20)), 0.0),
            )
            robot = env.robots[0]
            gripper = single_component(robot.gripper, name="gripper")
            wrist_force_sensor = gripper.important_sensors["force_ee"]
            left_name = _first_geom_name(gripper.important_geoms["left_fingerpad"])
            right_name = _first_geom_name(gripper.important_geoms["right_fingerpad"])
            left_id = int(env.sim.model.geom_name2id(left_name))
            right_id = int(env.sim.model.geom_name2id(right_name))
            cube_geoms = env.cube.contact_geoms
            cube_geom_ids = {int(env.sim.model.geom_name2id(name)) for name in cube_geoms}
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
            for phase_steps, target, gripper_command in phase_specs:
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

                    cube_position = np.asarray(env.sim.data.body_xpos[cube_body_id]).copy()
                    cube_quaternion = np.asarray(env.sim.data.body_xquat[cube_body_id]).copy()
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
                        object_geom_ids=cube_geom_ids,
                    )
                    right_normal_force_n = finger_normal_force_n(
                        mujoco,
                        env.sim,
                        finger_geom_id=right_id,
                        object_geom_ids=cube_geom_ids,
                    )
                    wrist_force_xyz_n = sensor_vector(
                        env.sim,
                        name=wrist_force_sensor,
                        expected_dimension=3,
                    )
                    samples.append(
                        SimulationSample(
                            time_s=step_index / CONTROL_HZ,
                            object_position_m=tuple(float(value) for value in cube_position),
                            object_quaternion_xyzw=_xyzw(cube_quaternion),
                            eef_position_m=tuple(float(value) for value in eef_position),
                            eef_quaternion_xyzw=eef_quaternion_xyzw,
                            gripper_width_m=gripper_width,
                            left_finger_contact=bool(check_contact(env.sim, left_name, cube_geoms)),
                            right_finger_contact=bool(
                                check_contact(env.sim, right_name, cube_geoms)
                            ),
                            gripper_command=float(gripper_command),
                            left_finger_normal_force_n=left_normal_force_n,
                            right_finger_normal_force_n=right_normal_force_n,
                            wrist_force_xyz_n=wrist_force_xyz_n,
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
            "control_hz": CONTROL_HZ,
            "horizon": HORIZON,
            "adapter_sha256": source_hash,
            "controller_sha256": controller_hash,
            "subject_code_sha": subject.code_sha or "",
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
