"""Portable simulator sensor evidence contract and extraction helpers."""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from policyci.schemas import SimulationSample, SimulationTrace


def _sensor_helpers():
    try:
        module = importlib.import_module("policyci.examples.mujoco_grasp.sensors")
    except ModuleNotFoundError:
        pytest.fail("MuJoCo sensor extraction helpers are missing")
    return module


def _sample_payload() -> dict:
    return {
        "time_s": 0.0,
        "object_position_m": [0.0, 0.0, 0.02],
        "object_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "eef_position_m": [0.0, 0.0, 0.10],
        "eef_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "gripper_width_m": 0.04,
        "left_finger_contact": False,
        "right_finger_contact": False,
    }


def test_simulation_sample_round_trips_sensor_evidence():
    sample = SimulationSample.model_validate(
        {
            **_sample_payload(),
            "gripper_command": 1.0,
            "left_finger_normal_force_n": 3.5,
            "right_finger_normal_force_n": 4.25,
            "wrist_force_xyz_n": [1.0, -2.0, 3.0],
        }
    )

    assert sample.gripper_command == 1.0
    assert sample.left_finger_normal_force_n == pytest.approx(3.5)
    assert sample.right_finger_normal_force_n == pytest.approx(4.25)
    assert sample.wrist_force_xyz_n == pytest.approx((1.0, -2.0, 3.0))


def test_simulation_sample_rejects_partial_sensor_evidence():
    with pytest.raises(ValueError, match="sensor evidence must be all present or all absent"):
        SimulationSample.model_validate(
            {
                **_sample_payload(),
                "gripper_command": 1.0,
            }
        )


def test_simulation_trace_rejects_sensor_evidence_missing_from_one_frame():
    sensor_sample = SimulationSample.model_validate(
        {
            **_sample_payload(),
            "gripper_command": 1.0,
            "left_finger_normal_force_n": 3.5,
            "right_finger_normal_force_n": 4.25,
            "wrist_force_xyz_n": [1.0, -2.0, 3.0],
        }
    )
    legacy_sample = SimulationSample.model_validate(
        {
            **_sample_payload(),
            "time_s": 0.05,
        }
    )

    with pytest.raises(ValueError, match="sensor evidence must cover every frame"):
        SimulationTrace(
            sample_rate_hz=20.0,
            seed=1000,
            initial_state_sha256="a" * 64,
            model_sha256="b" * 64,
            samples=[sensor_sample, legacy_sample],
        )


def test_sensor_vector_resolves_the_single_named_wrist_sensor():
    module = _sensor_helpers()
    model = SimpleNamespace(
        sensor_names=("robot0_force_ee", "robot0_torque_ee"),
        sensor_adr=(0, 3),
        sensor_dim=(3, 3),
        sensor_name2id=lambda name: {
            "robot0_force_ee": 0,
            "robot0_torque_ee": 1,
        }[name],
    )
    sim = SimpleNamespace(
        model=model,
        data=SimpleNamespace(sensordata=(1.0, -2.0, 3.0, 0.1, 0.2, 0.3)),
    )

    assert module.sensor_vector(
        sim,
        name="robot0_force_ee",
        expected_dimension=3,
    ) == pytest.approx((1.0, -2.0, 3.0))


def test_sensor_vector_rejects_missing_or_wrong_dimension_sensor():
    module = _sensor_helpers()
    sim = SimpleNamespace(
        model=SimpleNamespace(
            sensor_names=("robot0_force_ee",),
            sensor_adr=(0,),
            sensor_dim=(2,),
            sensor_name2id=lambda name: 0,
        ),
        data=SimpleNamespace(sensordata=(1.0, 2.0)),
    )

    with pytest.raises(ValueError, match="is not present"):
        module.sensor_vector(sim, name="missing_force", expected_dimension=3)
    with pytest.raises(ValueError, match="dimension 2"):
        module.sensor_vector(sim, name="robot0_force_ee", expected_dimension=3)


def test_sensor_sample_contract_rejects_missing_or_invalid_gripper_width():
    module = _sensor_helpers()
    valid = {
        "gripper_command": 1.0,
        "gripper_width_m": 0.04,
        "left_finger_normal_force_n": 3.5,
        "right_finger_normal_force_n": 4.25,
        "wrist_force_xyz_n": [1.0, -2.0, 3.0],
    }

    for invalid in (
        {key: value for key, value in valid.items() if key != "gripper_width_m"},
        {**valid, "gripper_width_m": -0.01},
        {**valid, "gripper_width_m": float("nan")},
    ):
        with pytest.raises(ValueError, match="gripper_width_m"):
            module.validate_sensor_evidence_sample(invalid)


def test_gripper_closing_response_compares_open_and_close_commands():
    module = _sensor_helpers()

    assert module.gripper_closing_delta_m(
        commands=[-1.0, -1.0, 1.0, 1.0],
        widths_m=[0.08, 0.07, 0.05, 0.04],
    ) == pytest.approx(0.04)
    assert module.gripper_closing_delta_m(
        commands=[-1.0, -1.0, 1.0, 1.0],
        widths_m=[0.02, 0.02, 0.04, 0.04],
    ) == pytest.approx(-0.02)


def test_importing_sensor_helper_does_not_eagerly_import_adapter():
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import policyci.examples.mujoco_grasp.sensors; "
                "assert 'policyci.examples.mujoco_grasp.adapter' not in sys.modules"
            ),
        ],
        cwd=repo,
        env={"PYTHONPATH": str(repo / "src")},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_finger_contact_force_sums_only_contacts_with_the_object():
    module = _sensor_helpers()
    contacts = (
        SimpleNamespace(geom1=10, geom2=20),
        SimpleNamespace(geom1=10, geom2=99),
        SimpleNamespace(geom1=21, geom2=10),
    )
    force_by_contact = {0: 2.5, 1: 99.0, 2: 1.25}

    class FakeMujoco:
        @staticmethod
        def mj_contactForce(model, data, index, result):
            assert model == "native-model"
            assert data == "native-data"
            result[0] = force_by_contact[index]

    sim = SimpleNamespace(
        model=SimpleNamespace(_model="native-model"),
        data=SimpleNamespace(
            _data="native-data",
            ncon=len(contacts),
            contact=contacts,
        ),
    )

    assert module.finger_normal_force_n(
        FakeMujoco,
        sim,
        finger_geom_id=10,
        object_geom_ids={20, 21},
    ) == pytest.approx(3.75)
