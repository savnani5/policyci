"""Sensor-rich Rerun generation for ALOHA replay evidence."""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from policyci.aloha_rerun import AlohaMcapEvent, render_aloha_rerun
from policyci.aloha_visualization import AlohaReplayVisualization
from policyci.rerun_visualizer import open_visualization, render_visualization
from policyci.schemas import GateVerdict, TaskStatus


def _model(tmp_path: Path) -> AlohaReplayVisualization:
    for name in ("input.mcap", "baseline-output.mcap", "candidate-output.mcap"):
        (tmp_path / name).write_bytes(name.encode())
    return AlohaReplayVisualization(
        check_dir=tmp_path,
        check_id="check_aloha",
        suite_name="aloha-coffee-replay",
        case_id="aloha_coffee_replay",
        unit_id="repeat-000",
        baseline_id="aloha-action-baseline",
        candidate_id="aloha-gripper-delay-300ms",
        verdict=GateVerdict.BLOCK,
        gate_summary="BLOCK: candidate regressed",
        comparison_status=TaskStatus.REGRESSED,
        metric="within_tolerance",
        baseline_value=True,
        candidate_value=False,
        input_mcap=tmp_path / "input.mcap",
        baseline_output_mcap=tmp_path / "baseline-output.mcap",
        candidate_output_mcap=tmp_path / "candidate-output.mcap",
        input_sha256="a" * 64,
        time_s=[0.0, 0.02, 0.04, 0.06],
        left_gripper_expected=[0.0, 0.1, 0.2, 0.3],
        right_gripper_expected=[0.0, 0.2, 0.4, 0.6],
        baseline_left_gripper_observed=[0.0, 0.1, 0.2, 0.3],
        baseline_right_gripper_observed=[0.0, 0.2, 0.4, 0.6],
        candidate_left_gripper_observed=[0.0, 0.0, 0.1, 0.2],
        candidate_right_gripper_observed=[0.0, 0.0, 0.2, 0.4],
        baseline_left_gripper_error=[0.0, 0.0, 0.0, 0.0],
        baseline_right_gripper_error=[0.0, 0.0, 0.0, 0.0],
        candidate_left_gripper_error=[0.0, 0.1, 0.1, 0.1],
        candidate_right_gripper_error=[0.0, 0.2, 0.2, 0.2],
        baseline_max_gripper_error=[0.0, 0.0, 0.0, 0.0],
        candidate_max_gripper_error=[0.0, 0.2, 0.2, 0.2],
        baseline_max_arm_error=[0.0, 0.0, 0.0, 0.0],
        candidate_max_arm_error=[0.0, 0.0, 0.0, 0.0],
        gripper_tolerance=0.05,
        arm_tolerance=1e-9,
        first_candidate_violation_index=1,
        first_candidate_violation_time_s=0.02,
        baseline_unit_within_tolerance=True,
        candidate_unit_within_tolerance=False,
        summary_markdown=(
            "# BLOCK: OPEN-LOOP COMPONENT REGRESSION\n\n"
            "**Demo fault:** `300 ms GRIPPER COMMAND DELAY`\n\n"
            "Recorded cameras, joint states, and effort are immutable context. "
            "The gate proves an output regression under identical recorded inputs; "
            "it does not show a counterfactual physical outcome."
        ),
    )


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (40, 100, 180)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _events(model: AlohaReplayVisualization) -> list[AlohaMcapEvent]:
    events = [
        AlohaMcapEvent(
            topic="/episode/task",
            time_s=0.0,
            task="Make coffee",
        )
    ]
    frame = _jpeg()
    for index, time_s in enumerate(model.time_s):
        for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            events.append(
                AlohaMcapEvent(
                    topic=f"/episode/{camera}/image/compressed",
                    time_s=time_s,
                    jpeg=frame,
                )
            )
        efforts = [0.0] * 14
        efforts[6] = float(index)
        efforts[13] = float(index) + 0.5
        events.append(
            AlohaMcapEvent(
                topic="/episode/joint_states",
                time_s=time_s,
                joint_positions=tuple([0.0] * 14),
                joint_efforts=tuple(efforts),
            )
        )
        action = [0.0] * 14
        action[6] = model.left_gripper_expected[index]
        action[13] = model.right_gripper_expected[index]
        events.append(
            AlohaMcapEvent(
                topic="/episode/reference_action",
                time_s=time_s,
                action=tuple(action),
            )
        )
    return sorted(events, key=lambda event: event.time_s)


def test_rendered_aloha_dashboard_contains_three_cameras_and_gate_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _model(tmp_path)
    monkeypatch.setattr(
        "policyci.aloha_rerun.iter_aloha_mcap",
        lambda _path: iter(_events(model)),
    )

    output = render_aloha_rerun(model, tmp_path / "aloha.rrd")

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
    printed = subprocess.run(
        [rerun, "rrd", "print", "-vv", str(output)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert printed.returncode == 0, printed.stderr or printed.stdout
    for entity in (
        "/sensors/overhead",
        "/sensors/left_wrist",
        "/sensors/right_wrist",
        "/gripper_commands/left/candidate",
        "/gripper_commands/right/candidate",
        "/effort/left_gripper",
        "/effort/right_gripper",
        "/error/candidate",
        "/error/tolerance",
        "/events/first_candidate_violation",
    ):
        assert entity in printed.stdout


def test_generic_renderer_dispatches_aloha_and_opens_only_the_rrd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _model(tmp_path)
    monkeypatch.setattr(
        "policyci.aloha_rerun.iter_aloha_mcap",
        lambda _path: iter(_events(model)),
    )
    output = render_visualization(model, tmp_path / "generic-aloha.rrd")
    calls: list[list[str]] = []

    def fake_popen(argv: list[str], **_kwargs):
        calls.append(argv)
        return object()

    open_visualization(
        output,
        model,
        which=lambda executable: "/opt/rerun" if executable == "rerun" else None,
        popen=fake_popen,
    )

    assert calls == [["/opt/rerun", str(output)]]
