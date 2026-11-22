"""Durable Rerun dashboard generation and native launch."""

from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace
from pathlib import Path

import pytest

from policyci.rerun_visualizer import (
    _find_compatible_ffmpeg,
    open_rerun_visualization,
    render_rerun_visualization,
)
from policyci.schemas import GateVerdict, TaskStatus
from policyci.visualization import CheckVisualization


def _model(tmp_path: Path) -> CheckVisualization:
    baseline_mcap = tmp_path / "baseline.mcap"
    candidate_mcap = tmp_path / "candidate.mcap"
    baseline_mcap.write_bytes(b"baseline")
    candidate_mcap.write_bytes(b"candidate")
    return CheckVisualization(
        check_dir=tmp_path,
        check_id="check_demo",
        suite_name="twist-replay",
        case_id="stop_command_regression",
        unit_id="repeat-000",
        baseline_id="fixed",
        candidate_id="stop-regression",
        verdict=GateVerdict.BLOCK,
        gate_summary="BLOCK: required case regressed",
        comparison_status=TaskStatus.REGRESSED,
        metric="within_tolerance",
        baseline_value=True,
        candidate_value=False,
        judge={"tolerance": 0.05},
        tolerance=0.05,
        timeline="sample_index",
        expected=[0.0, 0.5, 0.0],
        baseline_observed=[0.0, 0.5, 0.0],
        candidate_observed=[0.0, 0.5, 0.4],
        baseline_error=[0.0, 0.0, 0.0],
        candidate_error=[0.0, 0.0, 0.4],
        baseline_unit_within_tolerance=True,
        candidate_unit_within_tolerance=False,
        first_candidate_violation_index=2,
        unit="m/s",
        baseline_mcaps=[baseline_mcap],
        candidate_mcaps=[candidate_mcap],
        summary_markdown=(
            "# BLOCK — REGRESSED\n\nRoboLens automatically evaluated the metric and gate."
        ),
    )


def test_rendered_rerun_dashboard_is_a_valid_rrd(tmp_path: Path):
    output = render_rerun_visualization(_model(tmp_path), tmp_path / "view.rrd")

    assert output == tmp_path / "view.rrd"
    assert output.stat().st_size > 0
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
    assert "/error/first_candidate_violation" in printed.stdout


def test_open_rerun_uses_argument_vector_with_rrd_then_paired_mcaps(tmp_path: Path):
    model = _model(tmp_path)
    rrd = tmp_path / "view.rrd"
    rrd.write_bytes(b"rrd")
    ffmpeg_dir = tmp_path / "ffmpeg-bin"
    ffmpeg_dir.mkdir()
    ffmpeg = ffmpeg_dir / "ffmpeg"
    ffmpeg.write_text("placeholder", encoding="utf-8")
    ffmpeg.chmod(0o755)
    calls: list[tuple[list[str], bool, dict[str, str]]] = []

    def fake_popen(
        argv: list[str],
        *,
        env: dict[str, str],
        start_new_session: bool,
    ):
        calls.append((argv, start_new_session, env))
        return object()

    def fake_run(argv: list[str], **_kwargs):
        assert argv == [str(ffmpeg), "-version"]
        return SimpleNamespace(returncode=0, stdout="ffmpeg version 8.1 Copyright")

    open_rerun_visualization(
        rrd,
        model.mcap_paths,
        which=lambda executable: "/opt/rerun" if executable == "rerun" else None,
        popen=fake_popen,
        run=fake_run,
        environ={"PATH": str(ffmpeg_dir), "ROBO_TEST": "present"},
    )

    assert calls == [
        (
            [
                "/opt/rerun",
                str(rrd),
                str(model.baseline_mcaps[0]),
                str(model.candidate_mcaps[0]),
            ],
            True,
            {"PATH": str(ffmpeg_dir), "ROBO_TEST": "present"},
        )
    ]


def test_find_compatible_ffmpeg_skips_unparseable_git_build(tmp_path: Path):
    git_bin = tmp_path / "git-build"
    release_bin = tmp_path / "release-build"
    git_bin.mkdir()
    release_bin.mkdir()
    for directory in (git_bin, release_bin):
        candidate = directory / "ffmpeg"
        candidate.write_text("placeholder", encoding="utf-8")
        candidate.chmod(0o755)

    def fake_run(argv: list[str], **_kwargs):
        version = (
            "ffmpeg version 1addefb Copyright"
            if Path(argv[0]).parent == git_bin
            else "ffmpeg version 8.1 Copyright"
        )
        return SimpleNamespace(returncode=0, stdout=version)

    selected = _find_compatible_ffmpeg(
        f"{git_bin}:{release_bin}",
        run=fake_run,
    )

    assert selected == release_bin / "ffmpeg"


def test_open_rerun_requires_compatible_ffmpeg_for_video_dashboard(tmp_path: Path):
    invalid_bin = tmp_path / "invalid-bin"
    invalid_bin.mkdir()
    ffmpeg = invalid_bin / "ffmpeg"
    ffmpeg.write_text("placeholder", encoding="utf-8")
    ffmpeg.chmod(0o755)

    def fake_run(_argv: list[str], **_kwargs):
        return SimpleNamespace(returncode=0, stdout="ffmpeg version 1addefb")

    with pytest.raises(RuntimeError, match="FFmpeg >= 5.1"):
        open_rerun_visualization(
            tmp_path / "view.rrd",
            [],
            which=lambda executable: "/opt/rerun" if executable == "rerun" else None,
            popen=lambda *_args, **_kwargs: object(),
            run=fake_run,
            environ={"PATH": str(invalid_bin)},
            require_compatible_ffmpeg=True,
        )


def test_renderer_rejects_output_outside_check_directory(tmp_path: Path):
    check_dir = tmp_path / "check"
    check_dir.mkdir()
    model = _model(check_dir)

    with pytest.raises(ValueError, match="confined"):
        render_rerun_visualization(model, tmp_path / "outside.rrd")


def test_renderer_rejects_visualizations_symlink_outside_check_directory(
    tmp_path: Path,
):
    check_dir = tmp_path / "check"
    check_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (check_dir / "visualizations").symlink_to(outside, target_is_directory=True)
    model = _model(check_dir)

    with pytest.raises(ValueError, match="confined"):
        render_rerun_visualization(
            model,
            check_dir / "visualizations/case/repeat-000.rrd",
        )
