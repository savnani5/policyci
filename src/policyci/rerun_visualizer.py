"""Optional Rerun renderer for validated replay evidence."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping

from policyci.visualization import CheckVisualization

if TYPE_CHECKING:
    from policyci.aloha_visualization import AlohaReplayVisualization
    from policyci.manipulation_visualization import ManipulationVisualization


def _rerun_modules():
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:
        raise RuntimeError(
            "Rerun visualization is not installed; run `uv sync --extra visualize`"
        ) from exc
    return rr, rrb


def _find_compatible_ffmpeg(
    search_path: str,
    *,
    run: Callable[..., object] = subprocess.run,
) -> Path | None:
    """Return the first FFmpeg release on PATH that Rerun can identify."""
    seen: set[Path] = set()
    for raw_directory in search_path.split(os.pathsep):
        directory = Path(raw_directory or ".")
        candidate = directory / "ffmpeg"
        if candidate in seen or not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        seen.add(candidate)
        try:
            completed = run(
                [str(candidate), "-version"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if getattr(completed, "returncode", 1) != 0:
            continue
        version_output = str(getattr(completed, "stdout", ""))
        match = re.match(r"^ffmpeg version (?:n)?(\d+)\.(\d+)", version_output)
        if match is None:
            continue
        version = (int(match.group(1)), int(match.group(2)))
        if version >= (5, 1):
            return candidate
    return None


def _path_with_first(directory: Path, search_path: str) -> str:
    selected = str(directory)
    remaining = [
        entry
        for entry in search_path.split(os.pathsep)
        if entry and Path(entry) != directory
    ]
    return os.pathsep.join([selected, *remaining])


def _blueprint(rrb, model: CheckVisualization):
    unit = f" ({model.unit})" if model.unit else ""
    first_failure = (
        f"; first candidate failure: sample {model.first_candidate_violation_index}"
        if model.first_candidate_violation_index is not None
        else ""
    )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.TextDocumentView(
                origin="/summary",
                name="Automatic verdict",
            ),
            rrb.TimeSeriesView(
                origin="/command",
                name=f"Expected vs observed command{unit}",
            ),
            rrb.TimeSeriesView(
                origin="/error",
                name=f"Absolute error vs tolerance{unit}{first_failure}",
            ),
            row_shares=[1, 3, 3],
        ),
        collapse_panels=True,
    )


def _log_series_styles(recording, rr, model: CheckVisualization) -> None:
    baseline_color = [57, 217, 138] if model.baseline_unit_pass else [255, 92, 105]
    candidate_color = [57, 217, 138] if model.candidate_unit_pass else [255, 92, 105]
    styles = {
        "/command/expected": ([160, 168, 178], "Expected"),
        "/command/baseline": (baseline_color, "Baseline observed"),
        "/command/candidate": (candidate_color, "Candidate observed"),
        "/command/tolerance_upper": ([244, 189, 80], "Tolerance upper"),
        "/command/tolerance_lower": ([244, 189, 80], "Tolerance lower"),
        "/error/baseline": (baseline_color, "Baseline absolute error"),
        "/error/candidate": (candidate_color, "Candidate absolute error"),
        "/error/tolerance": ([244, 189, 80], "Tolerance"),
        "/error/first_candidate_violation": ([255, 255, 255], "First candidate violation"),
    }
    for entity, (color, name) in styles.items():
        recording.log(
            entity,
            rr.SeriesLines(colors=color, names=name),
            static=True,
        )


def render_rerun_visualization(
    model: CheckVisualization,
    output_path: Path,
) -> Path:
    """Write a durable three-panel RoboLens RRD."""
    rr, rrb = _rerun_modules()
    output = output_path.resolve()
    if not output.is_relative_to(model.check_dir.resolve()):
        raise ValueError(f"Rerun output is not confined to the check directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    blueprint = _blueprint(rrb, model)
    recording_id = f"{model.check_id}-{model.case_id}-{model.unit_id}"

    with rr.RecordingStream(
        "robolens_replay_check",
        recording_id=recording_id,
        send_properties=False,
    ) as recording:
        recording.save(output, default_blueprint=blueprint)
        recording.log(
            "/summary",
            rr.TextDocument(model.summary_markdown, media_type="text/markdown"),
            static=True,
        )
        _log_series_styles(recording, rr, model)
        for index, expected in enumerate(model.expected):
            recording.set_time("sample", sequence=index)
            recording.log("/command/expected", rr.Scalars(expected))
            recording.log(
                "/command/baseline",
                rr.Scalars(model.baseline_observed[index]),
            )
            recording.log(
                "/command/candidate",
                rr.Scalars(model.candidate_observed[index]),
            )
            recording.log(
                "/command/tolerance_upper",
                rr.Scalars(expected + model.tolerance),
            )
            recording.log(
                "/command/tolerance_lower",
                rr.Scalars(expected - model.tolerance),
            )
            recording.log(
                "/error/baseline",
                rr.Scalars(model.baseline_error[index]),
            )
            recording.log(
                "/error/candidate",
                rr.Scalars(model.candidate_error[index]),
            )
            recording.log("/error/tolerance", rr.Scalars(model.tolerance))
            if index == model.first_candidate_violation_index:
                recording.log(
                    "/error/first_candidate_violation",
                    rr.Scalars(model.candidate_error[index]),
                )

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Rerun did not produce a recording: {output}")
    return output


def render_visualization(
    model: CheckVisualization | AlohaReplayVisualization | ManipulationVisualization,
    output_path: Path,
) -> Path:
    """Dispatch a validated evidence model to its runner-specific renderer."""
    if isinstance(model, CheckVisualization):
        return render_rerun_visualization(model, output_path)

    from policyci.aloha_visualization import AlohaReplayVisualization

    if isinstance(model, AlohaReplayVisualization):
        from policyci.aloha_rerun import render_aloha_rerun

        return render_aloha_rerun(model, output_path)

    from policyci.manipulation_rerun import render_manipulation_rerun
    from policyci.manipulation_visualization import ManipulationVisualization

    if isinstance(model, ManipulationVisualization):
        return render_manipulation_rerun(model, output_path)
    raise TypeError(f"unsupported visualization model: {type(model).__name__}")


def open_rerun_visualization(
    rrd_path: Path,
    mcap_paths: list[Path],
    *,
    which: Callable[[str], str | None] = shutil.which,
    popen: Callable[..., object] = subprocess.Popen,
    run: Callable[..., object] = subprocess.run,
    environ: Mapping[str, str] | None = None,
    require_compatible_ffmpeg: bool = False,
) -> None:
    """Open the dashboard first and raw paired MCAP recordings afterward."""
    executable = which("rerun")
    if executable is None:
        raise RuntimeError(
            "Rerun executable is unavailable; run `uv sync --extra visualize`"
        )
    launch_env = dict(os.environ if environ is None else environ)
    search_path = launch_env.get("PATH", "")
    compatible_ffmpeg = _find_compatible_ffmpeg(search_path, run=run)
    if compatible_ffmpeg is None and require_compatible_ffmpeg:
        raise RuntimeError(
            "Rerun video playback requires FFmpeg >= 5.1 with a parseable release "
            "version, but none was found on PATH. Install it with `brew install "
            "ffmpeg`."
        )
    if compatible_ffmpeg is not None:
        launch_env["PATH"] = _path_with_first(
            compatible_ffmpeg.parent,
            search_path,
        )
    popen(
        [executable, str(rrd_path), *(str(path) for path in mcap_paths)],
        env=launch_env,
        start_new_session=True,
    )


def open_visualization(
    rrd_path: Path,
    model: CheckVisualization | AlohaReplayVisualization | ManipulationVisualization,
    *,
    which: Callable[[str], str | None] = shutil.which,
    popen: Callable[..., object] = subprocess.Popen,
) -> None:
    """Open a rendered dashboard plus any runner-specific raw evidence."""
    mcap_paths = model.mcap_paths if isinstance(model, CheckVisualization) else []
    open_rerun_visualization(
        rrd_path,
        mcap_paths,
        which=which,
        popen=popen,
        require_compatible_ffmpeg=not isinstance(model, CheckVisualization),
    )
