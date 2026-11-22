"""Optional Rerun renderer for absolute simulation robustness gates."""

from __future__ import annotations

from pathlib import Path

from policyci.robustness_visualization import RobustnessVisualization


def _rerun_modules():
    try:
        import rerun as rr
        import rerun.blueprint as rrb
        import rerun.blueprint.components as rrbc
    except ImportError as exc:
        raise RuntimeError(
            "Rerun visualization is not installed; run `uv sync --extra visualize`"
        ) from exc
    return rr, rrb, rrbc


def _scenario_label(
    offset_m: tuple[float, float, float],
    observation_bias_m: tuple[float, float, float],
) -> str:
    labels = []
    if any(abs(value) > 1e-12 for value in offset_m):
        labels.append(f"x={offset_m[0] * 100:+.0f} cm")
    if any(abs(value) > 1e-12 for value in observation_bias_m):
        labels.append(f"perception bias={observation_bias_m[0] * 100:+.0f} cm")
    return " · ".join(labels) if labels else "nominal"


def _blueprint(rrb, rrbc, model: RobustnessVisualization):
    timeline_end_s = model.times_s[-1]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.TextDocumentView(
                origin="/verdict",
                name="ROBUSTNESS GATE · BLOCK",
            ),
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial2DView(
                        origin="/base_case",
                        name=(
                            "BASE · "
                            f"{_scenario_label(model.base_offset_m, model.base_observation_bias_m)} "
                            "· PASS"
                        ),
                        background=[12, 16, 24],
                    ),
                    rrb.TimeSeriesView(
                        origin="/task_progress/base_case",
                        name=f"BASE · VERIFIED PROGRESS {model.base_task_progress_pct[-1]:.0f}%",
                        axis_y=rrb.ScalarAxis(range=[0.0, 100.0], zoom_lock=True),
                        plot_legend=rrb.PlotLegend(visible=False),
                    ),
                    row_shares=[8, 2],
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(
                        origin="/perturbed",
                        name=(
                            "PERTURBED · "
                            f"{_scenario_label(model.perturbed_offset_m, model.perturbed_observation_bias_m)} "
                            "· BLOCK"
                        ),
                        background=[12, 16, 24],
                    ),
                    rrb.TimeSeriesView(
                        origin="/task_progress/perturbed",
                        name=(
                            "PERTURBED · VERIFIED PROGRESS "
                            f"{model.perturbed_task_progress_pct[-1]:.0f}% · BLOCK"
                        ),
                        axis_y=rrb.ScalarAxis(range=[0.0, 100.0], zoom_lock=True),
                        plot_legend=rrb.PlotLegend(visible=False),
                    ),
                    row_shares=[8, 2],
                ),
                column_shares=[1, 1],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    origin="/exact_state/target_distance",
                    name="Object-to-target XY distance (m)",
                ),
                rrb.TimeSeriesView(
                    origin="/exact_state/object_height",
                    name="Object height (m)",
                ),
                column_shares=[1, 1],
            ),
            rrb.Spatial2DView(
                origin="/subtask_timeline",
                name="SUBTASK SEGMENTATION · SHARED CONTROLLER SCHEDULE",
                background=[7, 9, 8],
                visual_bounds=rrb.VisualBounds2D(
                    x_range=[-1.2, timeline_end_s + 0.35],
                    y_range=[-0.22, 0.22],
                ),
            ),
            row_shares=[2, 13, 3, 3],
        ),
        rrb.TimePanel(
            state="collapsed",
            timeline=model.timeline,
            playback_speed=1.0,
            play_state=rrbc.PlayState.Playing,
            loop_mode=rrbc.LoopMode.Off,
        ),
        collapse_panels=True,
    )


def _log_styles(recording, rr) -> None:
    styles = {
        "/exact_state/target_distance/base_case": (
            [57, 217, 138],
            "Base case distance",
        ),
        "/exact_state/target_distance/perturbed": (
            [255, 92, 105],
            "Perturbed distance",
        ),
        "/exact_state/target_distance/threshold": (
            [244, 189, 80],
            "5 cm diagnostic limit",
        ),
        "/exact_state/object_height/base_case": (
            [57, 217, 138],
            "Base case height",
        ),
        "/exact_state/object_height/perturbed": (
            [255, 92, 105],
            "Perturbed height",
        ),
        "/exact_state/task_success/base_case": (
            [57, 217, 138],
            "Base case success",
        ),
        "/exact_state/task_success/perturbed": (
            [255, 92, 105],
            "Perturbed success",
        ),
        "/task_progress/base_case": (
            [57, 217, 138],
            "Base verified progress",
        ),
        "/task_progress/perturbed": (
            [255, 92, 105],
            "Perturbed verified progress",
        ),
    }
    for entity, (color, name) in styles.items():
        interpolation_mode = (
            rr.components.InterpolationMode.StepAfter
            if entity.startswith("/task_progress/")
            else rr.components.InterpolationMode.Linear
        )
        recording.log(
            entity,
            rr.SeriesLines(
                colors=color,
                names=name,
                interpolation_mode=interpolation_mode,
            ),
            static=True,
        )


_SUBTASK_COLORS = {
    "approach": [73, 91, 111, 220],
    "align": [74, 126, 171, 220],
    "grasp": [214, 164, 74, 230],
    "lift": [73, 155, 116, 225],
    "transport": [65, 139, 145, 225],
    "place": [132, 111, 176, 225],
    "release": [181, 103, 94, 225],
    "retreat": [99, 107, 116, 220],
}


def _segments(
    times_s: list[float],
    subtasks: list[str],
) -> list[tuple[str, float, float]]:
    if not times_s or len(times_s) != len(subtasks):
        raise ValueError("subtask labels must align with the simulation timeline")
    step_s = times_s[-1] - times_s[-2] if len(times_s) > 1 else 0.05
    segments: list[tuple[str, float, float]] = []
    active = subtasks[0]
    start_s = times_s[0]
    for index in range(1, len(times_s)):
        if subtasks[index] != active:
            segments.append((active, start_s, times_s[index]))
            active = subtasks[index]
            start_s = times_s[index]
    segments.append((active, start_s, times_s[-1] + step_s))
    return segments


def _shared_subtask_segments(
    model: RobustnessVisualization,
) -> list[tuple[str, float, float]]:
    if model.base_subtasks != model.perturbed_subtasks:
        raise ValueError(
            "shared subtask timeline requires matching base and perturbed annotations"
        )
    return _segments(model.times_s, model.base_subtasks)


def _first_unverified_milestone(progress_pct: float) -> str:
    for threshold, milestone in (
        (20.0, "approach"),
        (40.0, "grasp"),
        (60.0, "lift"),
        (80.0, "transport"),
        (100.0, "placement"),
    ):
        if progress_pct < threshold:
            return milestone
    return "none"


def _verdict_markdown(model: RobustnessVisualization) -> str:
    base_progress = model.base_task_progress_pct[-1]
    perturbed_progress = model.perturbed_task_progress_pct[-1]
    milestone = _first_unverified_milestone(perturbed_progress).upper()
    return (
        f"**NOMINAL** &nbsp; PASS · {base_progress:.0f}%"
        " &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; "
        f"**PERTURBED +5 CM PERCEPTION BIAS** &nbsp; BLOCK · {perturbed_progress:.0f}%"
        " &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; "
        f"**FIRST UNVERIFIED MILESTONE** &nbsp; {milestone}"
    )


def _log_subtask_segments(recording, rr, model: RobustnessVisualization) -> None:
    centers = []
    half_sizes = []
    labels = []
    colors = []
    for subtask, start_s, end_s in _shared_subtask_segments(model):
        centers.append(((start_s + end_s) / 2.0, 0.0))
        half_sizes.append(((end_s - start_s) / 2.0, 0.13))
        labels.append(subtask.upper())
        colors.append(_SUBTASK_COLORS.get(subtask, [100, 108, 116, 220]))
    recording.log(
        "/subtask_timeline/segments",
        rr.Boxes2D(
            centers=centers,
            half_sizes=half_sizes,
            labels=labels,
            colors=colors,
            show_labels=True,
        ),
        static=True,
    )


def render_robustness_rerun(
    model: RobustnessVisualization,
    output_path: Path,
) -> Path:
    """Write the approved full-episode nominal-versus-perturbed dashboard."""

    rr, rrb, rrbc = _rerun_modules()
    output = output_path.resolve()
    if not output.is_relative_to(model.check_dir.resolve()):
        raise ValueError(f"Rerun output is not confined to the result directory: {output}")
    series = (
        model.base_target_distances_m,
        model.perturbed_target_distances_m,
        model.base_object_heights_m,
        model.perturbed_object_heights_m,
        model.base_task_success,
        model.perturbed_task_success,
        model.base_subtasks,
        model.perturbed_subtasks,
        model.base_task_progress_pct,
        model.perturbed_task_progress_pct,
    )
    if any(len(values) != len(model.times_s) for values in series):
        raise ValueError("robustness visualization signals are not frame-aligned")
    output.parent.mkdir(parents=True, exist_ok=True)

    with rr.RecordingStream(
        "robolens_robustness_gate",
        recording_id=f"{model.check_id}-{model.perturbed_case_id}",
        send_properties=False,
    ) as recording:
        recording.save(output, default_blueprint=_blueprint(rrb, rrbc, model))
        recording.log(
            "/verdict",
            rr.TextDocument(
                _verdict_markdown(model),
                media_type="text/markdown",
            ),
            static=True,
        )
        recording.log(
            "/base_case/video",
            rr.AssetVideo(path=model.base_video),
            static=True,
        )
        recording.log(
            "/perturbed/video",
            rr.AssetVideo(path=model.perturbed_video),
            static=True,
        )
        _log_styles(recording, rr)
        _log_subtask_segments(recording, rr, model)
        for index, time_s in enumerate(model.times_s):
            recording.set_time(model.timeline, duration=time_s)
            recording.log(
                "/base_case/frame",
                rr.VideoFrameReference(
                    seconds=time_s,
                    video_reference="/base_case/video",
                ),
            )
            recording.log(
                "/perturbed/frame",
                rr.VideoFrameReference(
                    seconds=time_s,
                    video_reference="/perturbed/video",
                ),
            )
            recording.log(
                "/exact_state/target_distance/base_case",
                rr.Scalars(model.base_target_distances_m[index]),
            )
            recording.log(
                "/exact_state/target_distance/perturbed",
                rr.Scalars(model.perturbed_target_distances_m[index]),
            )
            recording.log(
                "/exact_state/target_distance/threshold",
                rr.Scalars(model.target_distance_threshold_m),
            )
            recording.log(
                "/exact_state/object_height/base_case",
                rr.Scalars(model.base_object_heights_m[index]),
            )
            recording.log(
                "/exact_state/object_height/perturbed",
                rr.Scalars(model.perturbed_object_heights_m[index]),
            )
            recording.log(
                "/exact_state/task_success/base_case",
                rr.Scalars(float(model.base_task_success[index])),
            )
            recording.log(
                "/exact_state/task_success/perturbed",
                rr.Scalars(float(model.perturbed_task_success[index])),
            )
            recording.log(
                "/task_progress/base_case",
                rr.Scalars(model.base_task_progress_pct[index]),
            )
            recording.log(
                "/task_progress/perturbed",
                rr.Scalars(model.perturbed_task_progress_pct[index]),
            )
            recording.log(
                "/subtask_timeline/playhead",
                rr.LineStrips2D(
                    [[(time_s, -0.18), (time_s, 0.18)]],
                    radii=0.008,
                    colors=[235, 241, 238, 230],
                ),
            )

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Rerun did not produce a recording: {output}")
    return output
