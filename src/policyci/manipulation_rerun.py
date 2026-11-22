"""Optional Rerun renderer for synchronized manipulation evidence."""

from __future__ import annotations

from pathlib import Path

from policyci.manipulation_visualization import ManipulationVisualization


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


def _blueprint(rrb, rrbc, model: ManipulationVisualization):
    if model.task_kind == "placement":
        before_name = "BEFORE — calibration error misses the can"
        after_name = "AFTER — grasp, transport, and place in target bin"
        primary_origin = "/exact_state/target_distance"
        primary_name = "Object-to-target XY distance (m)"
        secondary_origin = "/exact_state/task_success"
        secondary_name = "Exact placement success"
    else:
        before_name = "BEFORE — calibration offset misses the cube"
        after_name = "AFTER — fixed action frame grasps and lifts"
        primary_origin = "/exact_state/object_height"
        primary_name = "Exact object lift height (m)"
        secondary_origin = "/exact_state/bilateral_contact"
        secondary_name = "Bilateral finger contact + failure window"
    metric_views = rrb.Vertical(
        rrb.TimeSeriesView(
            origin=primary_origin,
            name=primary_name,
        ),
        rrb.TimeSeriesView(
            origin=secondary_origin,
            name=secondary_name,
        ),
        row_shares=[2, 1],
    )
    sensor_evidence = model.baseline_sensors is not None
    if sensor_evidence:
        chart_row = rrb.Horizontal(
            metric_views,
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin="/sensors/force",
                    name="Fingertip normal + wrist force (N)",
                ),
                rrb.TimeSeriesView(
                    origin="/sensors/gripper",
                    name="Gripper width (cm) + command",
                ),
                row_shares=[1, 1],
            ),
            column_shares=[1, 1],
        )
        dashboard_row_shares = [1, 20, 4]
    else:
        chart_row = metric_views
        dashboard_row_shares = [1, 12, 2]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.TextDocumentView(
                origin="/summary",
                name="Automatic verdict",
            ),
            rrb.Horizontal(
                rrb.Spatial2DView(
                    origin="/baseline",
                    name=before_name,
                    background=[12, 16, 24],
                ),
                rrb.Spatial2DView(
                    origin="/candidate",
                    name=after_name,
                    background=[12, 16, 24],
                ),
                column_shares=[1, 1],
            ),
            chart_row,
            row_shares=dashboard_row_shares,
        ),
        rrb.TimePanel(
            state="collapsed",
            timeline=model.timeline,
            playback_speed=1.0,
            play_state=rrbc.PlayState.Playing,
            loop_mode=rrbc.LoopMode.Selection,
            time_selection=rrbc.AbsoluteTimeRange(
                min=int(model.failure_start_s * 1_000_000_000),
                max=int(model.failure_end_s * 1_000_000_000),
            ),
        ),
        collapse_panels=True,
    )


def _log_styles(recording, rr, model: ManipulationVisualization) -> None:
    if model.task_kind == "placement":
        styles = {
            "/exact_state/target_distance/baseline": (
                [255, 92, 105],
                "Baseline distance",
            ),
            "/exact_state/target_distance/candidate": (
                [57, 217, 138],
                "Candidate distance",
            ),
            "/exact_state/task_success/baseline": (
                [255, 92, 105],
                "Baseline success",
            ),
            "/exact_state/task_success/candidate": (
                [57, 217, 138],
                "Candidate success",
            ),
        }
    else:
        styles = {
            "/exact_state/object_height/baseline": ([255, 92, 105], "Baseline lift"),
            "/exact_state/object_height/candidate": ([57, 217, 138], "Candidate lift"),
            "/exact_state/object_height/threshold": ([244, 189, 80], "Lift threshold"),
            "/exact_state/bilateral_contact/baseline": (
                [255, 92, 105],
                "Baseline contact",
            ),
            "/exact_state/bilateral_contact/candidate": (
                [57, 217, 138],
                "Candidate contact",
            ),
            "/exact_state/bilateral_contact/failure_window": (
                [244, 93, 93],
                "Failure window",
            ),
        }
    if model.baseline_sensors is not None:
        styles.update(
            {
                "/sensors/force/left_normal/baseline": (
                    [255, 92, 105],
                    "Baseline left normal force",
                ),
                "/sensors/force/left_normal/candidate": (
                    [57, 217, 138],
                    "Candidate left normal force",
                ),
                "/sensors/force/right_normal/baseline": (
                    [255, 158, 168],
                    "Baseline right normal force",
                ),
                "/sensors/force/right_normal/candidate": (
                    [135, 240, 190],
                    "Candidate right normal force",
                ),
                "/sensors/force/wrist/baseline": (
                    [244, 189, 80],
                    "Baseline wrist force",
                ),
                "/sensors/force/wrist/candidate": (
                    [100, 170, 255],
                    "Candidate wrist force",
                ),
                "/sensors/gripper/width_cm/baseline": (
                    [255, 92, 105],
                    "Baseline width (cm)",
                ),
                "/sensors/gripper/width_cm/candidate": (
                    [57, 217, 138],
                    "Candidate width (cm)",
                ),
                "/sensors/gripper/command/baseline": (
                    [244, 189, 80],
                    "Baseline command",
                ),
                "/sensors/gripper/command/candidate": (
                    [100, 170, 255],
                    "Candidate command",
                ),
            }
        )
    for entity, (color, name) in styles.items():
        recording.log(
            entity,
            rr.SeriesLines(colors=color, names=name),
            static=True,
        )


def render_manipulation_rerun(
    model: ManipulationVisualization,
    output_path: Path,
) -> Path:
    """Write a durable, video-first manipulation dashboard."""
    rr, rrb, rrbc = _rerun_modules()
    output = output_path.resolve()
    if not output.is_relative_to(model.check_dir.resolve()):
        raise ValueError(f"Rerun output is not confined to the check directory: {output}")
    if (model.baseline_sensors is None) != (model.candidate_sensors is None):
        raise ValueError("baseline and candidate sensor evidence availability differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    recording_id = f"{model.check_id}-{model.case_id}-{model.unit_id}"

    with rr.RecordingStream(
        "robolens_manipulation_check",
        recording_id=recording_id,
        send_properties=False,
    ) as recording:
        recording.save(output, default_blueprint=_blueprint(rrb, rrbc, model))
        recording.log(
            "/summary",
            rr.TextDocument(model.summary_markdown, media_type="text/markdown"),
            static=True,
        )
        recording.log(
            "/baseline/video",
            rr.AssetVideo(path=model.baseline_video),
            static=True,
        )
        recording.log(
            "/candidate/video",
            rr.AssetVideo(path=model.candidate_video),
            static=True,
        )
        _log_styles(recording, rr, model)
        for index, time_s in enumerate(model.times_s):
            recording.set_time(model.timeline, duration=time_s)
            recording.log(
                "/baseline/frame",
                rr.VideoFrameReference(
                    seconds=time_s,
                    video_reference="/baseline/video",
                ),
            )
            recording.log(
                "/candidate/frame",
                rr.VideoFrameReference(
                    seconds=time_s,
                    video_reference="/candidate/video",
                ),
            )
            if model.task_kind == "placement":
                recording.log(
                    "/exact_state/target_distance/baseline",
                    rr.Scalars(model.baseline_target_distances_m[index]),
                )
                recording.log(
                    "/exact_state/target_distance/candidate",
                    rr.Scalars(model.candidate_target_distances_m[index]),
                )
                recording.log(
                    "/exact_state/task_success/baseline",
                    rr.Scalars(float(model.baseline_task_success[index])),
                )
                recording.log(
                    "/exact_state/task_success/candidate",
                    rr.Scalars(float(model.candidate_task_success[index])),
                )
            else:
                recording.log(
                    "/exact_state/object_height/baseline",
                    rr.Scalars(model.baseline_heights_m[index]),
                )
                recording.log(
                    "/exact_state/object_height/candidate",
                    rr.Scalars(model.candidate_heights_m[index]),
                )
                recording.log(
                    "/exact_state/object_height/threshold",
                    rr.Scalars(model.lift_threshold_m),
                )
                recording.log(
                    "/exact_state/bilateral_contact/baseline",
                    rr.Scalars(float(model.baseline_bilateral_contact[index])),
                )
                recording.log(
                    "/exact_state/bilateral_contact/candidate",
                    rr.Scalars(float(model.candidate_bilateral_contact[index])),
                )
                in_failure_window = model.failure_start_s <= time_s <= model.failure_end_s
                recording.log(
                    "/exact_state/bilateral_contact/failure_window",
                    rr.Scalars(float(in_failure_window)),
                )
            if model.baseline_sensors is not None and model.candidate_sensors is not None:
                baseline_sensors = model.baseline_sensors
                candidate_sensors = model.candidate_sensors
                recording.log(
                    "/sensors/force/left_normal/baseline",
                    rr.Scalars(baseline_sensors.left_normal_force_n[index]),
                )
                recording.log(
                    "/sensors/force/left_normal/candidate",
                    rr.Scalars(candidate_sensors.left_normal_force_n[index]),
                )
                recording.log(
                    "/sensors/force/right_normal/baseline",
                    rr.Scalars(baseline_sensors.right_normal_force_n[index]),
                )
                recording.log(
                    "/sensors/force/right_normal/candidate",
                    rr.Scalars(candidate_sensors.right_normal_force_n[index]),
                )
                recording.log(
                    "/sensors/force/wrist/baseline",
                    rr.Scalars(baseline_sensors.wrist_force_magnitude_n[index]),
                )
                recording.log(
                    "/sensors/force/wrist/candidate",
                    rr.Scalars(candidate_sensors.wrist_force_magnitude_n[index]),
                )
                recording.log(
                    "/sensors/gripper/width_cm/baseline",
                    rr.Scalars(baseline_sensors.gripper_width_m[index] * 100.0),
                )
                recording.log(
                    "/sensors/gripper/width_cm/candidate",
                    rr.Scalars(candidate_sensors.gripper_width_m[index] * 100.0),
                )
                recording.log(
                    "/sensors/gripper/command/baseline",
                    rr.Scalars(baseline_sensors.gripper_command[index]),
                )
                recording.log(
                    "/sensors/gripper/command/candidate",
                    rr.Scalars(candidate_sensors.gripper_command[index]),
                )

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Rerun did not produce a recording: {output}")
    return output
