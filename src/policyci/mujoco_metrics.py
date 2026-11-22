"""Metrics derived only from exact physics state, never from rendered pixels."""

from __future__ import annotations

import math

from policyci.schemas import MetricValue, SimulationTrace


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _longest_interval(times: list[float], active: list[bool]) -> float:
    longest = 0.0
    start: float | None = None
    previous_time = times[0]
    for time_s, enabled in zip(times, active):
        if enabled and start is None:
            start = time_s
        elif not enabled and start is not None:
            longest = max(longest, previous_time - start)
            start = None
        previous_time = time_s
    if start is not None:
        longest = max(longest, times[-1] - start)
    return longest


def derive_lift_metrics(
    trace: SimulationTrace,
    *,
    lift_threshold_m: float,
    hold_duration_s: float,
) -> dict[str, MetricValue]:
    """Derive a sustained bilateral-contact lift verdict from simulator state."""

    threshold = _positive("lift_threshold_m", lift_threshold_m)
    hold_duration = _positive("hold_duration_s", hold_duration_s)
    initial_z = trace.samples[0].object_position_m[2]
    times = [sample.time_s for sample in trace.samples]
    heights = [sample.object_position_m[2] - initial_z for sample in trace.samples]
    bilateral_contact = [
        sample.left_finger_contact and sample.right_finger_contact
        for sample in trace.samples
    ]
    qualified_lift = [
        height >= threshold and contact
        for height, contact in zip(heights, bilateral_contact)
    ]
    distances = [
        math.dist(sample.eef_position_m, sample.object_position_m)
        for sample in trace.samples
    ]
    qualified_duration = _longest_interval(times, qualified_lift)

    return {
        "lift_success": qualified_duration >= hold_duration,
        "max_object_lift_m": max(heights),
        "bilateral_contact_duration_s": _longest_interval(
            times,
            bilateral_contact,
        ),
        "min_eef_object_distance_m": min(distances),
    }


def derive_placement_metrics(
    trace: SimulationTrace,
    *,
    hold_duration_s: float,
) -> dict[str, MetricValue]:
    """Derive a sustained environment-success verdict from exact simulator state."""

    hold_duration = _positive("hold_duration_s", hold_duration_s)
    if any(
        sample.target_position_m is None or sample.task_success is None
        for sample in trace.samples
    ):
        raise ValueError(
            "placement metrics require target_position_m and task_success on every sample"
        )
    times = [sample.time_s for sample in trace.samples]
    success = [bool(sample.task_success) for sample in trace.samples]
    target_distances = [
        math.dist(sample.object_position_m[:2], sample.target_position_m[:2])
        for sample in trace.samples
        if sample.target_position_m is not None
    ]
    initial_z = trace.samples[0].object_position_m[2]
    heights = [
        sample.object_position_m[2] - initial_z for sample in trace.samples
    ]
    bilateral_contact = [
        sample.left_finger_contact and sample.right_finger_contact
        for sample in trace.samples
    ]
    success_duration = _longest_interval(times, success)
    return {
        "placement_success": success_duration >= hold_duration,
        "placement_success_duration_s": success_duration,
        "min_object_target_xy_distance_m": min(target_distances),
        "final_object_target_xy_distance_m": target_distances[-1],
        "max_object_lift_m": max(heights),
        "bilateral_contact_duration_s": _longest_interval(
            times,
            bilateral_contact,
        ),
    }


def derive_pick_place_progress(
    trace: SimulationTrace,
    *,
    placement_hold_duration_s: float,
    approach_distance_m: float = 0.07,
    lift_height_m: float = 0.05,
    transport_distance_m: float = 0.10,
) -> list[float]:
    """Return monotonic progress from ordered, exact-state milestones."""

    hold_duration = _positive(
        "placement_hold_duration_s",
        placement_hold_duration_s,
    )
    approach_distance = _positive("approach_distance_m", approach_distance_m)
    lift_height = _positive("lift_height_m", lift_height_m)
    transport_distance = _positive("transport_distance_m", transport_distance_m)
    if any(
        sample.target_position_m is None or sample.task_success is None
        for sample in trace.samples
    ):
        raise ValueError(
            "pick-place progress requires target_position_m and task_success on every sample"
        )

    initial_z = trace.samples[0].object_position_m[2]
    progress = 0.0
    success_start_s: float | None = None
    values: list[float] = []
    for sample in trace.samples:
        eef_object_distance = math.dist(
            sample.eef_position_m,
            sample.object_position_m,
        )
        bilateral_contact = (
            sample.left_finger_contact and sample.right_finger_contact
        )
        object_lift_m = sample.object_position_m[2] - initial_z
        target_distance_m = math.dist(
            sample.object_position_m[:2],
            sample.target_position_m[:2],
        )

        if progress < 20.0 and eef_object_distance <= approach_distance:
            progress = 20.0
        if progress >= 20.0 and progress < 40.0 and bilateral_contact:
            progress = 40.0
        if progress >= 40.0 and progress < 60.0 and object_lift_m >= lift_height:
            progress = 60.0
        if (
            progress >= 60.0
            and progress < 80.0
            and object_lift_m >= lift_height
            and target_distance_m <= transport_distance
        ):
            progress = 80.0

        if sample.task_success:
            if success_start_s is None:
                success_start_s = sample.time_s
            if (
                progress >= 80.0
                and sample.time_s - success_start_s >= hold_duration
            ):
                progress = 100.0
        else:
            success_start_s = None
        values.append(progress)
    return values


def failure_window(
    trace: SimulationTrace,
    *,
    lift_threshold_m: float,
    window_duration_s: float,
) -> tuple[float, float]:
    """Return a compact interval around the first visibly failed grasp attempt."""

    threshold = _positive("lift_threshold_m", lift_threshold_m)
    duration = _positive("window_duration_s", window_duration_s)
    initial_z = trace.samples[0].object_position_m[2]
    trace_start = trace.samples[0].time_s
    trace_end = trace.samples[-1].time_s

    attempt_start: float | None = None
    for sample in trace.samples:
        lift = sample.object_position_m[2] - initial_z
        if (
            sample.left_finger_contact
            and sample.right_finger_contact
            and lift < threshold
        ):
            attempt_start = sample.time_s
            break

    if attempt_start is None:
        return (max(trace_start, trace_end - duration), trace_end)

    return (attempt_start, min(trace_end, attempt_start + duration))
