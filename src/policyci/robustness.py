"""Robustness profiles + failure clustering (v1 differentiator)."""

from __future__ import annotations

from collections import defaultdict

from policyci.schemas import (
    FailureCluster,
    RobustnessPoint,
    RobustnessProfile,
    RunManifest,
    TaskResult,
)


def _axis_level_groups(
    tasks: list[TaskResult],
) -> dict[tuple[str | None, str | None, int], list[TaskResult]]:
    """Group tasks by (axis, task_name|None for overall, level)."""
    groups: dict[tuple[str | None, str | None, int], list[TaskResult]] = defaultdict(list)
    for t in tasks:
        from policyci.suite import normalize_axis

        axis = normalize_axis(t.perturbation_axis)
        level = t.perturbation_level
        groups[(axis, t.task, level)].append(t)
        groups[(axis, None, level)].append(t)  # overall
    return groups


def _aggregate(tasks: list[TaskResult]) -> RobustnessPoint:
    n_ep = sum(t.n_episodes for t in tasks)
    n_ok = sum(t.n_successes for t in tasks)
    pc = (100.0 * n_ok / n_ep) if n_ep else 0.0
    # Prefer explicit level from first task
    level = tasks[0].perturbation_level if tasks else 0
    return RobustnessPoint(level=level, pc_success=pc, n_episodes=n_ep, n_successes=n_ok)


def build_robustness_profiles(
    manifest: RunManifest,
    *,
    threshold_pc: float = 50.0,
    per_task: bool = True,
) -> list[RobustnessProfile]:
    """Build success-vs-level curves per axis (and optionally per task).

    holds_until = highest level whose pc_success >= threshold_pc
    breaks_at   = first level whose pc_success < threshold_pc (None if never breaks)
    """
    groups = _axis_level_groups(manifest.tasks)
    # Collect axes and tasks
    axes = sorted({a for a, _, _ in groups.keys() if a and a != "nominal"})
    profiles: list[RobustnessProfile] = []

    scopes: list[str | None] = [None]
    if per_task:
        scopes.extend(sorted({t.task for t in manifest.tasks}))

    for axis in axes:
        for task_scope in scopes:
            points: list[RobustnessPoint] = []
            levels = sorted({lvl for a, t, lvl in groups if a == axis and t == task_scope})
            # Include nominal (level 0) from nominal axis if present
            nominal_key = ("nominal", task_scope, 0)
            if nominal_key in groups:
                points.append(_aggregate(groups[nominal_key]))
            for lvl in levels:
                key = (axis, task_scope, lvl)
                if key in groups:
                    points.append(_aggregate(groups[key]))
            if not points:
                continue
            points.sort(key=lambda p: p.level)

            holds_until: int | None = None
            breaks_at: int | None = None
            for p in points:
                if p.pc_success >= threshold_pc:
                    holds_until = p.level
                elif breaks_at is None:
                    breaks_at = p.level

            profiles.append(
                RobustnessProfile(
                    axis=axis,
                    task=task_scope,
                    points=points,
                    holds_until=holds_until,
                    breaks_at=breaks_at,
                    threshold_pc=threshold_pc,
                )
            )
    return profiles


def cluster_failures(manifest: RunManifest) -> list[FailureCluster]:
    """Group failed episodes by axis × task × level (deterministic)."""
    buckets: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for ep in manifest.episodes:
        if ep.success:
            continue
        axis = ep.perturbation_axis or "nominal"
        level = ep.perturbation_level
        buckets[(axis, ep.task, level)].append(ep.episode_id)

    clusters: list[FailureCluster] = []
    for (axis, task, level), ids in sorted(buckets.items()):
        name = f"{axis} × {task} @ L{level}"
        clusters.append(
            FailureCluster(
                axis=axis,
                task=task,
                level=level,
                n_failures=len(ids),
                episode_ids=ids,
                name=name,
            )
        )
    # Largest clusters first
    clusters.sort(key=lambda c: (-c.n_failures, c.name))
    return clusters
