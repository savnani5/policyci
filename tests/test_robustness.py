"""Robustness profiles + failure clustering."""

from __future__ import annotations

from policyci.robustness import build_robustness_profiles, cluster_failures
from policyci.schemas import EpisodeRecord, RunManifest, TaskResult


def _sweep_manifest() -> RunManifest:
    """Synthetic cliff: nominal+L1 hold at 90%, L2+ collapse to 10%."""
    tasks = []
    episodes = []
    for axis in ("instruction", "pose"):
        for level in (0, 1, 2, 3):
            # level 0 stored as nominal
            ax = "nominal" if level == 0 else axis
            lvl = 0 if level == 0 else level
            # Only add nominal once
            if level == 0 and axis != "instruction":
                continue
            pc = 90.0 if level <= 1 else 10.0
            n_ep = 4
            n_ok = int(round(pc / 100.0 * n_ep))
            task = "libero_object/0"
            tasks.append(
                TaskResult(
                    task=task,
                    task_id=0,
                    suite="libero_object",
                    n_episodes=n_ep,
                    n_successes=n_ok,
                    pc_success=pc,
                    perturbation_axis=ax if level > 0 else "nominal",
                    perturbation_level=lvl,
                )
            )
            for e in range(n_ep):
                episodes.append(
                    EpisodeRecord(
                        episode_id=f"{ax}_L{lvl}_ep{e}",
                        task=task,
                        task_id=0,
                        suite="libero_object",
                        seed=1000 + e,
                        success=e < n_ok,
                        policy_id="test/policy",
                        perturbation_axis=ax if level > 0 else "nominal",
                        perturbation_level=lvl,
                    )
                )
    return RunManifest(
        run_id="sweep_test",
        policy_id="test/policy",
        suite="libero_object",
        seed=1000,
        n_episodes=sum(t.n_episodes for t in tasks),
        status="succeeded",
        tasks=tasks,
        episodes=episodes,
        overall_pc_success=50.0,
    )


def test_robustness_holds_until_and_breaks_at():
    m = _sweep_manifest()
    profiles = build_robustness_profiles(m, threshold_pc=50.0, per_task=False)
    by_axis = {p.axis: p for p in profiles if p.task is None}
    assert "instruction" in by_axis
    p = by_axis["instruction"]
    assert p.holds_until == 1
    assert p.breaks_at == 2


def test_failure_clusters_by_axis_task_level():
    m = _sweep_manifest()
    clusters = cluster_failures(m)
    assert clusters
    # Failures only at L2+
    assert all(c.level >= 2 or c.axis == "nominal" for c in clusters if c.n_failures > 0)
    top = clusters[0]
    assert top.n_failures >= 1
    assert "×" in top.name
