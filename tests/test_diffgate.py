"""TDD tests for diff/gate engine."""

from __future__ import annotations

from policyci.diffgate import diff_runs, gate_exit_code
from policyci.schemas import GateVerdict, RunManifest, TaskResult, TaskStatus


def _manifest(run_id: str, policy: str, tasks: list[tuple[str, float, int]]) -> RunManifest:
    task_results = [
        TaskResult(
            task=name,
            suite="libero_object",
            n_episodes=n,
            n_successes=int(round(pc / 100.0 * n)),
            pc_success=pc,
        )
        for name, pc, n in tasks
    ]
    return RunManifest(
        run_id=run_id,
        policy_id=policy,
        suite="libero_object",
        seed=1000,
        n_episodes=sum(n for _, _, n in tasks),
        status="succeeded",
        tasks=task_results,
        overall_pc_success=sum(t.pc_success for t in task_results) / max(len(task_results), 1),
    )


def test_green_no_regression():
    v1 = _manifest("r1", "pol/v1", [("task_a", 90.0, 10), ("task_b", 80.0, 10)])
    v2 = _manifest("r2", "pol/v2", [("task_a", 88.0, 10), ("task_b", 82.0, 10)])
    d = diff_runs(v1, v2, threshold_pp=10.0, min_episodes=5)
    assert d.verdict == GateVerdict.PASS
    assert d.exit_code == 0
    assert d.n_regressed == 0
    assert gate_exit_code(d) == 0


def test_red_regression():
    v1 = _manifest("r1", "pol/v1", [("task_a", 90.0, 10), ("task_b", 80.0, 10)])
    v2 = _manifest("r2", "pol/v2", [("task_a", 40.0, 10), ("task_b", 78.0, 10)])
    d = diff_runs(v1, v2, threshold_pp=10.0, min_episodes=5)
    assert d.verdict == GateVerdict.BLOCK
    assert d.exit_code == 1
    assert d.n_regressed == 1
    reg = [t for t in d.tasks if t.status == TaskStatus.REGRESSED][0]
    assert reg.task == "task_a"
    assert reg.delta_pp == -50.0


def test_fixed_task():
    v1 = _manifest("r1", "pol/v1", [("task_a", 40.0, 10)])
    v2 = _manifest("r2", "pol/v2", [("task_a", 90.0, 10)])
    d = diff_runs(v1, v2, threshold_pp=10.0, min_episodes=5)
    assert d.n_fixed == 1
    assert d.verdict == GateVerdict.PASS


def test_task_mismatch_added_removed():
    v1 = _manifest("r1", "pol/v1", [("task_a", 90.0, 10)])
    v2 = _manifest("r2", "pol/v2", [("task_b", 90.0, 10)])
    d = diff_runs(v1, v2)
    statuses = {t.task: t.status for t in d.tasks}
    assert statuses["task_a"] == TaskStatus.NOT_EVALUATED
    assert statuses["task_b"] == TaskStatus.NOT_EVALUATED
    assert d.verdict == GateVerdict.NO_VERDICT
    assert d.exit_code == 2


def test_threshold_edge_exactly_at_threshold_is_unchanged():
    v1 = _manifest("r1", "pol/v1", [("task_a", 90.0, 10)])
    v2 = _manifest("r2", "pol/v2", [("task_a", 80.0, 10)])  # exactly -10 pp
    d = diff_runs(v1, v2, threshold_pp=10.0, min_episodes=5)
    # delta < -threshold triggers regression; exactly -10 is NOT < -10
    assert d.n_regressed == 0
    assert d.tasks[0].status == TaskStatus.UNCHANGED


def test_insufficient_episodes_is_no_verdict():
    v1 = _manifest("r1", "pol/v1", [("task_a", 90.0, 2)])
    v2 = _manifest("r2", "pol/v2", [("task_a", 10.0, 2)])
    d = diff_runs(v1, v2, threshold_pp=10.0, min_episodes=5)
    assert d.n_regressed == 0
    assert d.tasks[0].status == TaskStatus.NOT_EVALUATED
    assert "insufficient episodes" in d.tasks[0].reason
    assert d.verdict == GateVerdict.NO_VERDICT
    assert d.exit_code == 2


def test_failed_run_cannot_pass():
    v1 = _manifest("r1", "pol/v1", [("task_a", 90.0, 10)])
    v2 = _manifest("r2", "pol/v2", [("task_a", 90.0, 10)])
    v2.status = "failed"
    v2.error = "worker exited"

    d = diff_runs(v1, v2)

    assert d.tasks[0].status == TaskStatus.NOT_EVALUATED
    assert "candidate run status=failed" in d.tasks[0].reason
    assert d.verdict == GateVerdict.NO_VERDICT


def test_regression_precedes_a_different_missing_case():
    v1 = _manifest(
        "r1",
        "pol/v1",
        [("task_a", 90.0, 10), ("task_b", 90.0, 10)],
    )
    v2 = _manifest("r2", "pol/v2", [("task_a", 20.0, 10)])

    d = diff_runs(v1, v2)

    assert d.n_regressed == 1
    assert any(t.status == TaskStatus.NOT_EVALUATED for t in d.tasks)
    assert d.verdict == GateVerdict.BLOCK
    assert d.exit_code == 1
