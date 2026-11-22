"""Two-version regression diff + deploy gate.

Pure functions over RunManifest — no I/O, no cloud. Fully unit-testable.
"""

from __future__ import annotations

from policyci.schemas import (
    DiffResult,
    GateVerdict,
    RunManifest,
    TaskDiff,
    TaskResult,
    TaskStatus,
)


def _task_key(t: TaskResult) -> str:
    """Stable key for matching tasks across runs."""
    from policyci.suite import normalize_axis

    axis = normalize_axis(t.perturbation_axis)
    level = t.perturbation_level
    return f"{t.suite}/{t.task}|{axis}|L{level}"


def _index_tasks(manifest: RunManifest) -> dict[str, TaskResult]:
    return {_task_key(t): t for t in manifest.tasks}


def diff_runs(
    v1: RunManifest,
    v2: RunManifest,
    *,
    threshold_pp: float = 10.0,
    min_episodes: int = 5,
) -> DiffResult:
    """Compare two manifests task-by-task.

    A task is REGRESSED when v2.pc_success drops by more than `threshold_pp`
    percentage points relative to v1, and both sides have >= `min_episodes`.
    Invalid or incomplete comparisons are NOT_EVALUATED and can never pass.
    """
    idx1 = _index_tasks(v1)
    idx2 = _index_tasks(v2)
    all_keys = sorted(set(idx1) | set(idx2))

    task_diffs: list[TaskDiff] = []
    for key in all_keys:
        t1 = idx1.get(key)
        t2 = idx2.get(key)

        run_reasons = []
        if v1.status != "succeeded":
            run_reasons.append(f"baseline run status={v1.status}")
        if v2.status != "succeeded":
            run_reasons.append(f"candidate run status={v2.status}")
        if run_reasons:
            task = t1 or t2
            assert task is not None
            task_diffs.append(
                TaskDiff(
                    task=task.task,
                    suite=task.suite,
                    status=TaskStatus.NOT_EVALUATED,
                    v1_pc_success=t1.pc_success if t1 else None,
                    v2_pc_success=t2.pc_success if t2 else None,
                    v1_n_episodes=t1.n_episodes if t1 else 0,
                    v2_n_episodes=t2.n_episodes if t2 else 0,
                    reason="; ".join(run_reasons),
                )
            )
            continue

        if t1 is None and t2 is not None:
            task_diffs.append(
                TaskDiff(
                    task=t2.task,
                    suite=t2.suite,
                    status=TaskStatus.NOT_EVALUATED,
                    v2_pc_success=t2.pc_success,
                    v2_n_episodes=t2.n_episodes,
                    reason="baseline task is missing",
                )
            )
            continue
        if t2 is None and t1 is not None:
            task_diffs.append(
                TaskDiff(
                    task=t1.task,
                    suite=t1.suite,
                    status=TaskStatus.NOT_EVALUATED,
                    v1_pc_success=t1.pc_success,
                    v1_n_episodes=t1.n_episodes,
                    reason="candidate task is missing",
                )
            )
            continue

        assert t1 is not None and t2 is not None
        delta = t2.pc_success - t1.pc_success
        enough = t1.n_episodes >= min_episodes and t2.n_episodes >= min_episodes

        reason = ""
        if not enough:
            status = TaskStatus.NOT_EVALUATED
            reason = (
                f"insufficient episodes: baseline={t1.n_episodes}, "
                f"candidate={t2.n_episodes}, required={min_episodes}"
            )
        elif delta < -threshold_pp:
            status = TaskStatus.REGRESSED
        elif delta > threshold_pp:
            status = TaskStatus.FIXED
        else:
            status = TaskStatus.UNCHANGED

        task_diffs.append(
            TaskDiff(
                task=t1.task,
                suite=t1.suite,
                status=status,
                v1_pc_success=t1.pc_success,
                v2_pc_success=t2.pc_success,
                delta_pp=delta,
                v1_n_episodes=t1.n_episodes,
                v2_n_episodes=t2.n_episodes,
                reason=reason,
            )
        )

    result = DiffResult(
        v1_run_id=v1.run_id,
        v2_run_id=v2.run_id,
        v1_policy_id=v1.policy_id,
        v2_policy_id=v2.policy_id,
        threshold_pp=threshold_pp,
        min_episodes=min_episodes,
        tasks=task_diffs,
    )
    result.recount()

    if result.n_regressed > 0:
        result.verdict = GateVerdict.BLOCK
        result.exit_code = 1
        names = [t.task for t in task_diffs if t.status == TaskStatus.REGRESSED]
        result.summary = (
            f"BLOCK: {result.n_regressed} regressed task(s): {', '.join(names)}. "
            f"({result.n_fixed} fixed, {result.n_unchanged} unchanged, "
            f"{result.n_not_evaluated} not evaluated)"
        )
    elif result.n_not_evaluated > 0:
        result.verdict = GateVerdict.NO_VERDICT
        result.exit_code = 2
        result.summary = (
            f"NO_VERDICT: {result.n_not_evaluated} task(s) could not be evaluated. "
            f"({result.n_fixed} fixed, {result.n_unchanged} unchanged)"
        )
    else:
        result.verdict = GateVerdict.PASS
        result.exit_code = 0
        result.summary = (
            f"PASS: no regressions "
            f"({result.n_fixed} fixed, {result.n_unchanged} unchanged)."
        )

    return result


def gate_exit_code(diff: DiffResult) -> int:
    return diff.exit_code
