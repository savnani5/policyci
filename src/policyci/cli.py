"""Typer CLI for RoboLens release checks and legacy PolicyCI workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

from policyci import __version__
from policyci.diffgate import diff_runs
from policyci.report.html import render_diff_report, render_run_report
from policyci.schemas import EpisodeRecord, RunManifest, TaskResult
from policyci.store import (
    DEFAULT_RUNS_DIR,
    load_diff,
    load_run,
    load_run_from_path,
    new_run_id,
    save_diff,
    save_run,
)

app = typer.Typer(
    name="robolens",
    help="Compare a baseline and candidate on protected robot behaviors before release.",
    no_args_is_help=True,
)


def _build_check_runner(suite):
    """Select a built-in runner without a plugin framework."""
    if suite.runner.type == "libero":
        from policyci.runner.libero import LiberoRunner

        return LiberoRunner(suite)
    if suite.runner.type == "ros2_replay":
        from policyci.runner.ros2_replay import Ros2ReplayRunner

        runner = Ros2ReplayRunner(suite)
        # Adapter imports are configuration, not an execution-time soundness issue.
        runner.adapter
        return runner
    if suite.runner.type == "mujoco":
        from policyci.runner.mujoco import MujocoRunner

        return MujocoRunner(suite)
    raise ValueError(f"unsupported runner: {suite.runner.type}")


def _parse_task_ids(raw: Optional[str]) -> list[int]:
    if not raw:
        return [0]
    return [int(x.strip()) for x in raw.split(",") if x.strip() != ""]


def _dry_run_from_scenarios(
    policy: str,
    scenarios: list,
    *,
    tag: str,
    suite_name: str,
    regress_first: bool = False,
) -> RunManifest:
    """Synthetic manifest for offline demos. Optionally tank success on first scenario."""
    tasks: list[TaskResult] = []
    episodes: list[EpisodeRecord] = []
    for i, sc in enumerate(scenarios):
        if regress_first and i == 0:
            pc = 40.0
        elif sc.axis in (None, "nominal") or sc.level <= 1:
            pc = 90.0
        else:
            pc = 10.0
        n_ok = int(round(pc / 100.0 * sc.n_episodes))
        task_name = f"{sc.suite}/{sc.task_id}"
        tasks.append(
            TaskResult(
                task=task_name,
                task_id=sc.task_id,
                suite=sc.suite,
                n_episodes=sc.n_episodes,
                n_successes=n_ok,
                pc_success=pc,
                perturbation_axis=sc.axis,
                perturbation_level=sc.level,
            )
        )
        for e in range(sc.n_episodes):
            episodes.append(
                EpisodeRecord(
                    episode_id=f"{tag}_{i}_ep{e}",
                    task=task_name,
                    task_id=sc.task_id,
                    suite=sc.suite,
                    instruction=sc.instruction or "",
                    seed=sc.seed + e,
                    success=e < n_ok,
                    policy_id=policy,
                    perturbation_axis=sc.axis,
                    perturbation_level=sc.level,
                    perturbation_params=sc.params,
                )
            )
    overall = sum(t.pc_success * t.n_episodes for t in tasks) / max(
        sum(t.n_episodes for t in tasks), 1
    )
    return RunManifest(
        run_id=new_run_id(tag),
        policy_id=policy,
        suite=scenarios[0].suite if scenarios else "libero_object",
        seed=scenarios[0].seed if scenarios else 0,
        n_episodes=sum(s.n_episodes for s in scenarios),
        status="succeeded",
        gpu_seconds=float(len(scenarios)) * 8.0,
        gpu_type="dry-run",
        tasks=tasks,
        overall_pc_success=overall,
        episodes=episodes,
        suite_name=suite_name,
        notes="dry-run",
    )


@app.command()
def version() -> None:
    """Print version."""
    rprint(__version__)


@app.command("check")
def check_cmd(
    suite_path: Path = typer.Option(
        ...,
        "--suite",
        help="Protected-case suite YAML",
    ),
    baseline_path: Path = typer.Option(
        ...,
        "--baseline",
        help="Baseline subject YAML",
    ),
    candidate_path: Path = typer.Option(
        ...,
        "--candidate",
        help="Candidate subject YAML",
    ),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir"),
) -> None:
    """Run baseline and candidate on the same cases, then emit a release verdict."""
    try:
        from policyci.check import execute_check
        from policyci.checkspec import load_check_suite, load_subject

        suite = load_check_suite(suite_path)
        baseline = load_subject(baseline_path)
        candidate = load_subject(candidate_path)
        runner = _build_check_runner(suite)
        result, root = execute_check(
            suite,
            baseline,
            candidate,
            runner,
            runs_root=runs_dir,
        )
    except Exception as exc:
        rprint(f"[red]configuration error[/red]: {exc}")
        raise typer.Exit(code=3) from exc

    evidence_path = root / "evidence.html"
    try:
        from policyci.report.html import render_check_report

        render_check_report(result, check_dir=root, output_path=evidence_path)
    except Exception as exc:
        rprint(f"[yellow]evidence report error[/yellow]: {exc}")

    color = {
        "pass": "green",
        "block": "red",
        "no_verdict": "yellow",
    }[result.gate.verdict.value]
    rprint(f"[{color}]{result.gate.verdict.value.upper()}[/{color}] — {result.gate.summary}")
    rprint(f"result={root / 'result.json'}")
    if evidence_path.exists():
        rprint(f"evidence={evidence_path}")
    raise typer.Exit(code=result.gate.exit_code)


@app.command("visualize")
def visualize_cmd(
    check_dir: Path = typer.Argument(
        ...,
        help="Completed RoboLens check directory containing result.json",
    ),
    case_id: Optional[str] = typer.Option(
        None,
        "--case",
        help="Protected case to visualize; defaults to the first changed case",
    ),
    repeat_id: Optional[str] = typer.Option(
        None,
        "--repeat",
        help="Evidence repeat to visualize; defaults to the first pair",
    ),
    no_open: bool = typer.Option(
        False,
        "--no-open",
        help="Generate and verify inputs without launching the native viewer",
    ),
) -> None:
    """Open paired robot evidence, automatic metrics, and the gate in Rerun."""
    try:
        from policyci.rerun_visualizer import (
            open_visualization,
            render_visualization,
        )
        from policyci.visualization import load_visualization

        model = load_visualization(
            check_dir,
            case_id=case_id,
            unit_id=repeat_id,
        )
        output = (
            model.check_dir
            / "visualizations"
            / model.case_id
            / f"{model.unit_id}.rrd"
        )
        rendered = render_visualization(model, output)
        rprint(
            f"[{model.verdict.value.upper()}] {model.gate_summary}\n"
            f"comparison={model.comparison_status.value.upper()} "
            f"case={model.case_id} / {model.unit_id}\n"
            f"aggregate_metric={model.metric} baseline={model.baseline_value} "
            f"candidate={model.candidate_value}\n"
            f"selected_repeat=baseline_"
            f"{'PASS' if model.baseline_unit_pass else 'FAIL'} "
            f"candidate_"
            f"{'PASS' if model.candidate_unit_pass else 'FAIL'}\n"
            f"rerun={rendered}"
        )
        if not no_open:
            open_visualization(rendered, model)
    except Exception as exc:
        rprint(f"[red]visualization error[/red]: {exc}")
        raise typer.Exit(code=3) from exc


@app.command("robustness-check")
def robustness_check_cmd(
    suite_path: Path = typer.Option(
        ...,
        "--suite",
        help="Absolute simulation scenario suite YAML",
    ),
    subject_path: Path = typer.Option(
        ...,
        "--subject",
        help="Single controller or policy subject YAML",
    ),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir"),
) -> None:
    """Run one subject over protected simulation scenarios and gate absolutely."""

    try:
        from policyci.checkspec import load_check_suite, load_subject
        from policyci.robustness_check import execute_robustness_check

        suite = load_check_suite(suite_path)
        subject = load_subject(subject_path)
        runner = _build_check_runner(suite)
        result, root = execute_robustness_check(
            suite,
            subject,
            runner,
            runs_root=runs_dir,
        )
    except Exception as exc:
        rprint(f"[red]configuration error[/red]: {exc}")
        raise typer.Exit(code=3) from exc

    for case in result.cases:
        if case.validity.value != "valid":
            status = "NO_VERDICT"
            color = "yellow"
        elif case.passed:
            status = "PASS"
            color = "green"
        else:
            status = "FAIL"
            color = "red"
        offset = case.object_position_offset_m
        bias = case.object_observation_bias_m
        rprint(
            f"[{color}]{status}[/{color}] "
            f"{case.case_id} object_offset_m="
            f"[{offset[0]:+.3f}, {offset[1]:+.3f}, {offset[2]:+.3f}] "
            f"observation_bias_m="
            f"[{bias[0]:+.3f}, {bias[1]:+.3f}, {bias[2]:+.3f}] "
            f"{case.metric}={case.metric_value}"
        )

    if result.gate.verdict.value == "block":
        title = "ROBUSTNESS GATE BLOCKED"
        color = "red"
    elif result.gate.verdict.value == "pass":
        title = "ROBUSTNESS GATE PASSED"
        color = "green"
    else:
        title = "ROBUSTNESS GATE NO_VERDICT"
        color = "yellow"
    rprint(f"[{color}]{title}[/{color}]: {result.gate.summary}")
    rprint(f"result={root / 'result.json'}")
    raise typer.Exit(code=result.gate.exit_code)


@app.command("robustness-visualize")
def robustness_visualize_cmd(
    check_dir: Path = typer.Argument(
        ...,
        help="Completed robustness-check directory containing result.json",
    ),
    case_id: Optional[str] = typer.Option(
        None,
        "--case",
        help="Failed perturbed case; defaults to the first failed required case",
    ),
    no_open: bool = typer.Option(
        False,
        "--no-open",
        help="Generate the RRD without launching the native viewer",
    ),
) -> None:
    """Show the passing base case beside one failed simulation perturbation."""

    try:
        from policyci.rerun_visualizer import open_rerun_visualization
        from policyci.robustness_rerun import render_robustness_rerun
        from policyci.robustness_visualization import (
            load_robustness_visualization,
        )

        model = load_robustness_visualization(
            check_dir,
            failed_case_id=case_id,
        )
        output = (
            model.check_dir
            / "visualizations"
            / "robustness"
            / f"{model.perturbed_case_id}.rrd"
        )
        rendered = render_robustness_rerun(model, output)
        rprint(
            f"[red]ROBUSTNESS GATE BLOCKED[/red]: {model.gate_summary}\n"
            f"subject={model.subject_id} controller={model.controller_sha256[:12]}\n"
            f"base={model.base_case_id} PASS "
            f"perturbed={model.perturbed_case_id} FAIL\n"
            f"rerun={rendered}"
        )
        if not no_open:
            open_rerun_visualization(
                rendered,
                [],
                require_compatible_ffmpeg=True,
            )
    except Exception as exc:
        rprint(f"[red]visualization error[/red]: {exc}")
        raise typer.Exit(code=3) from exc


@app.command("run")
def run_cmd(
    policy: str = typer.Option(..., "--policy", "-p", help="Hub id or local checkpoint path"),
    suite: str = typer.Option("libero_object", "--suite", "-s"),
    episodes: int = typer.Option(10, "--episodes", "-n"),
    seed: int = typer.Option(1000, "--seed"),
    task_ids: Optional[str] = typer.Option("0", "--task-ids", help="Comma-separated task ids"),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip Modal; write a stub manifest"),
    pull: bool = typer.Option(True, "--pull/--no-pull", help="Download videos from Modal Volume"),
) -> None:
    """Eval one checkpoint on Modal → runs/<id>/."""
    ids = _parse_task_ids(task_ids)
    if dry_run:
        rid = new_run_id()
        manifest = RunManifest(
            run_id=rid,
            policy_id=policy,
            suite=suite,
            seed=seed,
            n_episodes=episodes,
            task_ids=ids,
            status="succeeded",
            gpu_seconds=0.0,
            tasks=[
                TaskResult(
                    task=f"{suite}/{tid}",
                    task_id=tid,
                    suite=suite,
                    n_episodes=episodes,
                    n_successes=episodes,
                    pc_success=100.0,
                )
                for tid in ids
            ],
            overall_pc_success=100.0,
            notes="dry-run stub",
        )
        path = save_run(manifest, runs_dir)
        rprint(f"[green]dry-run[/green] wrote {path}")
        return

    from policyci.runner.client import run_eval

    manifest = run_eval(
        policy,
        suite=suite,
        task_ids=ids,
        n_episodes=episodes,
        seed=seed,
        runs_root=runs_dir,
        pull_videos=pull,
    )
    n_vids = 0
    vdir = runs_dir / manifest.run_id / "videos"
    if vdir.exists():
        n_vids = len(list(vdir.rglob("*.mp4")))
    rprint(
        f"[green]ok[/green] run_id={manifest.run_id} "
        f"pc_success={manifest.overall_pc_success} "
        f"gpu_hours={manifest.gpu_hours:.3f} "
        f"videos={n_vids}"
    )


@app.command("eval")
def eval_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="Suite YAML (flat tests: or legacy)"),
    policy: str = typer.Option(..., "--policy", "-p", help="Hub id or local checkpoint path"),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir"),
    report: Path = typer.Option(Path("runs/eval_report.html"), "--report"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    pull: bool = typer.Option(True, "--pull/--no-pull"),
) -> None:
    """Run all scenarios in a suite YAML for one policy."""
    from policyci.suite import load_suite_spec

    spec = load_suite_spec(config)
    scenarios = spec.expand()
    rprint(f"suite={spec.name} format={spec.format} scenarios={len(scenarios)}")

    if dry_run:
        manifest = _dry_run_from_scenarios(policy, scenarios, tag="eval", suite_name=spec.name)
        save_run(manifest, runs_dir)
    else:
        from policyci.runner.client import run_eval_fanout

        shards = [sc.to_shard(policy) for sc in scenarios]
        manifest = run_eval_fanout(policy, shards, runs_root=runs_dir, pull_videos=pull)

    report.parent.mkdir(parents=True, exist_ok=True)
    render_run_report(manifest, output_path=report, runs_root=runs_dir)
    rprint(
        f"[green]ok[/green] run_id={manifest.run_id} "
        f"pc_success={manifest.overall_pc_success} report={report}"
    )


@app.command("pull")
def pull_cmd(
    run_id: str = typer.Argument(..., help="Run id on the Modal Volume (e.g. derisk_day1)"),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir"),
    report: Optional[Path] = typer.Option(None, "--report", help="Also write HTML report"),
) -> None:
    """Download a run's JSON + mp4s from the Modal Volume into runs/<id>/."""
    from policyci.runner.client import ensure_local_manifest, pull_volume_run

    dest = pull_volume_run(run_id, runs_dir)
    vids = list((dest / "videos").rglob("*.mp4")) if (dest / "videos").exists() else []
    rprint(f"[green]pulled[/green] {dest} videos={len(vids)}")
    for v in vids[:10]:
        rprint(f"  {v}")

    manifest = ensure_local_manifest(run_id, runs_dir)
    if report:
        render_run_report(manifest, output_path=report, runs_root=runs_dir)
        rprint(f"report {report}")


@app.command("diff")
def diff_cmd(
    v1: Path = typer.Argument(..., help="Path to v1 run dir (or run_id under --runs-dir)"),
    v2: Path = typer.Argument(..., help="Path to v2 run dir (or run_id under --runs-dir)"),
    threshold: float = typer.Option(10.0, "--threshold", help="Regression threshold in pp"),
    min_episodes: int = typer.Option(5, "--min-episodes"),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir"),
    report: Optional[Path] = typer.Option(None, "--report", help="Write HTML report here"),
) -> None:
    """Diff two runs → gate verdict + exit code."""
    m1 = _resolve_run(v1, runs_dir)
    m2 = _resolve_run(v2, runs_dir)
    result = diff_runs(m1, m2, threshold_pp=threshold, min_episodes=min_episodes)
    diff_dir = save_diff(result, runs_dir)
    rprint(result.summary)
    rprint(f"saved {diff_dir / 'diff.json'}")
    if report:
        render_diff_report(result, v1=m1, v2=m2, output_path=report, runs_root=runs_dir)
        rprint(f"report {report}")
    raise typer.Exit(code=result.exit_code)


@app.command("report")
def report_cmd(
    target: Path = typer.Argument(..., help="Run dir, diff dir, or manifest/diff json"),
    output: Path = typer.Option(Path("report.html"), "--output", "-o"),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir"),
) -> None:
    """Render HTML report from a run or diff."""
    p = target
    if not p.exists() and (runs_dir / target).exists():
        p = runs_dir / target

    if p.is_dir() and (p / "diff.json").exists():
        diff = load_diff(p)
        v1 = load_run(diff.v1_run_id, runs_dir) if (runs_dir / diff.v1_run_id).exists() else None
        v2 = load_run(diff.v2_run_id, runs_dir) if (runs_dir / diff.v2_run_id).exists() else None
        render_diff_report(diff, v1=v1, v2=v2, output_path=output, runs_root=runs_dir)
    elif p.is_file() and p.name == "diff.json":
        diff = load_diff(p)
        render_diff_report(diff, output_path=output)
    else:
        manifest = load_run_from_path(p) if p.exists() else load_run(str(target), runs_dir)
        render_run_report(manifest, output_path=output, runs_root=runs_dir)
    rprint(f"[green]wrote[/green] {output}")


@app.command("ci")
def ci_cmd(
    v1: str = typer.Option(..., "--v1", help="Baseline policy (Hub id or path)"),
    v2: str = typer.Option(..., "--v2", help="Candidate policy (Hub id or path)"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Suite YAML (flat tests: preferred)"
    ),
    suite: str = typer.Option("libero_object", "--suite", "-s"),
    episodes: int = typer.Option(10, "--episodes", "-n"),
    seed: int = typer.Option(1000, "--seed"),
    task_ids: Optional[str] = typer.Option("0", "--task-ids"),
    threshold: Optional[float] = typer.Option(
        None, "--threshold", help="Regression threshold in pp (overrides suite gate:)"
    ),
    min_episodes: Optional[int] = typer.Option(
        None, "--min-episodes", help="Min episodes for gate (overrides suite gate:)"
    ),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir"),
    report: Path = typer.Option(Path("runs/ci_report.html"), "--report"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    pull: bool = typer.Option(True, "--pull/--no-pull"),
) -> None:
    """Run v1 + v2 → diff → gate → HTML report (the demo flow)."""
    scenarios = None
    suite_name = suite
    thr = 10.0 if threshold is None else threshold
    min_ep = 5 if min_episodes is None else min_episodes

    if config is not None:
        from policyci.suite import load_suite_spec

        spec = load_suite_spec(config)
        scenarios = spec.expand()
        suite_name = spec.name
        suite = spec.suite
        if threshold is None:
            thr = spec.gate.threshold_pp
        if min_episodes is None:
            min_ep = spec.gate.min_episodes
        rprint(
            f"suite={spec.name} format={spec.format} "
            f"scenarios={len(scenarios)} gate={thr}pp/{min_ep}ep"
        )

    ids = _parse_task_ids(task_ids)

    if dry_run:
        if scenarios is not None:
            m1 = _dry_run_from_scenarios(v1, scenarios, tag="v1", suite_name=suite_name)
            m2 = _dry_run_from_scenarios(
                v2, scenarios, tag="v2", suite_name=suite_name, regress_first=True
            )
        else:

            def _mk(policy: str, pcs: list[float], tag: str) -> RunManifest:
                rid = new_run_id(tag)
                tasks = []
                episode_records = []
                n_eps = episodes
                for i, (tid, pc) in enumerate(zip(ids, pcs)):
                    n_ok = int(round(pc / 100.0 * n_eps))
                    tasks.append(
                        TaskResult(
                            task=f"{suite}/{tid}",
                            task_id=tid,
                            suite=suite,
                            n_episodes=n_eps,
                            n_successes=n_ok,
                            pc_success=pc,
                        )
                    )
                    for e in range(n_eps):
                        episode_records.append(
                            EpisodeRecord(
                                episode_id=f"{tag}_{tid}_ep{e}",
                                task=f"{suite}/{tid}",
                                task_id=tid,
                                suite=suite,
                                seed=seed + e,
                                success=e < n_ok,
                                policy_id=policy,
                            )
                        )
                overall = sum(t.pc_success * t.n_episodes for t in tasks) / max(
                    sum(t.n_episodes for t in tasks), 1
                )
                return RunManifest(
                    run_id=rid,
                    policy_id=policy,
                    suite=suite,
                    seed=seed,
                    n_episodes=n_eps,
                    task_ids=ids,
                    status="succeeded",
                    gpu_seconds=12.0,
                    gpu_type="dry-run",
                    tasks=tasks,
                    overall_pc_success=overall,
                    episodes=episode_records,
                )

            pcs_v1 = [95.0] * len(ids)
            pcs_v2 = [40.0 if i == 0 else 90.0 for i in range(len(ids))]
            m1 = _mk(v1, pcs_v1, "v1")
            m2 = _mk(v2, pcs_v2, "v2")
        save_run(m1, runs_dir)
        save_run(m2, runs_dir)
    else:
        from policyci.runner.client import run_eval, run_eval_fanout

        if scenarios is not None:
            rprint(f"[cyan]eval v1[/cyan] {v1}")
            m1 = run_eval_fanout(
                v1,
                [sc.to_shard(v1) for sc in scenarios],
                runs_root=runs_dir,
                pull_videos=pull,
            )
            rprint(f"[cyan]eval v2[/cyan] {v2}")
            m2 = run_eval_fanout(
                v2,
                [sc.to_shard(v2) for sc in scenarios],
                runs_root=runs_dir,
                pull_videos=pull,
            )
        else:
            rprint(f"[cyan]eval v1[/cyan] {v1}")
            m1 = run_eval(
                v1,
                suite=suite,
                task_ids=ids,
                n_episodes=episodes,
                seed=seed,
                runs_root=runs_dir,
                pull_videos=pull,
            )
            rprint(f"[cyan]eval v2[/cyan] {v2}")
            m2 = run_eval(
                v2,
                suite=suite,
                task_ids=ids,
                n_episodes=episodes,
                seed=seed,
                runs_root=runs_dir,
                pull_videos=pull,
            )

    result = diff_runs(m1, m2, threshold_pp=thr, min_episodes=min_ep)
    diff_dir = save_diff(result, runs_dir)
    report.parent.mkdir(parents=True, exist_ok=True)
    render_diff_report(result, v1=m1, v2=m2, output_path=report, runs_root=runs_dir)
    rprint(result.summary)
    rprint(f"diff={diff_dir} report={report}")
    raise typer.Exit(code=result.exit_code)


@app.command("sweep")
def sweep_cmd(
    policy: str = typer.Option(..., "--policy", "-p"),
    suite_yaml: Path = typer.Option(..., "--suite-yaml", help="Path to suite YAML"),
    runs_dir: Path = typer.Option(DEFAULT_RUNS_DIR, "--runs-dir"),
    report: Path = typer.Option(Path("runs/sweep_report.html"), "--report"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Perturbation sweep from suite YAML (flat or legacy). Prefer `policyci eval -c`."""
    from policyci.suite import load_suite_spec

    spec = load_suite_spec(suite_yaml)
    scenarios = spec.expand()
    rprint(f"suite={spec.name} scenarios={len(scenarios)}")

    if dry_run:
        manifest = _dry_run_from_scenarios(policy, scenarios, tag="sweep", suite_name=spec.name)
        save_run(manifest, runs_dir)
    else:
        from policyci.runner.client import run_eval_fanout

        shards = [sc.to_shard(policy) for sc in scenarios]
        manifest = run_eval_fanout(policy, shards, runs_root=runs_dir)

    report.parent.mkdir(parents=True, exist_ok=True)
    render_run_report(manifest, output_path=report, runs_root=runs_dir)
    from policyci.robustness import build_robustness_profiles, cluster_failures

    profiles = build_robustness_profiles(manifest)
    clusters = cluster_failures(manifest)
    rprint(f"profiles={len(profiles)} clusters={len(clusters)} gpu_hours={manifest.gpu_hours:.3f}")
    for p in profiles:
        if p.task is None:
            rprint(
                f"  {p.axis} overall: holds_until=L{p.holds_until} breaks_at=L{p.breaks_at}"
            )
    if clusters:
        rprint(f"top cluster: {clusters[0].name} ({clusters[0].n_failures} failures)")
    rprint(f"report={report}")


def _resolve_run(path_or_id: Path, runs_dir: Path):
    if path_or_id.exists():
        return load_run_from_path(path_or_id)
    candidate = runs_dir / path_or_id
    if candidate.exists():
        return load_run_from_path(candidate)
    return load_run(str(path_or_id), runs_dir)


if __name__ == "__main__":
    app()
