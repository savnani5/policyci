"""Render HTML reports from RunManifest / DiffResult."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from policyci.schemas import CheckResult, DiffResult, RunManifest
from policyci.robustness import build_robustness_profiles, cluster_failures
from policyci.store import DEFAULT_RUNS_DIR, run_dir

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.1f}"


def _rel_to_report(video_file: Path, report_path: Path | None) -> str:
    if report_path is None:
        return str(video_file)
    try:
        return os.path.relpath(video_file.resolve(), report_path.parent.resolve())
    except ValueError:
        return str(video_file)


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(_flatten(item, path))
        else:
            flattened[path] = item
    return flattened


def _display_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list, bool)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _check_artifact_src(path: str, check_dir: Path, report_path: Path) -> str:
    return _rel_to_report(check_dir / path, report_path)


def render_check_report(
    result: CheckResult,
    *,
    check_dir: Path,
    output_path: Path | None = None,
) -> str:
    """Render the evidence-first baseline/candidate release decision."""
    report_path = output_path or check_dir / "evidence.html"
    baseline_flat = _flatten(result.baseline.model_dump(mode="json"))
    candidate_flat = _flatten(result.candidate.model_dump(mode="json"))
    subject_diff = [
        {
            "field": key,
            "baseline": _display_value(baseline_flat.get(key)),
            "candidate": _display_value(candidate_flat.get(key)),
        }
        for key in sorted(set(baseline_flat) | set(candidate_flat))
        if baseline_flat.get(key) != candidate_flat.get(key)
    ]

    comparisons = [
        {
            "case_id": comparison.case_id,
            "required": comparison.required,
            "status": comparison.status.value,
            "baseline_validity": comparison.baseline_validity.value,
            "candidate_validity": comparison.candidate_validity.value,
            "metric": comparison.metric,
            "baseline_value": _display_value(comparison.baseline_value),
            "candidate_value": _display_value(comparison.candidate_value),
            "delta": _display_value(comparison.delta),
            "judge": json.dumps(
                result.case_judges.get(comparison.case_id, {}), sort_keys=True
            ),
            "reason": comparison.reason,
        }
        for comparison in result.comparisons
    ]

    pairs = []
    for comparison in result.comparisons:
        for pair in comparison.evidence_pairs:
            pairs.append(
                {
                    "case_id": pair.case_id,
                    "unit_id": pair.unit_id,
                    "status": comparison.status.value,
                    "baseline": [
                        {
                            "kind": artifact.kind,
                            "src": _check_artifact_src(
                                artifact.path, check_dir, report_path
                            ),
                        }
                        for artifact in pair.baseline_artifacts
                    ],
                    "candidate": [
                        {
                            "kind": artifact.kind,
                            "src": _check_artifact_src(
                                artifact.path, check_dir, report_path
                            ),
                        }
                        for artifact in pair.candidate_artifacts
                    ],
                }
            )

    html = _env().get_template("check.html.j2").render(
        result=result,
        subject_diff=subject_diff,
        comparisons=comparisons,
        pairs=pairs,
        gate_rule=json.dumps(result.gate_config, sort_keys=True),
        suite_version=result.suite_version,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html)
    return html


def collect_run_videos(
    manifest: RunManifest,
    *,
    runs_root: Path | None = None,
    report_path: Path | None = None,
    failures_only: bool = False,
) -> list[dict[str, str]]:
    """Build video gallery entries with paths relative to the HTML report."""
    root = runs_root or DEFAULT_RUNS_DIR
    d = run_dir(manifest.run_id, root)
    videos: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(src: str, caption: str) -> None:
        if src in seen:
            return
        seen.add(src)
        videos.append({"src": src, "caption": caption})

    # Prefer on-disk mp4s under runs/<id>/videos/
    disk_vids = sorted((d / "videos").rglob("*.mp4")) if (d / "videos").exists() else []
    if disk_vids:
        ep_by_name = {
            Path(ep.video_path).name: ep
            for ep in manifest.episodes
            if ep.video_path
        }
        for vid in disk_vids:
            ep = ep_by_name.get(vid.name)
            if failures_only and ep is not None and ep.success:
                continue
            caption = ep.episode_id if ep else vid.name
            if ep is not None:
                caption = f"{caption} ({'ok' if ep.success else 'fail'})"
            add(_rel_to_report(vid, report_path), caption)
        return videos

    # Fallback: manifest-relative paths
    for ep in manifest.episodes:
        if not ep.video_path:
            continue
        if failures_only and ep.success:
            continue
        p = Path(ep.video_path)
        if not p.is_absolute():
            p = d / ep.video_path
        if p.exists():
            add(_rel_to_report(p, report_path), f"{ep.episode_id} ({'ok' if ep.success else 'fail'})")
        else:
            add(ep.video_path, ep.episode_id)
    return videos


def render_diff_report(
    diff: DiffResult,
    *,
    v1: RunManifest | None = None,
    v2: RunManifest | None = None,
    output_path: Path | None = None,
    video_paths: list[tuple[str, str]] | None = None,
    runs_root: Path | None = None,
) -> str:
    gpu_hours = 0.0
    if v1:
        gpu_hours += v1.gpu_hours
    if v2:
        gpu_hours += v2.gpu_hours

    meta = [
        {"label": "v1 policy", "value": diff.v1_policy_id},
        {"label": "v2 policy", "value": diff.v2_policy_id},
        {"label": "Threshold", "value": f"{diff.threshold_pp:g} pp"},
        {"label": "Regressed", "value": str(diff.n_regressed)},
        {"label": "Fixed", "value": str(diff.n_fixed)},
        {"label": "Unchanged", "value": str(diff.n_unchanged)},
        {"label": "Not evaluated", "value": str(diff.n_not_evaluated)},
        {"label": "GPU-hours", "value": f"{gpu_hours:.3f}"},
    ]
    task_rows = [
        {
            "task": t.task,
            "suite": t.suite,
            "status": t.status.value,
            "v1": _fmt_pct(t.v1_pc_success),
            "v2": _fmt_pct(t.v2_pc_success),
            "delta": _fmt_pct(t.delta_pp),
            "reason": t.reason,
        }
        for t in diff.tasks
    ]
    if video_paths:
        videos = [{"src": src, "caption": cap} for src, cap in video_paths]
    else:
        videos = []
        # Prefer failure videos from v2 (candidate), then v1
        for m in (v2, v1):
            if m is None:
                continue
            videos.extend(
                collect_run_videos(
                    m, runs_root=runs_root, report_path=output_path, failures_only=False
                )
            )

    profiles = []
    clusters = []
    run_tasks = []
    if v2:
        profiles = [p.model_dump() for p in build_robustness_profiles(v2)]
        clusters = [c.model_dump() for c in cluster_failures(v2)]
        run_tasks = [t.model_dump() for t in v2.tasks]

    html = _env().get_template("report.html.j2").render(
        title=f"Diff {diff.v1_run_id} → {diff.v2_run_id}",
        verdict=diff.verdict.value,
        summary=diff.summary,
        meta=meta,
        task_rows=task_rows,
        run_tasks=run_tasks,
        profiles=profiles,
        clusters=clusters,
        videos=videos,
        gpu_hours=f"{gpu_hours:.3f}",
    )
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)
    return html


def render_run_report(
    manifest: RunManifest,
    *,
    output_path: Path | None = None,
    video_paths: list[tuple[str, str]] | None = None,
    runs_root: Path | None = None,
) -> str:
    meta = [
        {"label": "Policy", "value": manifest.policy_id},
        {"label": "Suite", "value": manifest.suite},
        {"label": "Seed", "value": str(manifest.seed)},
        {"label": "Episodes", "value": str(manifest.n_episodes)},
        {
            "label": "Overall success",
            "value": _fmt_pct(manifest.overall_pc_success),
        },
        {"label": "GPU-hours", "value": f"{manifest.gpu_hours:.3f}"},
        {"label": "Status", "value": manifest.status},
    ]
    profiles = [p.model_dump() for p in build_robustness_profiles(manifest)]
    clusters = [c.model_dump() for c in cluster_failures(manifest)]
    if video_paths:
        videos = [{"src": src, "caption": cap} for src, cap in video_paths]
    else:
        videos = collect_run_videos(
            manifest, runs_root=runs_root, report_path=output_path, failures_only=False
        )

    html = _env().get_template("report.html.j2").render(
        title=f"Run {manifest.run_id}",
        verdict=None,
        summary=None,
        meta=meta,
        task_rows=None,
        run_tasks=[t.model_dump() for t in manifest.tasks],
        profiles=profiles,
        clusters=clusters,
        videos=videos,
        gpu_hours=f"{manifest.gpu_hours:.3f}",
    )
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)
    return html
