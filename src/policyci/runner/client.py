"""Local helpers that call Modal and materialize runs/ on disk."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from policyci.schemas import EpisodeRecord, RunManifest, TaskResult
from policyci.store import DEFAULT_RUNS_DIR, new_run_id, run_dir, save_run


def _shard_to_manifest(result: dict[str, Any]) -> RunManifest:
    tasks = [TaskResult.model_validate(t) for t in result.get("tasks") or []]
    episodes = [EpisodeRecord.model_validate(e) for e in result.get("episodes") or []]
    return RunManifest(
        run_id=result["run_id"],
        policy_id=result["policy_id"],
        suite=result.get("suite", ""),
        seed=int(result.get("seed", 0)),
        n_episodes=int(result.get("n_episodes", 0)),
        task_ids=result.get("task_ids"),
        gpu_seconds=float(result.get("gpu_seconds", 0)),
        gpu_type=result.get("gpu_type", ""),
        status=result.get("status", "succeeded"),
        error=result.get("error"),
        tasks=tasks,
        overall_pc_success=result.get("overall_pc_success"),
        eval_info=result.get("eval_info"),
        episodes=episodes,
        notes=json.dumps(result.get("perturbation")) if result.get("perturbation") else "",
    )


def _to_relative_video_path(path: str | None) -> str | None:
    """Map absolute Volume paths to local relative videos/... paths."""
    if not path:
        return None
    p = path.replace("\\", "/")
    if p.startswith("videos/"):
        return p
    # /vol/runs/<run_id>/videos/... or .../videos/...
    m = re.search(r"(?:^|/)(videos/.+)$", p)
    if m:
        return m.group(1)
    name = Path(p).name
    if name.endswith(".mp4"):
        return f"videos/{name}"
    return None


def localize_manifest_video_paths(manifest: RunManifest) -> RunManifest:
    """Rewrite episode/task video paths to relative videos/... form."""
    for ep in manifest.episodes:
        ep.video_path = _to_relative_video_path(ep.video_path) or ep.video_path
    for task in manifest.tasks:
        task.video_paths = [
            _to_relative_video_path(p) or p for p in task.video_paths if p
        ]
    return manifest


def _write_pull_payload(dest: Path, payload: dict[str, Any]) -> Path:
    if not payload.get("ok"):
        raise FileNotFoundError(payload.get("error") or "pull failed")
    for rel, text in (payload.get("files") or {}).items():
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for rel, data in (payload.get("videos") or {}).items():
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return dest


def _attach_disk_videos(
    manifest: RunManifest,
    dest: Path,
    *,
    force_rewrite: bool = False,
) -> RunManifest:
    """Stamp relative videos/... paths onto episodes/tasks from on-disk mp4s."""
    local_vids = sorted((dest / "videos").rglob("*.mp4")) if (dest / "videos").exists() else []
    if not local_vids:
        return manifest
    for i, ep in enumerate(manifest.episodes):
        if (force_rewrite or not ep.video_path) and i < len(local_vids):
            ep.video_path = f"videos/{local_vids[i].relative_to(dest / 'videos').as_posix()}"
    if not manifest.episodes:
        for i, vid in enumerate(local_vids):
            manifest.episodes.append(
                EpisodeRecord(
                    episode_id=f"{manifest.suite}_ep{i}",
                    task=manifest.suite,
                    suite=manifest.suite,
                    seed=manifest.seed + i,
                    success=False,
                    policy_id=manifest.policy_id,
                    video_path=f"videos/{vid.relative_to(dest / 'videos').as_posix()}",
                )
            )
    for task in manifest.tasks:
        if force_rewrite or not task.video_paths:
            task.video_paths = [
                f"videos/{v.relative_to(dest / 'videos').as_posix()}" for v in local_vids
            ]
    return localize_manifest_video_paths(manifest)


def ensure_local_manifest(
    run_id: str,
    runs_root: Path | None = None,
) -> RunManifest:
    """Load or rebuild manifest.json after a Volume pull (from shard_result.json)."""
    import json

    root = runs_root or DEFAULT_RUNS_DIR
    dest = run_dir(run_id, root)
    manifest_path = dest / "manifest.json"
    if manifest_path.exists():
        from policyci.store import load_run

        manifest = load_run(run_id, root)
    else:
        shard_path = dest / "shard_result.json"
        if not shard_path.exists():
            raise FileNotFoundError(
                f"No manifest.json or shard_result.json under {dest}"
            )
        result = json.loads(shard_path.read_text())
        manifest = localize_manifest_video_paths(_shard_to_manifest(result))

    manifest = _attach_disk_videos(manifest, dest)
    save_run(manifest, root)
    return manifest


def pull_volume_run(
    run_id: str,
    runs_root: Path | None = None,
    *,
    use_existing_app: bool = False,
) -> Path:
    """Download a run's artifacts (JSON + mp4) from the Modal Volume into runs/<id>/.

    Prefer ``modal volume get`` (no image build). When already inside an active
    Modal app session (post-eval), optionally use ``fetch_run_artifacts.remote``.
    """
    root = runs_root or DEFAULT_RUNS_DIR
    dest = root / run_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "videos").mkdir(exist_ok=True)

    if use_existing_app:
        from policyci.runner.modal_app import fetch_run_artifacts

        payload = fetch_run_artifacts.remote(run_id)
        return _write_pull_payload(dest, payload)

    # Fast path: Modal CLI volume download (avoids spinning up the GPU image)
    from policyci.runner.modal_app import VOLUME_NAME

    import shutil
    import subprocess

    # modal volume get dumps folder *contents* into LOCAL_DEST when REMOTE is a dir
    # Use a staging dir so nested run_ids (sweep/shard_0000) land correctly.
    staging = root / f".pull_staging_{run_id.replace('/', '_')}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    cmd = [
        "modal",
        "volume",
        "get",
        VOLUME_NAME,
        run_id,
        str(staging),
        "--force",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        shutil.rmtree(staging, ignore_errors=True)
        raise FileNotFoundError(
            f"modal volume get failed for {run_id}: {proc.stderr.strip() or proc.stdout.strip()}"
        )

    # Staging may contain either files directly or a single run_id subdirectory
    payload_root = staging
    nested = staging / Path(run_id).name
    if nested.is_dir() and not any(staging.glob("*.json")):
        # Sometimes CLI nests under the basename
        if (nested / "manifest.json").exists() or (nested / "videos").exists() or list(nested.iterdir()):
            payload_root = nested
    # Another layout: staging/run_id/... when run_id has no slash
    alt = staging / run_id
    if alt.is_dir() and (alt / "videos").exists():
        payload_root = alt

    # Move/copy into dest
    for src in payload_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(payload_root)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)

    shutil.rmtree(staging, ignore_errors=True)

    if not any(dest.rglob("*")):
        raise FileNotFoundError(f"No artifacts downloaded for run {run_id}")
    return dest


def materialize_run_from_result(
    result: dict[str, Any],
    *,
    runs_root: Path | None = None,
    pull_videos: bool = True,
    use_existing_app: bool = False,  # retained for callers; pull always uses volume get
) -> Path:
    """Save manifest from an eval result and optionally pull videos from the Volume."""
    del use_existing_app  # volume get is preferred over RPC byte transfer
    root = runs_root or DEFAULT_RUNS_DIR
    manifest = localize_manifest_video_paths(_shard_to_manifest(result))
    save_run(manifest, root)
    dest = run_dir(manifest.run_id, root)

    if pull_videos and manifest.status == "succeeded":
        try:
            # Always use `modal volume get` — avoids shipping mp4 bytes through RPC
            # and avoids rebuilding the GPU image for a standalone pull.
            pull_volume_run(manifest.run_id, root, use_existing_app=False)
            manifest = _attach_disk_videos(manifest, dest)
            save_run(manifest, root)
        except Exception as e:
            # Keep JSON even if pull fails (e.g. volume lag)
            manifest.notes = (manifest.notes + f" | video_pull_error={e}").strip(" |")
            save_run(manifest, root)

    return dest


def run_eval(
    policy_path: str,
    *,
    suite: str = "libero_object",
    task_ids: list[int] | None = None,
    n_episodes: int = 10,
    seed: int = 1000,
    runs_root: Path | None = None,
    run_id: str | None = None,
    n_action_steps: int = 10,
    batch_size: int = 1,
    perturbation: dict[str, Any] | None = None,
    pull_videos: bool = True,
) -> RunManifest:
    """Fire one Modal worker, pull videos, save manifest under runs/."""
    from policyci.runner.modal_app import app, eval_checkpoint
    from policyci.store import load_run

    run_id = run_id or new_run_id()
    root = runs_root or DEFAULT_RUNS_DIR
    with app.run():
        result = eval_checkpoint.remote(
            policy_path=policy_path,
            suite=suite,
            task_ids=task_ids if task_ids is not None else [0],
            n_episodes=n_episodes,
            seed=seed,
            run_id=run_id,
            n_action_steps=n_action_steps,
            batch_size=batch_size,
            perturbation=perturbation,
        )
    # Pull outside app.run() via `modal volume get`
    materialize_run_from_result(
        result,
        runs_root=root,
        pull_videos=pull_videos,
        use_existing_app=False,
    )
    return load_run(result["run_id"], root)


def run_eval_fanout(
    policy_path: str,
    shards: list[dict[str, Any]],
    *,
    aggregate_run_id: str | None = None,
    runs_root: Path | None = None,
    pull_videos: bool = True,
) -> RunManifest:
    """Fan out scenario shards via Modal .map(), aggregate, pull videos."""
    from policyci.runner.modal_app import app, eval_shard

    aggregate_run_id = aggregate_run_id or new_run_id("sweep")
    prepared = []
    for i, shard in enumerate(shards):
        s = dict(shard)
        s["policy_path"] = policy_path
        s["run_id"] = f"{aggregate_run_id}/shard_{i:04d}"
        prepared.append(s)

    root = runs_root or DEFAULT_RUNS_DIR
    with app.run():
        results = list(eval_shard.map(prepared))
        if pull_videos:
            _pull_fanout_runs(results, root)

    all_tasks: list[TaskResult] = []
    all_episodes: list[EpisodeRecord] = []
    gpu_seconds = 0.0
    status = "succeeded"
    errors: list[str] = []
    suite = prepared[0].get("suite", "libero_object") if prepared else "libero_object"
    seed = prepared[0].get("seed", 0) if prepared else 0
    n_episodes = sum(int(s.get("n_episodes", 0)) for s in prepared)

    for r in results:
        gpu_seconds += float(r.get("gpu_seconds", 0))
        if r.get("status") != "succeeded":
            status = "failed"
            if r.get("error"):
                errors.append(str(r["error"]))
        all_tasks.extend(TaskResult.model_validate(t) for t in r.get("tasks") or [])
        all_episodes.extend(EpisodeRecord.model_validate(e) for e in r.get("episodes") or [])

    total_ep = sum(t.n_episodes for t in all_tasks) or 1
    total_ok = sum(t.n_successes for t in all_tasks)
    overall = 100.0 * total_ok / total_ep

    manifest = localize_manifest_video_paths(
        RunManifest(
            run_id=aggregate_run_id,
            policy_id=policy_path,
            suite=suite,
            seed=int(seed),
            n_episodes=n_episodes,
            gpu_seconds=gpu_seconds,
            gpu_type="L4",
            status=status,
            error="; ".join(errors) if errors else None,
            tasks=all_tasks,
            overall_pc_success=overall,
            episodes=all_episodes,
            notes=f"fanout n_shards={len(prepared)}",
        )
    )
    # Aggregate may not have its own videos dir; attach from shard dirs if present
    agg_dest = run_dir(aggregate_run_id, root)
    shard_vids: list[Path] = []
    for r in results:
        rid = r.get("run_id")
        if not rid:
            continue
        shard_dir = root / rid / "videos"
        if shard_dir.exists():
            shard_vids.extend(sorted(shard_dir.rglob("*.mp4")))
    if shard_vids:
        videos_dir = agg_dest / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(shard_vids):
            dest = videos_dir / f"shard_{i:04d}_{src.name}"
            if not dest.exists():
                dest.write_bytes(src.read_bytes())
        manifest = _attach_disk_videos(manifest, agg_dest, force_rewrite=True)

    save_run(manifest, root)
    return manifest


def _pull_fanout_runs(results: list[dict[str, Any]], root: Path) -> None:
    """Pull every materialized shard, including evidence from failed workers."""
    for result in results:
        run_id = result.get("run_id")
        if not run_id:
            continue
        try:
            pull_volume_run(run_id, root, use_existing_app=False)
        except Exception:
            # The aggregate manifest still records the worker failure. Missing
            # evidence is handled as NO_VERDICT by the check runner.
            pass
