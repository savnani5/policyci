"""File-backed run store.

All artifact I/O goes through this module so the backend can later swap to
Postgres + S3 without touching callers.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from policyci.schemas import CheckResult, DiffResult, RunManifest

DEFAULT_RUNS_DIR = Path("runs")


def _utcnow_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{_utcnow_slug()}_{uuid.uuid4().hex[:8]}"


def new_check_id() -> str:
    return new_run_id("check")


def check_dir(check_id: str, runs_root: Path | None = None) -> Path:
    root = runs_root or DEFAULT_RUNS_DIR
    return root / check_id


def run_dir(run_id: str, runs_root: Path | None = None) -> Path:
    root = runs_root or DEFAULT_RUNS_DIR
    return root / run_id


def save_run(manifest: RunManifest, runs_root: Path | None = None) -> Path:
    """Write manifest.json (and optionally eval_info.json) under runs/<run_id>/."""
    d = run_dir(manifest.run_id, runs_root)
    d.mkdir(parents=True, exist_ok=True)
    (d / "videos").mkdir(exist_ok=True)
    (d / "episodes").mkdir(exist_ok=True)

    manifest_path = d / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    if manifest.eval_info is not None:
        (d / "eval_info.json").write_text(json.dumps(manifest.eval_info, indent=2))

    # Per-episode records for RoboLens seam
    for ep in manifest.episodes:
        ep_path = d / "episodes" / f"{ep.episode_id}.json"
        ep_path.write_text(ep.model_dump_json(indent=2))

    return d


def load_run(run_id: str, runs_root: Path | None = None) -> RunManifest:
    d = run_dir(run_id, runs_root)
    manifest_path = d / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No run found at {manifest_path}")
    return RunManifest.model_validate_json(manifest_path.read_text())


def load_run_from_path(path: Path) -> RunManifest:
    """Load a run from an explicit directory path (or a path to manifest.json)."""
    p = path
    if p.is_file() and p.name == "manifest.json":
        p = p.parent
    manifest_path = p / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json at {p}")
    return RunManifest.model_validate_json(manifest_path.read_text())


def list_runs(runs_root: Path | None = None) -> list[str]:
    root = runs_root or DEFAULT_RUNS_DIR
    if not root.exists():
        return []
    ids = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "manifest.json").exists():
            ids.append(child.name)
    return ids


def save_diff(diff: DiffResult, runs_root: Path | None = None) -> Path:
    root = runs_root or DEFAULT_RUNS_DIR
    diff_id = f"diff_{diff.v1_run_id}_vs_{diff.v2_run_id}"
    d = root / diff_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / "diff.json"
    path.write_text(diff.model_dump_json(indent=2))
    return d


def load_diff(path: Path) -> DiffResult:
    p = path
    if p.is_dir():
        p = p / "diff.json"
    return DiffResult.model_validate_json(p.read_text())


def save_check_result(result: CheckResult, runs_root: Path | None = None) -> Path:
    directory = check_dir(result.check_id, runs_root)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(result.model_dump_json(indent=2))
    return directory


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def copy_videos_into_run(src_videos: Path, run_id: str, runs_root: Path | None = None) -> list[str]:
    """Copy video files into runs/<id>/videos/ and return relative paths."""
    d = run_dir(run_id, runs_root) / "videos"
    d.mkdir(parents=True, exist_ok=True)
    rel_paths: list[str] = []
    if not src_videos.exists():
        return rel_paths
    for src in sorted(src_videos.rglob("*.mp4")):
        dest = d / src.name
        shutil.copy2(src, dest)
        rel_paths.append(f"videos/{src.name}")
    return rel_paths
