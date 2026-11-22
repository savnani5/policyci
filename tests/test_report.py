"""HTML report renders from fixtures."""

from __future__ import annotations

from pathlib import Path

from policyci.diffgate import diff_runs
from policyci.report.html import collect_run_videos, render_diff_report, render_run_report
from policyci.runner.client import _to_relative_video_path, localize_manifest_video_paths
from policyci.schemas import EpisodeRecord, RunManifest, TaskResult
from policyci.store import save_run


def _m(run_id: str, policy: str, pc: float) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        policy_id=policy,
        suite="libero_object",
        seed=1000,
        n_episodes=10,
        status="succeeded",
        gpu_seconds=120.0,
        gpu_type="L4",
        tasks=[
            TaskResult(
                task="libero_object/0",
                task_id=0,
                suite="libero_object",
                n_episodes=10,
                n_successes=int(pc / 10),
                pc_success=pc,
            )
        ],
        overall_pc_success=pc,
    )


def test_render_diff_report(tmp_path: Path):
    v1 = _m("r1", "pol/v1", 95.0)
    v2 = _m("r2", "pol/v2", 40.0)
    d = diff_runs(v1, v2, threshold_pp=10.0, min_episodes=5)
    out = tmp_path / "report.html"
    html = render_diff_report(d, v1=v1, v2=v2, output_path=out)
    assert out.exists()
    assert "BLOCK" in html
    assert "regressed" in html
    assert "pol/v1" in html


def test_render_no_verdict_report(tmp_path: Path):
    v1 = _m("r1", "pol/v1", 95.0)
    v2 = _m("r2", "pol/v2", 40.0)
    v2.tasks[0].n_episodes = 2
    d = diff_runs(v1, v2, threshold_pp=10.0, min_episodes=5)
    html = render_diff_report(d, v1=v1, v2=v2)
    assert d.verdict.value == "no_verdict"
    assert "NO_VERDICT" in html
    assert "not_evaluated" in html


def test_render_run_report(tmp_path: Path):
    m = _m("r1", "pol/v1", 95.0)
    out = tmp_path / "run.html"
    html = render_run_report(m, output_path=out)
    assert "pol/v1" in html
    assert out.exists()


def test_relative_video_path_normalization():
    assert _to_relative_video_path("/vol/runs/abc/videos/ep0.mp4") == "videos/ep0.mp4"
    assert _to_relative_video_path("videos/ep0.mp4") == "videos/ep0.mp4"
    assert _to_relative_video_path(None) is None


def test_collect_run_videos_relative_to_report(tmp_path: Path):
    runs = tmp_path / "runs"
    m = _m("vidrun", "pol/v1", 50.0)
    m.episodes = [
        EpisodeRecord(
            episode_id="ep0",
            task="libero_object/0",
            task_id=0,
            suite="libero_object",
            seed=1000,
            success=False,
            policy_id="pol/v1",
            video_path="videos/ep0.mp4",
        )
    ]
    save_run(m, runs)
    vid = runs / "vidrun" / "videos" / "ep0.mp4"
    vid.write_bytes(b"fake-mp4")
    report = tmp_path / "out" / "report.html"
    report.parent.mkdir(parents=True)
    videos = collect_run_videos(m, runs_root=runs, report_path=report)
    assert len(videos) == 1
    assert videos[0]["src"] == "../runs/vidrun/videos/ep0.mp4"
    html = render_run_report(m, output_path=report, runs_root=runs)
    assert "Episode videos" in html
    assert "../runs/vidrun/videos/ep0.mp4" in html


def test_localize_manifest_video_paths():
    m = _m("r1", "pol/v1", 95.0)
    m.episodes = [
        EpisodeRecord(
            episode_id="ep0",
            task="libero_object/0",
            suite="libero_object",
            seed=1,
            success=True,
            policy_id="pol/v1",
            video_path="/vol/runs/r1/videos/cam.mp4",
        )
    ]
    m.tasks[0].video_paths = ["/vol/runs/r1/videos/cam.mp4"]
    out = localize_manifest_video_paths(m)
    assert out.episodes[0].video_path == "videos/cam.mp4"
    assert out.tasks[0].video_paths == ["videos/cam.mp4"]
